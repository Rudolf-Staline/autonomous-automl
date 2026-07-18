"""Evidence-backed explanation of the existing finalist selection rule."""

from __future__ import annotations

import math
from collections.abc import Sequence

from autonomous_automl.contracts import (
    MetricName,
    OptimizationProfile,
    PipelineSelectionExplanation,
    SelectionContextItem,
    TrialResult,
    TrialStatus,
)
from autonomous_automl.evaluation.selection import Leaderboard, LeaderboardEntry, select_finalist


def explain_pipeline_selection(
    leaderboard: Leaderboard,
    trials: Sequence[TrialResult],
    profile: OptimizationProfile | str,
    *,
    selected_trial_id: str,
    pipeline_size_bytes: int | None = None,
) -> PipelineSelectionExplanation:
    """Describe why the existing selector chose ``selected_trial_id``.

    This function is diagnostic only. It first invokes the production selector and
    refuses to emit an explanation if the supplied selected trial differs.
    """

    optimization_profile = OptimizationProfile(profile)
    require_full_fidelity = bool(leaderboard.finalists)
    selection = select_finalist(
        leaderboard,
        optimization_profile,
        require_full_fidelity=require_full_fidelity,
    )
    if selection.selected.trial_id != selected_trial_id:
        raise ValueError("persisted selected trial differs from the existing selection rule")
    candidates = leaderboard.finalists if require_full_fidelity else leaderboard.current_candidates
    ordered, primary_values = _selection_order(candidates, optimization_profile)
    if not ordered or ordered[0].trial_id != selected_trial_id:
        raise ValueError("diagnostic selection ordering differs from the production selector")

    selected = ordered[0]
    runner_up = ordered[1] if len(ordered) > 1 else None
    tie_breaker_used = runner_up is not None and math.isclose(
        primary_values[selected.trial_id],
        primary_values[runner_up.trial_id],
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    context = _supporting_context(
        selected,
        runner_up,
        trials,
        leaderboard.primary_metric,
        pipeline_size_bytes,
    )
    scope = (
        "eligible completed full-fidelity trials"
        if require_full_fidelity
        else "eligible completed trials at the highest available fidelity per PipelineSpec"
    )
    reason, rule = _reason_and_rule(optimization_profile, len(candidates), scope)
    tie_detail = _tie_breaker_detail(optimization_profile) if tie_breaker_used else None
    return PipelineSelectionExplanation(
        selected_trial_id=selected.trial_id,
        optimization_profile=optimization_profile,
        eligible_trial_count=len(candidates),
        selection_reason=reason,
        selection_rule=rule,
        tie_breaker_used=tie_breaker_used,
        tie_breaker_detail=tie_detail,
        runner_up_trial_id=None if runner_up is None else runner_up.trial_id,
        supporting_context=context,
    )


def _selection_order(
    candidates: tuple[LeaderboardEntry, ...],
    profile: OptimizationProfile,
) -> tuple[list[LeaderboardEntry], dict[str, float]]:
    if not candidates:
        raise ValueError("selection explanation requires at least one candidate")
    if profile is OptimizationProfile.ACCURACY:
        primary = {entry.trial_id: entry.adjusted_score for entry in candidates}
        return sorted(candidates, key=_accuracy_key), primary

    score_benefit = _benefit_values([entry.adjusted_score for entry in candidates])
    fit_benefit = _cost_benefit([entry.fit_seconds for entry in candidates])
    latency_benefit = _cost_benefit([entry.predict_seconds for entry in candidates])
    memory_benefit = _cost_benefit(_finite_memory_values(candidates))
    weights = (
        (0.60, 0.15, 0.15, 0.10)
        if profile is OptimizationProfile.BALANCED
        else (0.10, 0.35, 0.45, 0.10)
    )
    utility = {
        entry.trial_id: (
            weights[0] * score_benefit[index]
            + weights[1] * fit_benefit[index]
            + weights[2] * latency_benefit[index]
            + weights[3] * memory_benefit[index]
        )
        for index, entry in enumerate(candidates)
    }
    ordered = sorted(
        candidates,
        key=lambda entry: (
            -utility[entry.trial_id],
            -entry.adjusted_score,
            -entry.fidelity.level,
            -entry.mean_score,
            entry.std_score,
            entry.total_seconds,
            entry.predict_seconds,
            _memory_objective(entry),
            entry.trial_id,
        ),
    )
    return ordered, utility


def _reason_and_rule(
    profile: OptimizationProfile,
    candidate_count: int,
    scope: str,
) -> tuple[str, str]:
    if profile is OptimizationProfile.ACCURACY:
        return (
            f"Highest uncertainty-adjusted verified objective among {candidate_count} {scope}.",
            "Maximize mean internal validation score minus one fold standard deviation; "
            "then apply the existing deterministic fidelity, score, duration, memory, "
            "and trial-id tie-breaker.",
        )
    if profile is OptimizationProfile.BALANCED:
        return (
            f"Highest deterministic balanced utility among {candidate_count} {scope}.",
            "Normalize candidate evidence and maximize 60% uncertainty-adjusted score, "
            "15% fit-time efficiency, 15% prediction-time efficiency, and 10% memory "
            "efficiency; then apply the existing deterministic tie-breaker.",
        )
    return (
        f"Highest deterministic fast-profile utility among {candidate_count} {scope}.",
        "Normalize candidate evidence and maximize 10% uncertainty-adjusted score, "
        "35% fit-time efficiency, 45% prediction-time efficiency, and 10% memory "
        "efficiency; then apply the existing deterministic tie-breaker.",
    )


def _supporting_context(
    selected: LeaderboardEntry,
    runner_up: LeaderboardEntry | None,
    trials: Sequence[TrialResult],
    metric: MetricName | None,
    pipeline_size_bytes: int | None,
) -> list[SelectionContextItem]:
    selected_trial = next(
        (trial for trial in trials if trial.trial_id == selected.trial_id),
        None,
    )
    if selected_trial is None:
        raise ValueError("selected trial is absent from persisted trials")
    context = [
        SelectionContextItem(
            label="Mean verified metric",
            value=_user_metric_value(selected_trial, metric),
            provenance="SQLite trials.result_json / TrialResult.fold_results",
        ),
        SelectionContextItem(
            label="Fold standard deviation (internal objective)",
            value=selected.std_score,
            provenance="SQLite trials.std_score",
        ),
        SelectionContextItem(
            label="Leaderboard rank by mean verified score",
            value=selected.rank,
            provenance="leaderboard.csv and deterministic build_leaderboard ordering",
        ),
        SelectionContextItem(
            label="Fidelity level",
            value=selected.fidelity.level,
            provenance="SQLite trials.fidelity_json",
        ),
        SelectionContextItem(
            label="Evaluation duration seconds",
            value=selected.total_seconds,
            provenance="SQLite trials.fit_seconds + trials.predict_seconds",
        ),
        SelectionContextItem(
            label="Failed trials in selected family",
            value=sum(
                trial.status is TrialStatus.FAILED and trial.family == selected.family
                for trial in trials
            ),
            provenance="SQLite trials.status and trials.family",
        ),
    ]
    if pipeline_size_bytes is not None:
        context.append(
            SelectionContextItem(
                label="Serialized pipeline size bytes",
                value=pipeline_size_bytes,
                provenance="SQLite artifacts.size_bytes for best_pipeline",
            )
        )
    if runner_up is not None:
        runner_trial = next(
            (trial for trial in trials if trial.trial_id == runner_up.trial_id),
            None,
        )
        if runner_trial is not None:
            selected_value = _user_metric_value(selected_trial, metric)
            runner_value = _user_metric_value(runner_trial, metric)
            context.extend(
                [
                    SelectionContextItem(
                        label="Runner-up trial",
                        value=runner_up.trial_id,
                        provenance="existing deterministic selection rule applied to SQLite trials",
                    ),
                    SelectionContextItem(
                        label="Absolute verified metric difference from runner-up",
                        value=abs(selected_value - runner_value),
                        provenance="derived from persisted TrialResult fold metric values",
                    ),
                ]
            )
    return context


def _user_metric_value(result: TrialResult, metric: MetricName | None) -> float:
    if result.fold_results:
        return float(
            sum(fold.user_metric_value for fold in result.fold_results) / len(result.fold_results)
        )
    if result.mean_score is None or metric is None:
        raise ValueError("selected trial has no persisted user-facing metric value")
    return (
        -result.mean_score
        if metric in {MetricName.RMSE, MetricName.MAE, MetricName.LOG_LOSS}
        else result.mean_score
    )


def _accuracy_key(
    entry: LeaderboardEntry,
) -> tuple[float, float, int, float, float, float, float, str]:
    return (
        -entry.adjusted_score,
        -entry.mean_score,
        -entry.fidelity.level,
        entry.std_score,
        entry.total_seconds,
        entry.predict_seconds,
        _memory_objective(entry),
        entry.trial_id,
    )


def _tie_breaker_detail(profile: OptimizationProfile) -> str:
    if profile is OptimizationProfile.ACCURACY:
        return (
            "Equal uncertainty-adjusted objectives were resolved by mean score, fidelity, "
            "fold deviation, duration, prediction latency, memory, then trial id."
        )
    return (
        "Equal profile utilities were resolved by uncertainty-adjusted score, fidelity, "
        "mean score, fold deviation, duration, prediction latency, memory, then trial id."
    )


def _memory_objective(entry: LeaderboardEntry) -> float:
    return entry.peak_memory_mb if entry.peak_memory_mb is not None else math.inf


def _finite_memory_values(entries: Sequence[LeaderboardEntry]) -> list[float]:
    known = [entry.peak_memory_mb for entry in entries if entry.peak_memory_mb is not None]
    if not known:
        return [0.0] * len(entries)
    worst = max(known)
    unknown = worst + max(1.0, worst * 0.25)
    return [
        entry.peak_memory_mb if entry.peak_memory_mb is not None else unknown for entry in entries
    ]


def _benefit_values(values: Sequence[float]) -> list[float]:
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return [1.0] * len(values)
    return [(value - minimum) / (maximum - minimum) for value in values]


def _cost_benefit(values: Sequence[float]) -> list[float]:
    return [1.0 - value for value in _benefit_values(values)]


__all__ = ["explain_pipeline_selection"]

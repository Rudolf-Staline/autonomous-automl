"""Deterministic leaderboard, Pareto analysis, and finalist selection."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

import pandas as pd

from autonomous_automl.contracts import (
    FidelitySpec,
    MetricName,
    OptimizationProfile,
    PipelineSpec,
    TrialResult,
    TrialStatus,
)
from autonomous_automl.pipelines import pipeline_fingerprint

_FINALIST_LEVELS = {2, 3}


@dataclass(frozen=True, slots=True)
class LeaderboardEntry:
    """One successful trial represented with comparable internal objectives."""

    rank: int
    trial_id: str
    family: str
    model_name: str
    pipeline_fingerprint: str
    pipeline_spec: PipelineSpec
    fidelity: FidelitySpec
    mean_score: float
    std_score: float
    adjusted_score: float
    fit_seconds: float
    predict_seconds: float
    peak_memory_mb: float | None
    is_baseline: bool

    @property
    def total_seconds(self) -> float:
        """Return fit plus prediction cost."""

        return self.fit_seconds + self.predict_seconds

    @property
    def finalist_eligible(self) -> bool:
        """Return whether the trial used full or confirmation fidelity."""

        return self.fidelity.level in _FINALIST_LEVELS


@dataclass(frozen=True, slots=True)
class Leaderboard:
    """Stable successful-trial ordering with convenience projections."""

    entries: tuple[LeaderboardEntry, ...]
    uncertainty_weight: float
    primary_metric: MetricName | None

    @property
    def current_candidates(self) -> tuple[LeaderboardEntry, ...]:
        """Return the highest-fidelity result available per PipelineSpec.

        A higher fidelity supersedes a lower fidelity even when its observed score
        is lower. Duplicate results at the same fidelity are resolved by the stable
        leaderboard order, then by trial identifier.
        """

        representatives: dict[str, LeaderboardEntry] = {}
        for entry in self.entries:
            current = representatives.get(entry.pipeline_fingerprint)
            if current is None or _representative_key(entry) < _representative_key(current):
                representatives[entry.pipeline_fingerprint] = entry
        return tuple(
            sorted(representatives.values(), key=lambda entry: (entry.rank, entry.trial_id))
        )

    @property
    def finalists(self) -> tuple[LeaderboardEntry, ...]:
        """Return one current full/confirmation result per PipelineSpec."""

        return tuple(entry for entry in self.current_candidates if entry.finalist_eligible)

    @property
    def baselines(self) -> tuple[LeaderboardEntry, ...]:
        """Return identifiable Dummy/baseline entries."""

        return tuple(entry for entry in self.entries if entry.is_baseline)

    def to_frame(self) -> pd.DataFrame:
        """Return a deterministic tabular view without performing I/O."""

        columns = [
            "rank",
            "trial_id",
            "family",
            "model_name",
            "pipeline_fingerprint",
            "fidelity_level",
            "primary_metric",
            "mean_score",
            "std_score",
            "adjusted_score",
            "fit_seconds",
            "predict_seconds",
            "total_seconds",
            "peak_memory_mb",
            "is_baseline",
            "finalist_eligible",
        ]
        rows = [
            {
                "rank": entry.rank,
                "trial_id": entry.trial_id,
                "family": entry.family,
                "model_name": entry.model_name,
                "pipeline_fingerprint": entry.pipeline_fingerprint,
                "fidelity_level": entry.fidelity.level,
                "primary_metric": (
                    self.primary_metric.value if self.primary_metric is not None else None
                ),
                "mean_score": entry.mean_score,
                "std_score": entry.std_score,
                "adjusted_score": entry.adjusted_score,
                "fit_seconds": entry.fit_seconds,
                "predict_seconds": entry.predict_seconds,
                "total_seconds": entry.total_seconds,
                "peak_memory_mb": entry.peak_memory_mb,
                "is_baseline": entry.is_baseline,
                "finalist_eligible": entry.finalist_eligible,
            }
            for entry in self.entries
        ]
        return pd.DataFrame(rows, columns=columns)


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Selected finalist, Pareto analysis, and comparable baseline context."""

    profile: OptimizationProfile
    selected: LeaderboardEntry
    pareto_front: tuple[LeaderboardEntry, ...]
    baseline: LeaderboardEntry | None
    baseline_comparable: bool
    baseline_score_delta: float | None

    @property
    def beats_baseline(self) -> bool | None:
        """Compare internal scores when a full-fidelity baseline exists."""

        if self.baseline_score_delta is None:
            return None
        return self.baseline_score_delta > 0.0


def build_leaderboard(
    trials: Iterable[TrialResult],
    *,
    uncertainty_weight: float = 1.0,
) -> Leaderboard:
    """Build a stable leaderboard from completed trials only.

    Scores are already in the internal greater-is-better orientation. No metric
    sign conversion is performed here.
    """

    if not math.isfinite(uncertainty_weight) or uncertainty_weight < 0.0:
        raise ValueError("uncertainty_weight must be finite and non-negative")
    entries: list[LeaderboardEntry] = []
    seen_ids: set[str] = set()
    primary_metric: MetricName | None = None
    for trial in trials:
        if trial.status is not TrialStatus.COMPLETED:
            continue
        if trial.trial_id in seen_ids:
            raise ValueError(f"duplicate completed trial id: {trial.trial_id}")
        seen_ids.add(trial.trial_id)
        if trial.mean_score is None or trial.std_score is None:
            raise ValueError("completed TrialResult lacks aggregate scores")
        if primary_metric is None:
            primary_metric = trial.primary_metric
        elif trial.primary_metric is not primary_metric:
            raise ValueError(
                "completed trials must use one primary_metric; "
                f"found {primary_metric.value!r} and {trial.primary_metric.value!r}"
            )
        entries.append(
            LeaderboardEntry(
                rank=0,
                trial_id=trial.trial_id,
                family=trial.family,
                model_name=trial.pipeline_spec.model_name,
                pipeline_fingerprint=pipeline_fingerprint(trial.pipeline_spec),
                pipeline_spec=trial.pipeline_spec,
                fidelity=trial.fidelity,
                mean_score=trial.mean_score,
                std_score=trial.std_score,
                adjusted_score=trial.mean_score - uncertainty_weight * trial.std_score,
                fit_seconds=trial.fit_seconds,
                predict_seconds=trial.predict_seconds,
                peak_memory_mb=trial.peak_memory_mb,
                is_baseline=(
                    trial.family.casefold() == "baseline"
                    or trial.pipeline_spec.model_name.casefold() == "dummy"
                ),
            )
        )
    entries.sort(key=_leaderboard_key)
    ranked = tuple(replace(entry, rank=index) for index, entry in enumerate(entries, start=1))
    return Leaderboard(
        entries=ranked,
        uncertainty_weight=uncertainty_weight,
        primary_metric=primary_metric,
    )


def pareto_front(entries: Sequence[LeaderboardEntry]) -> tuple[LeaderboardEntry, ...]:
    """Return non-dominated entries for score, fit cost, latency, and memory."""

    front = [
        candidate
        for candidate in entries
        if not any(
            _dominates(other, candidate)
            for other in entries
            if other.trial_id != candidate.trial_id
        )
    ]
    return tuple(sorted(front, key=lambda entry: entry.rank))


def select_finalist(
    leaderboard: Leaderboard,
    profile: OptimizationProfile | str,
    *,
    require_full_fidelity: bool = True,
) -> SelectionResult:
    """Select one full-fidelity candidate according to an explicit profile.

    The raw-score Pareto front is returned for diagnostics. Selection considers
    every eligible finalist because the uncertainty-adjusted score is an
    intentional additional objective: a lower-variance candidate can therefore
    be preferable even when its raw mean score is dominated.
    """

    optimization_profile = OptimizationProfile(profile)
    candidates = leaderboard.finalists if require_full_fidelity else leaderboard.current_candidates
    if not candidates:
        raise ValueError("finalist selection requires a completed full-fidelity trial")
    front = pareto_front(candidates)
    selected = _select_from_candidates(candidates, optimization_profile)
    baseline = _best_baseline(leaderboard.baselines)
    comparable = baseline is not None and baseline.finalist_eligible
    delta = selected.mean_score - baseline.mean_score if comparable and baseline else None
    return SelectionResult(
        profile=optimization_profile,
        selected=selected,
        pareto_front=front,
        baseline=baseline,
        baseline_comparable=comparable,
        baseline_score_delta=delta,
    )


def _leaderboard_key(entry: LeaderboardEntry) -> tuple[float, float, float, float, str]:
    return (
        -entry.mean_score,
        entry.std_score,
        entry.total_seconds,
        _memory_objective(entry),
        entry.trial_id,
    )


def _representative_key(entry: LeaderboardEntry) -> tuple[int, int, str]:
    return (-entry.fidelity.level, entry.rank, entry.trial_id)


def _dominates(left: LeaderboardEntry, right: LeaderboardEntry) -> bool:
    left_objectives = (
        left.mean_score,
        -left.fit_seconds,
        -left.predict_seconds,
        -_memory_objective(left),
    )
    right_objectives = (
        right.mean_score,
        -right.fit_seconds,
        -right.predict_seconds,
        -_memory_objective(right),
    )
    return all(a >= b for a, b in zip(left_objectives, right_objectives, strict=True)) and any(
        a > b for a, b in zip(left_objectives, right_objectives, strict=True)
    )


def _select_from_candidates(
    candidates: tuple[LeaderboardEntry, ...],
    profile: OptimizationProfile,
) -> LeaderboardEntry:
    if profile is OptimizationProfile.ACCURACY:
        return min(candidates, key=_accuracy_key)

    score_benefit = _benefit_values([entry.adjusted_score for entry in candidates])
    fit_benefit = _cost_benefit([entry.fit_seconds for entry in candidates])
    latency_benefit = _cost_benefit([entry.predict_seconds for entry in candidates])
    memory_values = _finite_memory_values(candidates)
    memory_benefit = _cost_benefit(memory_values)
    weights = (
        (0.60, 0.15, 0.15, 0.10)
        if profile is OptimizationProfile.BALANCED
        else (0.10, 0.35, 0.45, 0.10)
    )
    utilities = {
        entry.trial_id: (
            weights[0] * score_benefit[index]
            + weights[1] * fit_benefit[index]
            + weights[2] * latency_benefit[index]
            + weights[3] * memory_benefit[index]
        )
        for index, entry in enumerate(candidates)
    }
    return min(
        candidates,
        key=lambda entry: (
            -utilities[entry.trial_id],
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


def _best_baseline(entries: Sequence[LeaderboardEntry]) -> LeaderboardEntry | None:
    if not entries:
        return None
    return min(
        entries,
        key=lambda entry: (
            -int(entry.finalist_eligible),
            -entry.fidelity.level,
            -entry.mean_score,
            entry.std_score,
            entry.trial_id,
        ),
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
    normalized = _benefit_values(values)
    return [1.0 - value for value in normalized]


__all__ = [
    "Leaderboard",
    "LeaderboardEntry",
    "SelectionResult",
    "build_leaderboard",
    "pareto_front",
    "select_finalist",
]

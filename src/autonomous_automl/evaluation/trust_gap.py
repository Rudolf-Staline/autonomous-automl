"""Strictly post-selection Observed Trust Gap diagnostic."""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

from autonomous_automl.components import ModelRegistry
from autonomous_automl.contracts import (
    DatasetProfile,
    LeakageReport,
    MetricDirection,
    MetricName,
    ObservedTrustGap,
    TrialResult,
    TrialStatus,
    TrustGapProtocol,
    TrustGapStatus,
    ValidationPlan,
)
from autonomous_automl.data import LoadedDataset
from autonomous_automl.evaluation.evaluator import PipelineEvaluator
from autonomous_automl.evaluation.metrics import resolve_metric

RAW_PROTOCOL_PATH = "trust_gap/raw_protocol.json"
RAW_PREDICTIONS_PATH = "trust_gap/raw_oof_predictions.csv"
TRUST_GAP_RESULT_PATH = "trust_gap/result.json"


@dataclass(frozen=True, slots=True)
class TrustGapEvaluation:
    """Diagnostic contract plus optional artifacts that are safe to persist."""

    result: ObservedTrustGap
    raw_protocol: TrustGapProtocol | None
    raw_predictions: pd.DataFrame | None


def metric_direction(metric: MetricName) -> MetricDirection:
    """Return the user-facing direction for a resolved metric."""

    if metric in {MetricName.LOG_LOSS, MetricName.RMSE, MetricName.MAE}:
        return MetricDirection.MINIMIZE
    return MetricDirection.MAXIMIZE


def compute_observed_trust_gap(
    *,
    selected_trial: TrialResult,
    dataset: LoadedDataset,
    profile: DatasetProfile,
    leakage_report: LeakageReport,
    validation_plan: ValidationPlan,
    metric: MetricName,
    timeout_seconds: int,
    n_jobs: int = 1,
    registry: ModelRegistry | None = None,
) -> TrustGapEvaluation:
    """Evaluate a raw-feature diagnostic without touching search state.

    The diagnostic keeps the selected algorithm, parameters, fidelity and persisted
    folds. It restores only risk columns that are both recorded as excluded and still
    available in the loaded training frame. The result is never written to the trial
    registry by this service.
    """

    started = time.monotonic()
    direction = metric_direction(metric)
    unavailable = _precondition_failure(
        selected_trial,
        dataset,
        validation_plan,
        metric,
        timeout_seconds,
    )
    if unavailable is not None:
        return TrustGapEvaluation(
            result=_not_computed(metric, direction, unavailable, time.monotonic() - started),
            raw_protocol=None,
            raw_predictions=None,
        )

    restorable = [
        column
        for column in selected_trial.pipeline_spec.excluded_columns
        if column in leakage_report.excluded_columns and column in dataset.X.columns
    ]
    if not restorable:
        return TrustGapEvaluation(
            result=_not_computed(
                metric,
                direction,
                "No excluded risk column is available for a comparable raw-feature evaluation.",
                time.monotonic() - started,
            ),
            raw_protocol=None,
            raw_predictions=None,
        )
    unsupported_text = sorted(set(restorable).intersection(profile.text_columns))
    if unsupported_text:
        return TrustGapEvaluation(
            result=_not_computed(
                metric,
                direction,
                "The selected pipeline has no executable transformer for restored free-text "
                f"risk columns: {unsupported_text}.",
                time.monotonic() - started,
            ),
            raw_protocol=None,
            raw_predictions=None,
        )

    remaining_exclusions = [
        column
        for column in selected_trial.pipeline_spec.excluded_columns
        if column not in restorable
    ]
    raw_spec = selected_trial.pipeline_spec.model_copy(
        update={"excluded_columns": remaining_exclusions},
        deep=True,
    )
    diagnostic_profile = profile.model_copy(
        update={
            "id_candidates": [
                column for column in profile.id_candidates if column not in restorable
            ]
        },
        deep=True,
    )
    execution_seeds = [fold.seed for fold in selected_trial.fold_results if fold.seed is not None]
    protocol = TrustGapProtocol(
        protocol_name="raw_feature_inclusion_with_persisted_folds",
        description=(
            "Rebuild the selected algorithm and hyperparameters, restore only recorded "
            "risk columns available in the training frame, and evaluate on the same "
            "persisted folds and fidelity as the verified selected trial."
        ),
        metric=metric,
        metric_direction=direction,
        pipeline_spec=raw_spec,
        fidelity=selected_trial.fidelity,
        validation_plan=validation_plan,
        feature_columns=list(dataset.X.columns),
        excluded_columns=remaining_exclusions,
        restored_columns=restorable,
        execution_seeds=execution_seeds,
        timeout_seconds=timeout_seconds,
    )
    outcome = PipelineEvaluator(registry, n_jobs=n_jobs).evaluate(
        "trust-gap-raw-diagnostic",
        raw_spec,
        selected_trial.fidelity,
        dataset,
        diagnostic_profile,
        validation_plan,
        metric,
        timeout_seconds=float(timeout_seconds),
        execution_seeds=execution_seeds,
    )
    elapsed = time.monotonic() - started
    if outcome.trial_result.status is not TrialStatus.COMPLETED:
        failure_type = outcome.trial_result.failure_type or "DiagnosticEvaluationError"
        failure_message = outcome.trial_result.failure_message
        failure_detail = failure_type
        if failure_message:
            failure_detail = f"{failure_type}: {failure_message}"
        return TrustGapEvaluation(
            result=_not_computed(
                metric,
                direction,
                f"Raw diagnostic evaluation did not complete ({failure_detail}).",
                elapsed,
                raw_protocol=protocol.protocol_name,
                differing_columns=restorable,
                raw_protocol_path=RAW_PROTOCOL_PATH,
            ),
            raw_protocol=protocol,
            raw_predictions=None,
        )
    if outcome.oof_predictions is None:
        return TrustGapEvaluation(
            result=_not_computed(
                metric,
                direction,
                "Raw diagnostic evaluation produced no reproducible OOF predictions.",
                elapsed,
            ),
            raw_protocol=protocol,
            raw_predictions=None,
        )

    raw_score = _user_metric_value(outcome.trial_result, metric)
    verified_score = _user_metric_value(selected_trial, metric)
    gap = (
        raw_score - verified_score
        if direction is MetricDirection.MAXIMIZE
        else verified_score - raw_score
    )
    result = ObservedTrustGap(
        status=TrustGapStatus.COMPUTED,
        reason=(
            "Comparable post-selection diagnostic completed with the persisted folds; "
            "only recorded risk-column inclusion differs."
        ),
        metric=metric,
        metric_direction=direction,
        raw_score=raw_score,
        verified_score=verified_score,
        observed_trust_gap=gap,
        raw_protocol="raw_feature_inclusion_with_persisted_folds",
        verified_protocol="persisted_leakage_neutralized_selected_trial",
        differing_columns=restorable,
        splitter_different=False,
        warnings=[
            "This is an observed protocol-specific comparison, not a universal causal estimate.",
            "The raw diagnostic intentionally permits only recorded risk columns through "
            "the compatibility exclusion gate; search compatibility rules remain unchanged.",
        ],
        diagnostic_elapsed_seconds=elapsed,
        raw_protocol_path=RAW_PROTOCOL_PATH,
        raw_predictions_path=RAW_PREDICTIONS_PATH,
    )
    return TrustGapEvaluation(
        result=result,
        raw_protocol=protocol,
        raw_predictions=outcome.oof_predictions,
    )


def disabled_trust_gap(metric: MetricName) -> ObservedTrustGap:
    """Return explicit evidence when the optional diagnostic was not requested."""

    return _not_computed(
        metric,
        metric_direction(metric),
        "Observed Trust Gap was disabled by configuration.",
        0.0,
    )


def _precondition_failure(
    selected_trial: TrialResult,
    dataset: LoadedDataset,
    validation_plan: ValidationPlan,
    metric: MetricName,
    timeout_seconds: int,
) -> str | None:
    if timeout_seconds <= 0:
        return "Diagnostic timeout must be positive."
    if selected_trial.status is not TrialStatus.COMPLETED:
        return "The selected trial is not completed."
    if selected_trial.primary_metric is not metric:
        return "The selected and diagnostic metrics are not comparable."
    if not selected_trial.fold_results or selected_trial.mean_score is None:
        return "The selected trial has no persisted fold evidence."
    if any(fold.seed is None for fold in selected_trial.fold_results):
        return "The selected trial has no complete persisted execution-seed evidence."
    if selected_trial.fidelity.sample_fraction != 1.0:
        return "The selected trial did not use all available training rows."
    if selected_trial.fidelity.n_folds != validation_plan.n_splits:
        return "The selected trial and persisted validation plan cover different fold counts."
    if not validation_plan.folds:
        return "The persisted validation plan has no materialized folds."
    if len(dataset.X) != profile_row_count(dataset):
        return "The loaded feature frame is inconsistent with its persisted dataset bundle."
    try:
        resolve_metric(metric, selected_trial.pipeline_spec.task)
    except Exception:
        return "The persisted metric cannot be reconstructed for the selected task."
    return None


def profile_row_count(dataset: LoadedDataset) -> int:
    """Return the persisted row count used for a non-inferential consistency check."""

    return dataset.bundle.n_rows


def _not_computed(
    metric: MetricName,
    direction: MetricDirection,
    reason: str,
    elapsed: float,
    *,
    raw_protocol: str = "not_available",
    differing_columns: list[str] | None = None,
    raw_protocol_path: str | None = None,
) -> ObservedTrustGap:
    return ObservedTrustGap(
        status=TrustGapStatus.NOT_COMPUTED,
        reason=reason,
        metric=metric,
        metric_direction=direction,
        raw_protocol=raw_protocol,
        verified_protocol="persisted_leakage_neutralized_selected_trial",
        differing_columns=[] if differing_columns is None else differing_columns,
        splitter_different=None if raw_protocol == "not_available" else False,
        diagnostic_elapsed_seconds=max(0.0, elapsed),
        raw_protocol_path=raw_protocol_path,
    )


def _user_metric_value(result: TrialResult, metric: MetricName) -> float:
    if result.fold_results:
        return float(
            sum(fold.user_metric_value for fold in result.fold_results) / len(result.fold_results)
        )
    if result.mean_score is None:
        raise ValueError("completed diagnostic result has no aggregate score")
    return (
        -result.mean_score
        if metric in {MetricName.RMSE, MetricName.MAE, MetricName.LOG_LOSS}
        else result.mean_score
    )


__all__ = [
    "RAW_PREDICTIONS_PATH",
    "RAW_PROTOCOL_PATH",
    "TRUST_GAP_RESULT_PATH",
    "TrustGapEvaluation",
    "compute_observed_trust_gap",
    "disabled_trust_gap",
    "metric_direction",
]

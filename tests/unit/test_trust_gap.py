"""Observed Trust Gap is reproducible, post-selection, and non-authoritative."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from autonomous_automl.contracts import (
    AutoMLConfig,
    DatasetProfile,
    FidelitySpec,
    LeakageReport,
    MetricDirection,
    MetricName,
    PipelineSpec,
    TrialResult,
    TrialStatus,
    TrustGapStatus,
    ValidationPlan,
)
from autonomous_automl.data import LoadedDataset, load_dataset
from autonomous_automl.evaluation import (
    EvaluationOutcome,
    PipelineEvaluator,
    compute_observed_trust_gap,
)
from autonomous_automl.evaluation.selection import build_leaderboard, select_finalist
from autonomous_automl.profiling import detect_leakage, profile_dataset
from autonomous_automl.validation import ValidationPlanner


@dataclass(frozen=True, slots=True)
class Scenario:
    dataset: LoadedDataset
    profile: DatasetProfile
    leakage: LeakageReport
    plan: ValidationPlan
    selected: TrialResult
    metric: MetricName


def _make_scenario(directory: Path, *, regression: bool) -> Scenario:
    rng = np.random.default_rng(818 if regression else 717)
    rows = 100
    signal = rng.normal(size=rows)
    noise = rng.normal(size=rows)
    if regression:
        target = 3.0 * signal + rng.normal(0, 2.5, size=rows)
        task = "regression"
        metric = MetricName.RMSE
        model_name = "ridge"
        model_params: dict[str, object] = {"alpha": 1.0}
    else:
        latent = 0.35 * signal + noise
        target = (latent > np.median(latent)).astype(int)
        task = "binary_classification"
        metric = MetricName.ROC_AUC
        model_name = "logistic_regression"
        model_params = {"C": 1.0, "class_weight": None, "max_iter": 100}
    frame = pd.DataFrame(
        {
            "customer_id": [f"customer-{index:04d}" for index in range(rows)],
            "signal": signal,
            "noise": noise,
            "target_copy": target,
            "target": target,
        }
    )
    path = directory / ("regression.csv" if regression else "classification.csv")
    frame.to_csv(path, index=False)
    config = AutoMLConfig(
        target="target",
        task=task,
        metric=metric,
        budget_seconds=10,
        output_dir=directory / "unused-run",
    )
    dataset = load_dataset(path, config)
    profile = profile_dataset(dataset, config.task)
    leakage = detect_leakage(dataset, profile)
    plan = ValidationPlanner(max_splits=2).plan(dataset, profile, config)
    spec = PipelineSpec(
        family="linear",
        task=task,
        numeric_imputer="median",
        numeric_scaler="standard",
        categorical_imputer="constant",
        categorical_encoder="frequency",
        datetime_transformer="drop",
        model_name=model_name,
        model_params=model_params,
        excluded_columns=leakage.excluded_columns,
        random_seed=42,
    )
    fidelity = FidelitySpec(
        level=2,
        sample_fraction=1.0,
        n_folds=2,
        max_iterations=100,
        seeds=[42],
    )
    verified = PipelineEvaluator().evaluate(
        "selected-verified",
        spec,
        fidelity,
        dataset,
        profile,
        plan,
        metric,
    )
    assert verified.trial_result.status is TrialStatus.COMPLETED
    return Scenario(dataset, profile, leakage, plan, verified.trial_result, metric)


@pytest.fixture(scope="module")
def classification_scenario(tmp_path_factory: pytest.TempPathFactory) -> Scenario:
    return _make_scenario(tmp_path_factory.mktemp("trust-gap-classification"), regression=False)


@pytest.fixture(scope="module")
def regression_scenario(tmp_path_factory: pytest.TempPathFactory) -> Scenario:
    return _make_scenario(tmp_path_factory.mktemp("trust-gap-regression"), regression=True)


def _compute(scenario: Scenario, *, timeout: int = 5):
    return compute_observed_trust_gap(
        selected_trial=scenario.selected,
        dataset=scenario.dataset,
        profile=scenario.profile,
        leakage_report=scenario.leakage,
        validation_plan=scenario.plan,
        metric=scenario.metric,
        timeout_seconds=timeout,
    )


def test_maximize_metric_reports_positive_apparent_inflation(
    classification_scenario: Scenario,
) -> None:
    evaluation = _compute(classification_scenario)
    result = evaluation.result

    assert result.status is TrustGapStatus.COMPUTED
    assert result.metric_direction is MetricDirection.MAXIMIZE
    assert result.raw_score == pytest.approx(1.0)
    assert result.verified_score is not None
    assert result.observed_trust_gap is not None
    assert result.observed_trust_gap > 0
    assert result.observed_trust_gap == pytest.approx(result.raw_score - result.verified_score)


def test_minimize_metric_uses_verified_minus_raw_formula(regression_scenario: Scenario) -> None:
    result = _compute(regression_scenario).result

    assert result.status is TrustGapStatus.COMPUTED
    assert result.metric_direction is MetricDirection.MINIMIZE
    assert result.raw_score is not None
    assert result.verified_score is not None
    assert result.observed_trust_gap == pytest.approx(result.verified_score - result.raw_score)
    assert result.observed_trust_gap > 0


def test_diagnostic_reuses_persisted_folds_and_exact_execution_seeds(
    classification_scenario: Scenario,
) -> None:
    evaluation = _compute(classification_scenario)
    protocol = evaluation.raw_protocol

    assert protocol is not None
    assert protocol.validation_plan == classification_scenario.plan
    assert protocol.execution_seeds == [
        fold.seed for fold in classification_scenario.selected.fold_results
    ]
    assert not protocol.pipeline_spec.excluded_columns
    assert {"customer_id", "target_copy"}.issubset(protocol.restored_columns)


def test_diagnostic_is_reproducible_except_for_observed_wall_time(
    classification_scenario: Scenario,
) -> None:
    first = _compute(classification_scenario)
    second = _compute(classification_scenario)

    first_value = first.result.model_copy(update={"diagnostic_elapsed_seconds": 0})
    second_value = second.result.model_copy(update={"diagnostic_elapsed_seconds": 0})
    assert first_value == second_value
    assert first.raw_protocol == second.raw_protocol
    assert first.raw_predictions is not None
    assert second.raw_predictions is not None
    assert_frame_equal(first.raw_predictions, second.raw_predictions, check_exact=True)


def test_diagnostic_never_changes_selected_spec_or_main_leaderboard(
    classification_scenario: Scenario,
) -> None:
    before_json = classification_scenario.selected.to_json()
    before_board = build_leaderboard([classification_scenario.selected])
    before_selection = select_finalist(before_board, "accuracy").selected

    _compute(classification_scenario)

    after_board = build_leaderboard([classification_scenario.selected])
    after_selection = select_finalist(after_board, "accuracy").selected
    assert classification_scenario.selected.to_json() == before_json
    assert after_board == before_board
    assert after_selection.pipeline_spec == before_selection.pipeline_spec


def test_no_compatible_risk_column_returns_explicit_not_computed(
    classification_scenario: Scenario,
) -> None:
    result = compute_observed_trust_gap(
        selected_trial=classification_scenario.selected,
        dataset=classification_scenario.dataset,
        profile=classification_scenario.profile,
        leakage_report=LeakageReport(),
        validation_plan=classification_scenario.plan,
        metric=classification_scenario.metric,
        timeout_seconds=5,
    ).result

    assert result.status is TrustGapStatus.NOT_COMPUTED
    assert result.raw_score is None
    assert "No excluded risk column" in result.reason


def test_incompatible_metric_returns_explicit_not_computed(
    classification_scenario: Scenario,
) -> None:
    result = compute_observed_trust_gap(
        selected_trial=classification_scenario.selected,
        dataset=classification_scenario.dataset,
        profile=classification_scenario.profile,
        leakage_report=classification_scenario.leakage,
        validation_plan=classification_scenario.plan,
        metric=MetricName.ACCURACY,
        timeout_seconds=5,
    ).result

    assert result.status is TrustGapStatus.NOT_COMPUTED
    assert "not comparable" in result.reason


def test_diagnostic_timeout_is_contained_as_not_computed(
    classification_scenario: Scenario,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timed_out(
        _self: PipelineEvaluator,
        trial_id: str,
        pipeline_spec: PipelineSpec,
        fidelity: FidelitySpec,
        *_args: object,
        **_kwargs: object,
    ) -> EvaluationOutcome:
        now = datetime.now(UTC)
        return EvaluationOutcome(
            TrialResult(
                trial_id=trial_id,
                family=pipeline_spec.family,
                pipeline_spec=pipeline_spec,
                fidelity=fidelity,
                status=TrialStatus.FAILED,
                primary_metric=classification_scenario.metric,
                failure_type="TrialTimeoutError",
                failure_message="diagnostic timeout reached",
                started_at=now,
                finished_at=now,
            ),
            None,
        )

    monkeypatch.setattr(PipelineEvaluator, "evaluate", timed_out)

    evaluation = _compute(classification_scenario, timeout=1)

    assert evaluation.result.status is TrustGapStatus.NOT_COMPUTED
    assert "TrialTimeoutError" in evaluation.result.reason
    assert evaluation.raw_protocol is not None
    assert evaluation.raw_predictions is None

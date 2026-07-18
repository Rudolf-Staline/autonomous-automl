"""M1 contract validation and persistence tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from autonomous_automl.contracts import (
    AutoMLConfig,
    DatasetBundle,
    DatasetProfile,
    FidelitySpec,
    FindingSeverity,
    FoldAssignment,
    FoldResult,
    LeakageFinding,
    LeakageFindingType,
    LeakageReport,
    MetricName,
    NotebookValidation,
    NotebookValidationStatus,
    PipelineSpec,
    RunManifest,
    RunResult,
    RunStatus,
    TaskType,
    TrialResult,
    TrialStatus,
    ValidationPlan,
)
from autonomous_automl.contracts.base import ContractModel

DATASET_HASH = "a" * 64
TEST_HASH = "b" * 64
STARTED_AT = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
FINISHED_AT = STARTED_AT + timedelta(minutes=1)


def make_config(**updates: Any) -> AutoMLConfig:
    values: dict[str, Any] = {
        "target": "target",
        "task": "binary_classification",
        "metric": "roc_auc",
        "budget_seconds": 60,
        "random_seed": 42,
        "n_jobs": 2,
        "output_dir": "runs/unit",
        "group_column": "group_id",
        "time_column": "event_time",
        "id_column": "row_id",
        "predefined_fold_column": "fold_id",
        "optimization_profile": "balanced",
        "memory_limit_mb": 512,
        "trial_timeout_seconds": 15,
    }
    values.update(updates)
    return AutoMLConfig.model_validate(values)


def make_dataset_bundle(**updates: Any) -> DatasetBundle:
    values: dict[str, Any] = {
        "train_paths": ["examples/train-a.csv", "examples/train-b.csv"],
        "test_path": "examples/test.csv",
        "target": "target",
        "feature_columns": ["amount", "city", "active"],
        "excluded_columns": ["leaked_target", "row_id"],
        "n_rows": 12,
        "n_test_rows": 4,
        "row_id_column": "row_id",
        "source_hashes": {
            "examples/train-a.csv": DATASET_HASH,
            "examples/train-b.csv": TEST_HASH,
            "examples/test.csv": "c" * 64,
        },
    }
    values.update(updates)
    return DatasetBundle.model_validate(values)


def make_dataset_profile(**updates: Any) -> DatasetProfile:
    values: dict[str, Any] = {
        "dataset_hash": DATASET_HASH,
        "n_rows": 12,
        "n_features": 3,
        "inferred_task": "binary_classification",
        "inferred_types": {
            "amount": "numeric",
            "city": "categorical",
            "active": "boolean",
        },
        "numeric_columns": ["amount"],
        "categorical_columns": ["city"],
        "boolean_columns": ["active"],
        "missing_ratios": {"amount": 0.25, "city": 0.0, "active": 0.0},
        "cardinalities": {"amount": 9, "city": 3, "active": 2},
        "unique_ratios": {"amount": 0.75, "city": 0.25, "active": 2 / 12},
        "numeric_skewness": {"amount": 0.5},
        "infinite_counts": {"amount": 0},
        "target_profile": {"positive_ratio": 0.25},
        "landmarks": {"dummy_score": 0.5},
        "duplicate_row_count": 1,
        "potential_cross_fold_duplicate_count": 1,
        "estimated_memory_mb": 0.125,
    }
    values.update(updates)
    return DatasetProfile.model_validate(values)


def make_validation_plan(**updates: Any) -> ValidationPlan:
    values: dict[str, Any] = {
        "splitter_name": "StratifiedKFold",
        "n_splits": 2,
        "shuffle": True,
        "random_seed": 42,
        "rationale": ["binary target", "fixed seed"],
        "dataset_hash": DATASET_HASH,
        "folds": [
            {
                "fold_index": 0,
                "train_positions": [2, 3, 4, 5],
                "validation_positions": [0, 1],
            },
            {
                "fold_index": 1,
                "train_positions": [0, 1, 4, 5],
                "validation_positions": [2, 3],
            },
        ],
    }
    values.update(updates)
    return ValidationPlan.model_validate(values)


def make_pipeline_spec(**updates: Any) -> PipelineSpec:
    values: dict[str, Any] = {
        "family": "linear",
        "task": "binary_classification",
        "numeric_imputer": "median_with_indicator",
        "numeric_scaler": "standard",
        "categorical_imputer": "constant",
        "categorical_encoder": "one_hot",
        "datetime_transformer": "calendar",
        "feature_selector": None,
        "model_name": "logistic_regression",
        "model_params": {"C": 1.0, "class_weight": "balanced"},
        "excluded_columns": ["leaked_target", "row_id"],
        "random_seed": 42,
    }
    values.update(updates)
    return PipelineSpec.model_validate(values)


def make_fidelity(**updates: Any) -> FidelitySpec:
    values: dict[str, Any] = {
        "level": 2,
        "sample_fraction": 1.0,
        "n_folds": 2,
        "max_iterations": 100,
        "seeds": [42],
    }
    values.update(updates)
    return FidelitySpec.model_validate(values)


def make_completed_trial(**updates: Any) -> TrialResult:
    values: dict[str, Any] = {
        "trial_id": "trial-0001",
        "family": "linear",
        "pipeline_spec": make_pipeline_spec(),
        "fidelity": make_fidelity(),
        "status": "completed",
        "primary_metric": "roc_auc",
        "mean_score": 0.8,
        "std_score": 0.02,
        "fold_scores": [0.78, 0.82],
        "fold_results": [
            {
                "fold_index": 0,
                "score": 0.78,
                "user_metric_value": 0.78,
                "fit_seconds": 1.0,
                "predict_seconds": 0.1,
                "n_train_rows": 8,
                "n_validation_rows": 4,
            },
            {
                "fold_index": 1,
                "score": 0.82,
                "user_metric_value": 0.82,
                "fit_seconds": 1.1,
                "predict_seconds": 0.1,
                "n_train_rows": 8,
                "n_validation_rows": 4,
            },
        ],
        "fit_seconds": 2.1,
        "predict_seconds": 0.2,
        "peak_memory_mb": 64.0,
        "started_at": STARTED_AT,
        "finished_at": FINISHED_AT,
    }
    values.update(updates)
    return TrialResult.model_validate(values)


def make_failed_trial(**updates: Any) -> TrialResult:
    values: dict[str, Any] = {
        "trial_id": "trial-0002",
        "family": "linear",
        "pipeline_spec": make_pipeline_spec(),
        "fidelity": make_fidelity(),
        "status": "failed",
        "primary_metric": "roc_auc",
        "fit_seconds": 0.3,
        "predict_seconds": 0.0,
        "failure_type": "ValueError",
        "failure_message": "incompatible synthetic pipeline",
        "started_at": STARTED_AT,
        "finished_at": FINISHED_AT,
    }
    values.update(updates)
    return TrialResult.model_validate(values)


def make_leakage_report(**updates: Any) -> LeakageReport:
    values: dict[str, Any] = {
        "findings": [
            {
                "finding_type": "target_copy",
                "severity": "critical",
                "confidence": 1.0,
                "column": "leaked_target",
                "evidence_summary": {"exact_match": True},
                "action": "exclude",
            }
        ],
        "excluded_columns": ["leaked_target"],
    }
    values.update(updates)
    return LeakageReport.model_validate(values)


def make_manifest(**updates: Any) -> RunManifest:
    dataset = make_dataset_bundle()
    values: dict[str, Any] = {
        "run_id": "run-20260102-0001",
        "package_version": "0.1.0",
        "status": "completed",
        "created_at": STARTED_AT,
        "updated_at": FINISHED_AT,
        "completed_at": FINISHED_AT,
        "configuration": make_config(),
        "source_hashes": dataset.source_hashes,
        "dataset": dataset,
        "dataset_profile": make_dataset_profile(),
        "leakage_report": make_leakage_report(),
        "validation_plan": make_validation_plan(),
        "dependency_versions": {"python": "3.12.0", "scikit-learn": "1.6.1"},
        "random_seed": 42,
        "best_pipeline": make_pipeline_spec(),
        "finalist_trial_ids": ["trial-0001"],
        "metrics": {"roc_auc": 0.8},
        "artifacts": {
            "best_pipeline": "best_pipeline.joblib",
            "leaderboard": "leaderboard.csv",
        },
        "scheduler_state": {"fidelity_level": 2, "family": "linear"},
        "notebook_validation": {
            "status": "passed",
            "executed_at": FINISHED_AT,
            "prediction_match": True,
        },
    }
    values.update(updates)
    return RunManifest.model_validate(values)


def make_run_result(**updates: Any) -> RunResult:
    values: dict[str, Any] = {
        "run_id": "run-20260102-0001",
        "status": "completed",
        "output_dir": "runs/unit",
        "manifest_path": "runs/unit/manifest.json",
        "best_pipeline_path": "runs/unit/best_pipeline.joblib",
        "best_pipeline_spec_path": "runs/unit/best_pipeline_spec.json",
        "leaderboard_path": "runs/unit/leaderboard.csv",
        "notebook_path": "runs/unit/solution.ipynb",
        "report_path": "runs/unit/report.html",
        "primary_metric": "roc_auc",
        "best_score": 0.8,
        "oof_predictions_path": "runs/unit/oof_predictions.csv",
        "predictions_path": "runs/unit/predictions.csv",
    }
    values.update(updates)
    return RunResult.model_validate(values)


@pytest.mark.parametrize(
    "contract",
    [
        make_config(),
        make_dataset_bundle(),
        make_dataset_profile(),
        make_validation_plan(),
        make_pipeline_spec(),
        make_fidelity(),
        make_completed_trial(),
        make_failed_trial(),
        make_leakage_report(),
        make_manifest(),
        make_run_result(),
    ],
    ids=lambda contract: type(contract).__name__,
)
def test_contract_json_round_trip_is_lossless(contract: ContractModel) -> None:
    payload = contract.to_json()

    assert type(contract).from_json(payload) == contract
    assert type(contract).model_validate_json(contract.model_dump_json()) == contract
    assert contract.to_json() == contract.to_json()


def test_contract_file_round_trip_is_atomic_and_typed(tmp_path: Path) -> None:
    manifest = make_manifest()
    destination = tmp_path / "nested" / "manifest.json"

    written_path = manifest.write_json(destination)
    restored = RunManifest.read_json(destination)

    assert written_path == destination
    assert restored == manifest
    assert isinstance(restored.configuration.output_dir, Path)
    assert isinstance(restored.created_at, datetime)


@pytest.mark.parametrize(
    ("contract", "version_field"),
    [
        (make_config(), "schema_version"),
        (make_dataset_bundle(), "schema_version"),
        (make_dataset_profile(), "schema_version"),
        (make_validation_plan(), "schema_version"),
        (make_pipeline_spec(), "spec_version"),
        (make_fidelity(), "spec_version"),
        (make_completed_trial(), "schema_version"),
        (make_leakage_report(), "schema_version"),
        (make_manifest(), "manifest_version"),
        (make_run_result(), "schema_version"),
    ],
    ids=lambda value: type(value).__name__ if isinstance(value, ContractModel) else value,
)
def test_persistent_contracts_default_to_v1_and_reject_unknown_versions(
    contract: ContractModel,
    version_field: str,
) -> None:
    serialized = contract.to_json_value()
    assert serialized[version_field] == 1

    serialized[version_field] = 2
    with pytest.raises(ValidationError):
        type(contract).model_validate(serialized)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target", " "),
        ("task", "clustering"),
        ("metric", "mape"),
        ("budget_seconds", 0),
        ("budget_seconds", -1),
        ("random_seed", -1),
        ("n_jobs", 0),
        ("memory_limit_mb", 127),
        ("trial_timeout_seconds", 0),
    ],
)
def test_automl_config_rejects_invalid_scalar_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        make_config(**{field: value})


@pytest.mark.parametrize(
    "updates",
    [
        {"group_column": "target"},
        {"time_column": "target"},
        {"id_column": "target"},
        {"predefined_fold_column": "target"},
        {"group_column": "shared", "id_column": "shared"},
        {"time_column": " "},
    ],
)
def test_automl_config_rejects_ambiguous_column_roles(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make_config(**updates)


@pytest.mark.parametrize(
    "updates",
    [
        {"task": "regression", "metric": "roc_auc"},
        {"task": "binary_classification", "metric": "rmse"},
        {"task": "multiclass_classification", "metric": "f1"},
        {"task": "multiclass_classification", "metric": "average_precision"},
    ],
)
def test_automl_config_rejects_incompatible_metrics(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make_config(**updates)


def test_automl_config_accepts_auto_and_valid_resource_limits() -> None:
    config = make_config(
        task="auto",
        metric="auto",
        n_jobs=1,
        memory_limit_mb=None,
        trial_timeout_seconds=None,
    )

    assert config.task is TaskType.AUTO
    assert config.metric is MetricName.AUTO
    assert config.n_jobs == 1


@pytest.mark.parametrize(
    "updates",
    [
        {"train_paths": ["train.csv", "train.csv"]},
        {"feature_columns": ["amount", "amount"]},
        {"feature_columns": ["amount", "target"]},
        {"feature_columns": ["amount", "row_id"], "excluded_columns": ["row_id"]},
        {"test_path": None, "n_test_rows": 2},
        {"test_path": "test.csv", "n_test_rows": None},
        {"source_hashes": {"train.csv": "not-a-sha256"}},
    ],
)
def test_dataset_bundle_rejects_non_reconstructible_descriptions(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        make_dataset_bundle(**updates)


@pytest.mark.parametrize(
    "updates",
    [
        {"dataset_hash": "invalid"},
        {"missing_ratios": {"amount": 1.01}},
        {"unique_ratios": {"amount": -0.01}},
        {"cardinalities": {"amount": -1}},
        {"infinite_counts": {"amount": -1}},
        {"numeric_columns": ["amount", "city"]},
        {"n_features": 4},
        {"inferred_types": {"amount": "numeric", "city": "categorical"}},
    ],
)
def test_dataset_profile_rejects_invalid_statistics_or_column_roles(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        make_dataset_profile(**updates)


def test_fold_assignment_requires_unique_disjoint_non_negative_positions() -> None:
    for values in (
        {"fold_index": 0, "train_positions": [0, 0], "validation_positions": [1]},
        {"fold_index": 0, "train_positions": [0], "validation_positions": [0]},
        {"fold_index": 0, "train_positions": [-1], "validation_positions": [0]},
    ):
        with pytest.raises(ValidationError):
            FoldAssignment.model_validate(values)


@pytest.mark.parametrize(
    "updates",
    [
        {"splitter_name": "GroupKFold", "group_column": None},
        {
            "splitter_name": "TimeSeriesSplit",
            "time_column": "event_time",
            "shuffle": True,
            "random_seed": 42,
        },
        {"splitter_name": "PredefinedSplit", "predefined_fold_column": None},
        {"shuffle": True, "random_seed": None},
        {"shuffle": False, "random_seed": 42},
        {"n_splits": 3},
        {
            "folds": [
                {"fold_index": 1, "train_positions": [2], "validation_positions": [0]},
                {"fold_index": 0, "train_positions": [0], "validation_positions": [2]},
            ]
        },
    ],
)
def test_validation_plan_rejects_incoherent_splitter_state(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make_validation_plan(**updates)


@pytest.mark.parametrize(
    "updates",
    [
        {"task": "auto"},
        {"family": " "},
        {"model_name": ""},
        {"random_seed": -1},
        {"excluded_columns": ["row_id", "row_id"]},
        {"model_params": {"unsupported": object()}},
    ],
)
def test_pipeline_spec_rejects_non_reconstructible_values(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make_pipeline_spec(**updates)


@pytest.mark.parametrize(
    "updates",
    [
        {"level": -1},
        {"level": 4},
        {"sample_fraction": 0.0},
        {"sample_fraction": 1.01},
        {"n_folds": 1},
        {"max_iterations": 0},
        {"seeds": []},
        {"seeds": [42, 42]},
        {"seeds": [-1]},
        {"level": 2, "sample_fraction": 0.5},
        {"level": 3, "seeds": [42]},
    ],
)
def test_fidelity_spec_rejects_invalid_or_ambiguous_levels(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make_fidelity(**updates)


def test_confirmation_fidelity_requires_full_data_and_multiple_seeds() -> None:
    fidelity = make_fidelity(level=3, sample_fraction=1.0, seeds=[42, 43])

    assert fidelity.level == 3
    assert fidelity.seeds == [42, 43]


def test_completed_and_failed_trials_preserve_their_outcomes() -> None:
    completed = make_completed_trial()
    failed = make_failed_trial()

    assert completed.status is TrialStatus.COMPLETED
    assert completed.mean_score == pytest.approx(0.8)
    assert len(completed.fold_results) == completed.fidelity.n_folds
    assert failed.status is TrialStatus.FAILED
    assert failed.mean_score is None
    assert failed.failure_type == "ValueError"


@pytest.mark.parametrize(
    "updates",
    [
        {"mean_score": None},
        {"std_score": None},
        {"fold_scores": []},
        {"failure_type": "ValueError", "failure_message": "unexpected"},
        {"family": "tree"},
        {"finished_at": None},
        {"finished_at": STARTED_AT - timedelta(seconds=1)},
    ],
)
def test_completed_trial_rejects_inconsistent_state(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make_completed_trial(**updates)


@pytest.mark.parametrize(
    "updates",
    [
        {"failure_type": None},
        {"failure_message": None},
        {"mean_score": 0.1},
        {"finished_at": None},
    ],
)
def test_failed_trial_rejects_inconsistent_state(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make_failed_trial(**updates)


def test_non_terminal_trial_cannot_have_a_finish_timestamp() -> None:
    with pytest.raises(ValidationError):
        make_completed_trial(
            status="running",
            mean_score=None,
            std_score=None,
            fold_scores=[],
            fold_results=[],
            finished_at=FINISHED_AT,
        )


def test_fold_result_rejects_invalid_sizes_and_non_finite_score() -> None:
    valid = make_completed_trial().fold_results[0]
    assert isinstance(valid, FoldResult)

    for updates in ({"n_train_rows": 0}, {"score": float("nan")}, {"fit_seconds": -1}):
        values = valid.model_dump()
        values.update(updates)
        with pytest.raises(ValidationError):
            FoldResult.model_validate(values)


def test_leakage_contract_records_every_exclusion_without_raw_values() -> None:
    report = make_leakage_report()
    finding = report.findings[0]

    assert isinstance(finding, LeakageFinding)
    assert finding.finding_type is LeakageFindingType.TARGET_COPY
    assert finding.severity is FindingSeverity.CRITICAL
    assert report.excluded_columns == [finding.column]


@pytest.mark.parametrize(
    "updates",
    [
        {"excluded_columns": ["unrecorded_column"]},
        {"excluded_columns": ["leaked_target", "leaked_target"]},
        {
            "findings": [
                {
                    "finding_type": "target_copy",
                    "severity": "critical",
                    "confidence": 1.1,
                    "column": "leaked_target",
                    "action": "exclude",
                }
            ]
        },
    ],
)
def test_leakage_report_rejects_unexplained_or_invalid_findings(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        make_leakage_report(**updates)


def test_notebook_validation_enforces_status_evidence() -> None:
    pending = NotebookValidation()
    passed = make_manifest().notebook_validation

    assert pending.status is NotebookValidationStatus.PENDING
    assert passed.status is NotebookValidationStatus.PASSED
    assert passed.prediction_match is True

    invalid_values = (
        {"status": "pending", "prediction_match": True},
        {"status": "passed", "executed_at": FINISHED_AT, "prediction_match": False},
        {"status": "failed", "executed_at": FINISHED_AT},
    )
    for values in invalid_values:
        with pytest.raises(ValidationError):
            NotebookValidation.model_validate(values)


def test_completed_manifest_contains_reconstruction_and_environment_evidence() -> None:
    manifest = make_manifest()

    assert manifest.status is RunStatus.COMPLETED
    assert manifest.best_pipeline == make_pipeline_spec()
    assert manifest.source_hashes == manifest.dataset.source_hashes
    assert manifest.random_seed == manifest.configuration.random_seed
    assert manifest.dependency_versions["python"].startswith("3.12")
    assert manifest.notebook_validation.prediction_match is True


@pytest.mark.parametrize(
    "updates",
    [
        {"source_hashes": {"different.csv": "d" * 64}},
        {"random_seed": 43},
        {"completed_at": None},
        {"best_pipeline": None},
        {"updated_at": STARTED_AT - timedelta(seconds=1)},
        {"completed_at": STARTED_AT - timedelta(seconds=1)},
        {"status": "running"},
    ],
)
def test_run_manifest_rejects_inconsistent_authoritative_state(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        make_manifest(**updates)


def test_running_manifest_omits_completion_only_fields() -> None:
    manifest = make_manifest(status="running", completed_at=None, best_pipeline=None)

    assert manifest.status is RunStatus.RUNNING
    assert manifest.completed_at is None
    assert manifest.best_pipeline is None


def test_run_result_exposes_typed_artifact_paths() -> None:
    result = make_run_result()

    assert result.status is RunStatus.COMPLETED
    assert result.primary_metric is MetricName.ROC_AUC
    assert result.best_score == pytest.approx(0.8)
    assert isinstance(result.manifest_path, Path)
    assert result.best_pipeline_path.name == "best_pipeline.joblib"


def test_contracts_reject_unknown_fields_and_non_finite_numbers() -> None:
    with pytest.raises(ValidationError):
        make_config(unknown_option=True)
    with pytest.raises(ValidationError):
        make_run_result(best_score=float("inf"))

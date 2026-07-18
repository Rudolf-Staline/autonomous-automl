"""Persistent conditional Optuna optimizer tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import optuna
import pytest
from optuna.trial import TrialState

from autonomous_automl.components import ModelRegistry
from autonomous_automl.components.models import RidgeAdapter
from autonomous_automl.contracts import (
    DatasetProfile,
    FidelitySpec,
    PipelineSpec,
    TrialResult,
)
from autonomous_automl.search import OptunaFamilyOptimizer, SearchCandidate
from autonomous_automl.utils.errors import ConfigurationError, ResumeError


def _profile(*, dataset_hash: str = "a" * 64) -> DatasetProfile:
    return DatasetProfile(
        dataset_hash=dataset_hash,
        n_rows=30,
        n_features=1,
        inferred_task="regression",
        inferred_types={"feature": "numeric"},
        numeric_columns=["feature"],
        missing_ratios={"feature": 0.0},
        cardinalities={"feature": 30},
        unique_ratios={"feature": 1.0},
        numeric_skewness={"feature": 0.0},
        infinite_counts={"feature": 0},
        estimated_memory_mb=0.01,
    )


def _template(model_name: str) -> PipelineSpec:
    defaults: dict[str, object]
    if model_name == "ridge":
        defaults = {"alpha": 1.0}
    elif model_name == "elastic_net":
        defaults = {"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 1_000}
    else:
        defaults = {}
    return PipelineSpec.model_validate(
        {
            "family": "linear",
            "task": "regression",
            "numeric_imputer": "mean",
            "numeric_scaler": "standard",
            "categorical_imputer": "constant",
            "categorical_encoder": "ordinal",
            "datetime_transformer": "calendar",
            "model_name": model_name,
            "model_params": defaults,
            "random_seed": 42,
        }
    )


def _fidelity() -> FidelitySpec:
    return FidelitySpec(
        level=0,
        sample_fraction=0.5,
        n_folds=2,
        max_iterations=50,
        seeds=[42],
    )


def _result(candidate: SearchCandidate, status: str) -> TrialResult:
    now = datetime.now(UTC)
    common = {
        "trial_id": candidate.candidate_id,
        "family": candidate.family,
        "pipeline_spec": candidate.pipeline_spec,
        "fidelity": candidate.fidelity,
        "status": status,
        "primary_metric": "rmse",
        "started_at": now,
        "finished_at": now,
    }
    if status == "completed":
        common.update(
            {
                "mean_score": -1.25,
                "std_score": 0.05,
                "fold_scores": [-1.2, -1.3],
            }
        )
    elif status == "failed":
        common.update(
            {
                "failure_type": "ValueError",
                "failure_message": "synthetic failure without user data",
            }
        )
    return TrialResult.model_validate(common)


def test_optimizer_creates_one_persistent_maximize_study_per_model(tmp_path: Path) -> None:
    optimizer = OptunaFamilyOptimizer(
        "linear",
        [_template("ridge"), _template("elastic_net")],
        _profile(),
        tmp_path / "optuna.sqlite3",
        random_seed=7,
        registry=ModelRegistry.default(include_optional=False),
        study_prefix="unit",
    )

    first = optimizer.ask(_fidelity())
    second = optimizer.ask(_fidelity())

    assert optimizer.model_names == ("elastic_net", "ridge")
    assert {first.model_name, second.model_name} == {"elastic_net", "ridge"}
    assert first.pipeline_spec.model_params
    assert first.pipeline_spec.to_json()
    for model_name in optimizer.model_names:
        study = optimizer.study_for_model(model_name)
        assert study.direction is optuna.study.StudyDirection.MAXIMIZE
        assert study.study_name == f"unit:linear:{model_name}"


def test_completed_tell_and_reconcile_are_idempotent_across_restart(tmp_path: Path) -> None:
    database = tmp_path / "optuna.sqlite3"
    arguments = {
        "family": "linear",
        "templates": [_template("ridge")],
        "profile": _profile(),
        "storage_path": database,
        "random_seed": 11,
        "registry": ModelRegistry.default(include_optional=False),
        "study_prefix": "resume",
    }
    optimizer = OptunaFamilyOptimizer(**arguments)
    candidate = optimizer.ask(_fidelity())
    result = _result(candidate, "completed")

    optimizer.tell(candidate, result)
    optimizer.tell(candidate, result)
    restored = OptunaFamilyOptimizer(**arguments)
    restored.reconcile([(candidate, result)])
    following = restored.ask(_fidelity())

    trials = restored.study_for_model("ridge").trials
    assert trials[0].state is TrialState.COMPLETE
    assert trials[0].value == result.mean_score
    assert following.trial_number == 1


def test_business_result_reconciles_running_optuna_trial_after_crash(tmp_path: Path) -> None:
    database = tmp_path / "crash.sqlite3"
    arguments = {
        "family": "linear",
        "templates": [_template("ridge")],
        "profile": _profile(),
        "storage_path": database,
        "random_seed": 13,
        "registry": ModelRegistry.default(include_optional=False),
        "study_prefix": "crash",
    }
    before_crash = OptunaFamilyOptimizer(**arguments)
    candidate = before_crash.ask(_fidelity())
    business_result = _result(candidate, "completed")
    assert before_crash.study_for_model("ridge").trials[0].state is TrialState.RUNNING

    resumed = OptunaFamilyOptimizer(**arguments)
    resumed.reconcile_results([business_result])

    frozen = resumed.study_for_model("ridge").trials[0]
    assert frozen.state is TrialState.COMPLETE
    assert frozen.value == business_result.mean_score


def test_in_flight_candidate_is_recovered_exactly_after_restart(tmp_path: Path) -> None:
    database = tmp_path / "in-flight.sqlite3"
    arguments = {
        "family": "linear",
        "templates": [_template("ridge")],
        "profile": _profile(),
        "storage_path": database,
        "random_seed": 17,
        "registry": ModelRegistry.default(include_optional=False),
        "study_prefix": "in-flight",
    }
    original = OptunaFamilyOptimizer(**arguments)
    candidate = original.ask(_fidelity())

    restored = OptunaFamilyOptimizer(**arguments)
    recovered = restored.candidate_by_id(
        candidate.candidate_id,
        candidate.pipeline_spec,
        candidate.fidelity,
    )

    assert recovered == candidate
    assert (
        restored.candidate_by_id(
            "missing-candidate",
            candidate.pipeline_spec,
            candidate.fidelity,
        )
        is None
    )


def test_in_flight_candidate_recovery_rejects_changed_spec_or_fidelity(tmp_path: Path) -> None:
    optimizer = OptunaFamilyOptimizer(
        "linear",
        [_template("ridge")],
        _profile(),
        tmp_path / "changed-in-flight.sqlite3",
        random_seed=23,
        registry=ModelRegistry.default(include_optional=False),
        study_prefix="changed-in-flight",
    )
    candidate = optimizer.ask(_fidelity())
    changed_spec = candidate.pipeline_spec.model_copy(
        update={"model_params": {"alpha": 999.0}},
        deep=True,
    )
    changed_fidelity = candidate.fidelity.model_copy(update={"sample_fraction": 0.75})

    with pytest.raises(ResumeError, match="identity"):
        optimizer.candidate_by_id(candidate.candidate_id, changed_spec, candidate.fidelity)
    with pytest.raises(ResumeError, match="identity"):
        optimizer.candidate_by_id(candidate.candidate_id, candidate.pipeline_spec, changed_fidelity)


@pytest.mark.parametrize(
    ("status", "expected_state"),
    [("failed", TrialState.FAIL), ("pruned", TrialState.PRUNED)],
)
def test_tell_maps_non_success_terminal_states(
    tmp_path: Path,
    status: str,
    expected_state: TrialState,
) -> None:
    optimizer = OptunaFamilyOptimizer(
        "linear",
        [_template("ridge")],
        _profile(),
        tmp_path / f"{status}.sqlite3",
        random_seed=19,
        registry=ModelRegistry.default(include_optional=False),
        study_prefix=status,
    )
    candidate = optimizer.ask(_fidelity())

    optimizer.tell(candidate, _result(candidate, status))

    assert optimizer.study_for_model("ridge").trials[0].state is expected_state


def test_resume_rejects_changed_dataset_identity(tmp_path: Path) -> None:
    database = tmp_path / "identity.sqlite3"
    template = _template("ridge")
    registry = ModelRegistry.default(include_optional=False)
    OptunaFamilyOptimizer(
        "linear",
        [template],
        _profile(dataset_hash="a" * 64),
        database,
        random_seed=3,
        registry=registry,
        study_prefix="identity",
    )

    with pytest.raises(ResumeError, match="identity"):
        OptunaFamilyOptimizer(
            "linear",
            [template],
            _profile(dataset_hash="b" * 64),
            database,
            random_seed=3,
            registry=registry,
            study_prefix="identity",
        )


def test_adapter_parameters_must_be_json_serializable(tmp_path: Path) -> None:
    class NonJsonRidgeAdapter(RidgeAdapter):
        name = "non_json_ridge"

        def suggest_params(
            self,
            trial: optuna.Trial,
            profile: DatasetProfile,
            fidelity: FidelitySpec,
        ) -> dict[str, object]:
            return {"alpha": {1.0, 2.0}}

    registry = ModelRegistry([NonJsonRidgeAdapter()])
    optimizer = OptunaFamilyOptimizer(
        "linear",
        [_template("non_json_ridge")],
        _profile(),
        tmp_path / "invalid.sqlite3",
        random_seed=5,
        registry=registry,
        study_prefix="invalid",
    )

    with pytest.raises(ConfigurationError, match="JSON"):
        optimizer.ask(_fidelity())

    assert optimizer.study_for_model("non_json_ridge").trials[0].state is TrialState.FAIL

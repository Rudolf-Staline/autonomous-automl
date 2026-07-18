"""M7 acceptance tests for deterministic persistent Optuna ask/tell search."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path

import optuna
import pytest

from autonomous_automl.components import ModelRegistry
from autonomous_automl.contracts import (
    DatasetProfile,
    FidelitySpec,
    MetricName,
    PipelineSpec,
    TaskType,
    TrialResult,
    TrialStatus,
)
from autonomous_automl.pipelines import (
    PipelineGrammar,
    check_pipeline_compatibility,
    pipeline_fingerprint,
)
from autonomous_automl.search import OptunaFamilyOptimizer, SearchCandidate
from autonomous_automl.utils.errors import ResumeError

STARTED_AT = datetime(2026, 7, 17, 9, 0, tzinfo=UTC)
FINISHED_AT = STARTED_AT + timedelta(seconds=2)


def make_profile() -> DatasetProfile:
    return DatasetProfile(
        dataset_hash="a" * 64,
        n_rows=200,
        n_features=2,
        inferred_task=TaskType.BINARY_CLASSIFICATION,
        inferred_types={"signal": "numeric", "category": "categorical"},
        numeric_columns=["signal"],
        categorical_columns=["category"],
        missing_ratios={"signal": 0.1, "category": 0.0},
        cardinalities={"signal": 180, "category": 8},
        unique_ratios={"signal": 0.9, "category": 0.04},
        numeric_skewness={"signal": 0.2},
        infinite_counts={"signal": 0},
        target_profile={"imbalance_ratio": 1.5},
        landmarks={"numeric_fraction": 0.5, "categorical_fraction": 0.5},
        estimated_memory_mb=0.1,
    )


def make_fidelity() -> FidelitySpec:
    return FidelitySpec(
        level=1,
        sample_fraction=0.75,
        n_folds=3,
        max_iterations=50,
        seeds=[42],
    )


def core_candidates() -> tuple[ModelRegistry, list[PipelineSpec]]:
    registry = ModelRegistry.default(include_optional=False)
    candidates = PipelineGrammar(registry).generate(make_profile(), random_seed=42)
    return registry, candidates


def linear_templates() -> tuple[ModelRegistry, list[PipelineSpec]]:
    registry, candidates = core_candidates()
    templates = [candidate for candidate in candidates if candidate.family == "linear"]
    assert templates
    return registry, templates


def make_result(
    candidate: SearchCandidate,
    status: TrialStatus,
    *,
    score: float = 0.7,
) -> TrialResult:
    common: dict[str, object] = {
        "trial_id": candidate.candidate_id,
        "family": candidate.family,
        "pipeline_spec": candidate.pipeline_spec,
        "fidelity": candidate.fidelity,
        "status": status,
        "primary_metric": MetricName.ROC_AUC,
        "started_at": STARTED_AT,
        "finished_at": FINISHED_AT,
    }
    if status is TrialStatus.COMPLETED:
        common.update(
            {
                "mean_score": score,
                "std_score": 0.0,
                "fold_scores": [score, score, score],
                "fit_seconds": 1.0,
                "predict_seconds": 0.1,
            }
        )
    elif status is TrialStatus.FAILED:
        common.update(
            {
                "failure_type": "ValueError",
                "failure_message": "synthetic failed candidate",
            }
        )
    return TrialResult.model_validate(common)


def make_linear_optimizer(
    storage_path: Path,
    *,
    study_prefix: str,
    random_seed: int = 42,
) -> OptunaFamilyOptimizer:
    registry, templates = linear_templates()
    return OptunaFamilyOptimizer(
        "linear",
        templates,
        make_profile(),
        storage_path,
        random_seed=random_seed,
        registry=registry,
        study_prefix=study_prefix,
    )


def test_same_seed_and_identical_history_produce_identical_suggestions(
    tmp_path: Path,
) -> None:
    first = make_linear_optimizer(tmp_path / "first.sqlite3", study_prefix="deterministic")
    second = make_linear_optimizer(tmp_path / "second.sqlite3", study_prefix="deterministic")
    fidelity = make_fidelity()

    for index in range(6):
        first_candidate = first.ask(fidelity)
        second_candidate = second.ask(fidelity)

        assert first_candidate.pipeline_spec == second_candidate.pipeline_spec
        assert first_candidate.template_fingerprint == second_candidate.template_fingerprint
        assert first_candidate.candidate_id == second_candidate.candidate_id
        score = 0.55 + index / 100
        first.tell(
            first_candidate, make_result(first_candidate, TrialStatus.COMPLETED, score=score)
        )
        second.tell(
            second_candidate,
            make_result(second_candidate, TrialStatus.COMPLETED, score=score),
        )


def test_all_core_family_spaces_emit_compatible_json_pipeline_specs(tmp_path: Path) -> None:
    profile = make_profile()
    fidelity = make_fidelity()
    registry, candidates = core_candidates()
    optimized = [candidate for candidate in candidates if candidate.family != "baseline"]
    families = sorted({candidate.family for candidate in optimized})

    assert {"linear", "random_forest", "extra_trees", "hist_gradient_boosting"} <= set(families)
    for family in families:
        templates = [candidate for candidate in optimized if candidate.family == family]
        optimizer = OptunaFamilyOptimizer(
            family,
            templates,
            profile,
            tmp_path / "core.sqlite3",
            random_seed=42,
            registry=registry,
            study_prefix="core-spaces",
        )

        candidate = optimizer.ask(fidelity)
        restored = PipelineSpec.from_json(candidate.pipeline_spec.to_json())

        assert restored == candidate.pipeline_spec
        assert check_pipeline_compatibility(restored, profile, registry).compatible
        assert len(pipeline_fingerprint(restored)) == 64


def test_ask_tell_persists_completed_failed_and_pruned_states(tmp_path: Path) -> None:
    optimizer = make_linear_optimizer(tmp_path / "states.sqlite3", study_prefix="states")
    fidelity = make_fidelity()
    expected_states = (
        (TrialStatus.COMPLETED, optuna.trial.TrialState.COMPLETE),
        (TrialStatus.FAILED, optuna.trial.TrialState.FAIL),
        (TrialStatus.PRUNED, optuna.trial.TrialState.PRUNED),
    )

    for status, _ in expected_states:
        candidate = optimizer.ask(fidelity)
        optimizer.tell(candidate, make_result(candidate, status))

    study = optimizer.study_for_model("logistic_regression")
    assert [trial.state for trial in study.trials] == [state for _, state in expected_states]
    assert study.trials[0].value == pytest.approx(0.7)
    assert study.trials[1].value is None
    assert study.trials[2].value is None


def test_identical_tell_replay_is_idempotent_and_conflicts_are_rejected(
    tmp_path: Path,
) -> None:
    optimizer = make_linear_optimizer(tmp_path / "tell.sqlite3", study_prefix="tell-once")
    candidate = optimizer.ask(make_fidelity())
    result = make_result(candidate, TrialStatus.COMPLETED, score=0.73)

    optimizer.tell(candidate, result)
    optimizer.tell(candidate, result)
    optimizer.reconcile([(candidate, result), (candidate, result)])

    study = optimizer.study_for_model(candidate.model_name)
    assert len(study.trials) == 1
    assert study.trials[0].state is optuna.trial.TrialState.COMPLETE
    assert study.trials[0].value == pytest.approx(0.73)
    conflicting = make_result(candidate, TrialStatus.COMPLETED, score=0.12)
    with pytest.raises(ResumeError, match="value conflicts"):
        optimizer.tell(candidate, conflicting)


def test_reopened_sqlite_study_keeps_history_and_next_trial_number(tmp_path: Path) -> None:
    fidelity = make_fidelity()
    initial = make_linear_optimizer(
        tmp_path / "reopened.sqlite3",
        study_prefix="resume-determinism",
    )
    completed_candidate = initial.ask(fidelity)
    initial.tell(
        completed_candidate,
        make_result(completed_candidate, TrialStatus.COMPLETED, score=0.61),
    )

    reopened = make_linear_optimizer(
        tmp_path / "reopened.sqlite3",
        study_prefix="resume-determinism",
    )
    persisted = reopened.study_for_model("logistic_regression").trials
    assert len(persisted) == 1
    assert persisted[0].state is optuna.trial.TrialState.COMPLETE
    assert persisted[0].value == pytest.approx(0.61)
    reopened_next = reopened.ask(fidelity)

    assert len(reopened.study_for_model("logistic_regression").trials) == 2
    assert reopened_next.trial_number == 1
    assert PipelineSpec.from_json(reopened_next.pipeline_spec.to_json()) == (
        reopened_next.pipeline_spec
    )
    assert check_pipeline_compatibility(reopened_next.pipeline_spec, make_profile()).compatible


def test_reopened_optimizer_matches_uninterrupted_next_suggestion(tmp_path: Path) -> None:
    fidelity = make_fidelity()
    continuous = make_linear_optimizer(
        tmp_path / "continuous.sqlite3",
        study_prefix="restart-equivalence",
    )
    restart_source = make_linear_optimizer(
        tmp_path / "restarted.sqlite3",
        study_prefix="restart-equivalence",
    )
    for optimizer in (continuous, restart_source):
        candidate = optimizer.ask(fidelity)
        optimizer.tell(
            candidate,
            make_result(candidate, TrialStatus.COMPLETED, score=0.64),
        )

    uninterrupted_next = continuous.ask(fidelity)
    reopened = make_linear_optimizer(
        tmp_path / "restarted.sqlite3",
        study_prefix="restart-equivalence",
    )
    reopened_next = reopened.ask(fidelity)

    assert reopened_next.pipeline_spec == uninterrupted_next.pipeline_spec
    assert reopened_next.template_fingerprint == uninterrupted_next.template_fingerprint
    assert reopened_next.candidate_id == uninterrupted_next.candidate_id


def test_absent_optional_models_do_not_remove_baseline_or_break_optimizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optional_modules = {"xgboost", "lightgbm", "catboost"}
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, package: str | None = None) -> object | None:
        if name in optional_modules:
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    registry = ModelRegistry.default(include_optional=True)
    candidates = PipelineGrammar(registry).generate(make_profile(), random_seed=42)
    baselines = [candidate for candidate in candidates if candidate.family == "baseline"]
    optimization_templates = [candidate for candidate in candidates if candidate.family == "linear"]

    assert len(baselines) == 1
    assert baselines[0].model_name == "dummy"
    assert not {candidate.model_name for candidate in candidates}.intersection(optional_modules)
    optimizer = OptunaFamilyOptimizer(
        "linear",
        optimization_templates,
        make_profile(),
        tmp_path / "without-optionals.sqlite3",
        random_seed=42,
        registry=registry,
        study_prefix="without-optionals",
    )
    candidate = optimizer.ask(make_fidelity())
    assert candidate.model_name == "logistic_regression"
    assert "dummy" not in optimizer.model_names

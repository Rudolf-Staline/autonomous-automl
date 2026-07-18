"""Acceptance tests for persisted finalist full-fidelity confirmation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from autonomous_automl.contracts import (
    AutoMLConfig,
    FidelitySpec,
    MetricName,
    PipelineSpec,
    RunManifest,
    RunStatus,
    TaskType,
    TrialResult,
    TrialStatus,
    ValidationPlan,
)
from autonomous_automl.data import LoadedDataset, load_dataset
from autonomous_automl.evaluation.confirmation import FinalistConfirmer
from autonomous_automl.profiling import detect_leakage, profile_dataset
from autonomous_automl.search.budget import BudgetManager, BudgetPhase
from autonomous_automl.search.fidelity import FidelityPolicy
from autonomous_automl.tracking.store import ExperimentStore
from autonomous_automl.validation import ValidationPlanner, audit_validation


class ManualClock:
    """Monotonic clock advanced only by the fake evaluator."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


@dataclass(frozen=True, slots=True)
class FakeOutcome:
    trial_result: TrialResult


@dataclass(frozen=True, slots=True)
class EvaluationCall:
    trial_id: str
    model_name: str
    fidelity: FidelitySpec
    timeout_seconds: float | None


class FakeEvaluator:
    """Fast evaluator double returning real terminal result contracts."""

    def __init__(
        self,
        *,
        clock: ManualClock,
        elapsed_seconds: float = 1.0,
        failing_models: set[str] | None = None,
    ) -> None:
        self.clock = clock
        self.elapsed_seconds = elapsed_seconds
        self.failing_models = set(failing_models or set())
        self.calls: list[EvaluationCall] = []

    def evaluate(
        self,
        trial_id: str,
        pipeline_spec: PipelineSpec,
        fidelity: FidelitySpec,
        dataset: LoadedDataset,
        profile: object,
        validation_plan: ValidationPlan,
        metric: MetricName | str = MetricName.AUTO,
        *,
        timeout_seconds: float | None = None,
    ) -> FakeOutcome:
        del profile, validation_plan
        assert dataset.test_X is not None
        self.calls.append(
            EvaluationCall(
                trial_id,
                pipeline_spec.model_name,
                fidelity,
                timeout_seconds,
            )
        )
        self.clock.advance(self.elapsed_seconds)
        if pipeline_spec.model_name in self.failing_models:
            raise RuntimeError("synthetic finalist failure")
        score = {
            "logistic_regression": 0.91,
            "random_forest": 0.88,
            "extra_trees": 0.86,
            "hist_gradient_boosting": 0.84,
            "dummy": 0.50,
        }.get(pipeline_spec.model_name, 0.75)
        now = datetime.now(UTC)
        return FakeOutcome(
            TrialResult(
                trial_id=trial_id,
                family=pipeline_spec.family,
                pipeline_spec=pipeline_spec,
                fidelity=fidelity,
                status=TrialStatus.COMPLETED,
                primary_metric=MetricName(metric),
                mean_score=score,
                std_score=0.01,
                fold_scores=[score] * fidelity.n_folds,
                fit_seconds=self.elapsed_seconds,
                predict_seconds=0.0,
                started_at=now,
                finished_at=now,
            )
        )


@dataclass(frozen=True, slots=True)
class ConfirmationProblem:
    run_id: str
    dataset: LoadedDataset
    profile: object
    plan: ValidationPlan
    store: ExperimentStore
    policy: FidelityPolicy


def make_problem(
    root: Path,
    *,
    run_id: str = "confirmation-run",
    test_signal: float = 10.0,
    budget_seconds: int = 500,
) -> ConfirmationProblem:
    root.mkdir(parents=True, exist_ok=True)
    train_path = root / "train.csv"
    test_path = root / "test.csv"
    pd.DataFrame(
        {
            "signal": [float(index % 9) for index in range(36)],
            "category": [f"category-{index % 3}" for index in range(36)],
            "target": [index % 2 for index in range(36)],
        }
    ).to_csv(train_path, index=False)
    pd.DataFrame(
        {
            "signal": [test_signal, test_signal + 1.0],
            "category": ["unseen", "category-1"],
        }
    ).to_csv(test_path, index=False)
    config = AutoMLConfig(
        target="target",
        task=TaskType.BINARY_CLASSIFICATION,
        metric=MetricName.ACCURACY,
        budget_seconds=budget_seconds,
        random_seed=42,
        test_path=test_path,
        output_dir=root / "run",
    )
    dataset = load_dataset(train_path, config)
    profile = profile_dataset(dataset, TaskType.BINARY_CLASSIFICATION)
    plan = ValidationPlanner(max_splits=3).plan(dataset, profile, config)
    now = datetime.now(UTC)
    manifest = RunManifest(
        run_id=run_id,
        package_version="0.1.0",
        status=RunStatus.RUNNING,
        created_at=now,
        updated_at=now,
        configuration=config,
        source_hashes=dataset.bundle.source_hashes,
        dataset=dataset.bundle,
        dataset_profile=profile,
        leakage_report=detect_leakage(dataset, profile),
        validation_plan=plan,
        validation_audit=audit_validation(dataset, plan),
        dependency_versions={"python": "3.12"},
        random_seed=42,
    )
    store = ExperimentStore.open(root / "registry.sqlite3")
    store.create_run(manifest)
    policy = FidelityPolicy(
        available_folds=plan.n_splits,
        random_seed=42,
        confirmation_seed_count=3,
    )
    return ConfirmationProblem(run_id, dataset, profile, plan, store, policy)


def make_spec(model_name: str, *, random_seed: int) -> PipelineSpec:
    family = {
        "dummy": "baseline",
        "logistic_regression": "linear",
        "random_forest": "random_forest",
        "extra_trees": "extra_trees",
        "hist_gradient_boosting": "hist_gradient_boosting",
    }[model_name]
    return PipelineSpec(
        family=family,
        task=TaskType.BINARY_CLASSIFICATION,
        numeric_imputer="median",
        numeric_scaler="standard",
        categorical_imputer="constant",
        categorical_encoder="one_hot",
        datetime_transformer="drop",
        model_name=model_name,
        model_params={},
        random_seed=random_seed,
    )


def search_result(
    specification: PipelineSpec,
    fidelity: FidelitySpec,
    *,
    score: float,
) -> TrialResult:
    now = datetime.now(UTC)
    return TrialResult(
        trial_id=f"search-{specification.model_name}-{fidelity.level}",
        family=specification.family,
        pipeline_spec=specification,
        fidelity=fidelity,
        status=TrialStatus.COMPLETED,
        primary_metric=MetricName.ACCURACY,
        mean_score=score,
        std_score=0.01,
        fold_scores=[score] * fidelity.n_folds,
        started_at=now,
        finished_at=now,
    )


def candidates(problem: ConfirmationProblem) -> list[TrialResult]:
    low = problem.policy.level(0)
    return [
        search_result(make_spec("logistic_regression", random_seed=1), low, score=0.95),
        search_result(make_spec("random_forest", random_seed=2), low, score=0.90),
        search_result(make_spec("extra_trees", random_seed=3), low, score=0.85),
        search_result(
            make_spec("hist_gradient_boosting", random_seed=4),
            low,
            score=0.80,
        ),
        search_result(make_spec("dummy", random_seed=5), low, score=0.40),
    ]


def confirmer(
    problem: ConfirmationProblem,
    evaluator: FakeEvaluator,
    budget: BudgetManager,
    *,
    top_n: int,
    estimated_trial_seconds: float = 0.0,
) -> FinalistConfirmer:
    return FinalistConfirmer(
        run_id=problem.run_id,
        evaluator=evaluator,
        dataset=problem.dataset,
        profile=problem.profile,  # type: ignore[arg-type]
        validation_plan=problem.plan,
        store=problem.store,
        budget=budget,
        metric=MetricName.ACCURACY,
        lease_epoch=problem.store.get_run_progress(problem.run_id).lease_epoch,
        fidelity_policy=problem.policy,
        top_n=top_n,
        estimated_trial_seconds=estimated_trial_seconds,
    )


def persist_result(problem: ConfirmationProblem, result: TrialResult) -> None:
    reservation = problem.store.reserve_trial(
        problem.run_id,
        result.trial_id,
        result.pipeline_spec,
        result.fidelity,
        result.primary_metric,
    )
    assert reservation.reserved
    problem.store.record_trial_result(
        result,
        attempt_token=reservation.attempt_token,
        lease_epoch=reservation.lease_epoch,
    )


def test_shortlist_keeps_top_n_plus_baseline_and_runs_full_then_confirmation(
    tmp_path: Path,
) -> None:
    problem = make_problem(tmp_path)
    clock = ManualClock()
    evaluator = FakeEvaluator(clock=clock)
    service = confirmer(
        problem,
        evaluator,
        BudgetManager(500, clock=clock),
        top_n=2,
    )

    executed = service.run(candidates(problem))

    assert [call.model_name for call in evaluator.calls] == [
        "logistic_regression",
        "logistic_regression",
        "random_forest",
        "random_forest",
        "dummy",
        "dummy",
    ]
    assert [call.fidelity.level for call in evaluator.calls] == [2, 3, 2, 3, 2, 3]
    assert len(executed) == 6
    for full, confirmation in zip(evaluator.calls[::2], evaluator.calls[1::2], strict=True):
        assert full.fidelity == problem.policy.level(2)
        assert len(full.fidelity.seeds) == 1
        assert confirmation.fidelity == problem.policy.level(3)
        assert len(confirmation.fidelity.seeds) == 3


def test_confirmation_never_consumes_the_finalization_reserve(tmp_path: Path) -> None:
    problem = make_problem(tmp_path, budget_seconds=100)
    clock = ManualClock()
    evaluator = FakeEvaluator(clock=clock, elapsed_seconds=10.0)
    budget = BudgetManager(
        100,
        confirmation_fraction=0.10,
        finalization_fraction=0.20,
        clock=clock,
    )
    service = confirmer(
        problem,
        evaluator,
        budget,
        top_n=5,
        estimated_trial_seconds=10.0,
    )

    service.run(candidates(problem))

    assert len(evaluator.calls) == 8
    assert clock.value == pytest.approx(80.0)
    assert budget.remaining_seconds == pytest.approx(20.0)
    assert budget.available_for(BudgetPhase.CONFIRMATION) == pytest.approx(0.0)
    assert budget.available_for(BudgetPhase.FINALIZATION) == pytest.approx(20.0)
    progress = problem.store.get_run_progress(problem.run_id)
    assert progress.consumed_seconds == pytest.approx(80.0)


def test_one_failed_spec_is_persisted_and_next_finalists_continue(tmp_path: Path) -> None:
    problem = make_problem(tmp_path)
    clock = ManualClock()
    evaluator = FakeEvaluator(clock=clock, failing_models={"logistic_regression"})
    service = confirmer(
        problem,
        evaluator,
        BudgetManager(500, clock=clock),
        top_n=2,
    )

    executed = service.run(candidates(problem))

    assert [call.model_name for call in evaluator.calls] == [
        "logistic_regression",
        "random_forest",
        "random_forest",
        "dummy",
        "dummy",
    ]
    assert [call.fidelity.level for call in evaluator.calls] == [2, 2, 3, 2, 3]
    assert executed[0].status is TrialStatus.FAILED
    assert executed[0].pipeline_spec.model_name == "logistic_regression"
    assert all(result.status is TrialStatus.COMPLETED for result in executed[1:])
    assert problem.store.get_trial(executed[0].trial_id).status is TrialStatus.FAILED


def test_second_run_is_idempotent_and_executes_no_completed_work(tmp_path: Path) -> None:
    problem = make_problem(tmp_path)
    clock = ManualClock()
    evaluator = FakeEvaluator(clock=clock)
    service = confirmer(
        problem,
        evaluator,
        BudgetManager(500, clock=clock),
        top_n=1,
    )

    first = service.run(candidates(problem))
    call_count = len(evaluator.calls)
    second = service.run(candidates(problem))

    assert len(first) == 4
    assert second == []
    assert len(evaluator.calls) == call_count
    assert len(problem.store.list_trials(problem.run_id)) == 4


def test_changed_test_frame_cannot_change_shortlist_or_confirmation_ids(
    tmp_path: Path,
) -> None:
    first_problem = make_problem(
        tmp_path / "first",
        run_id="same-run",
        test_signal=-100.0,
    )
    changed_problem = make_problem(
        tmp_path / "changed",
        run_id="same-run",
        test_signal=1_000_000.0,
    )
    first_clock = ManualClock()
    changed_clock = ManualClock()
    first_evaluator = FakeEvaluator(clock=first_clock)
    changed_evaluator = FakeEvaluator(clock=changed_clock)

    confirmer(
        first_problem,
        first_evaluator,
        BudgetManager(500, clock=first_clock),
        top_n=2,
    ).run(candidates(first_problem))
    confirmer(
        changed_problem,
        changed_evaluator,
        BudgetManager(500, clock=changed_clock),
        top_n=2,
    ).run(candidates(changed_problem))

    assert first_problem.profile == changed_problem.profile
    assert [call.model_name for call in first_evaluator.calls] == [
        call.model_name for call in changed_evaluator.calls
    ]
    assert [call.trial_id for call in first_evaluator.calls] == [
        call.trial_id for call in changed_evaluator.calls
    ]


def test_incompatible_persisted_level_three_seeds_are_not_accepted_as_confirmation(
    tmp_path: Path,
) -> None:
    problem = make_problem(tmp_path)
    specification = candidates(problem)[0].pipeline_spec
    full = search_result(specification, problem.policy.level(2), score=0.90).model_copy(
        update={"trial_id": "persisted-full"},
        deep=True,
    )
    wrong_confirmation_fidelity = FidelitySpec(
        level=3,
        sample_fraction=1.0,
        n_folds=problem.plan.n_splits,
        max_iterations=problem.policy.level(3).max_iterations,
        seeds=[11, 12, 13],
    )
    assert wrong_confirmation_fidelity != problem.policy.level(3)
    wrong_confirmation = search_result(
        specification,
        wrong_confirmation_fidelity,
        score=0.99,
    ).model_copy(update={"trial_id": "persisted-wrong-confirmation"}, deep=True)
    persist_result(problem, full)
    persist_result(problem, wrong_confirmation)
    clock = ManualClock()
    evaluator = FakeEvaluator(clock=clock)
    service = confirmer(
        problem,
        evaluator,
        BudgetManager(500, clock=clock),
        top_n=1,
    )

    executed = service.run([candidates(problem)[0]])

    assert len(executed) == 1
    assert len(evaluator.calls) == 1
    assert evaluator.calls[0].fidelity == problem.policy.level(3)
    assert executed[0].fidelity.seeds == problem.policy.level(3).seeds
    persisted = problem.store.list_trials(problem.run_id)
    assert sum(result.fidelity.level == 3 for result in persisted) == 2

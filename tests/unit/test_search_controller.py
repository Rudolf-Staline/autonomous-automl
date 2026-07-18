"""End-to-end unit tests for the persisted sequential search controller."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from autonomous_automl.contracts import (
    AutoMLConfig,
    DatasetProfile,
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
from autonomous_automl.profiling import detect_leakage, profile_dataset
from autonomous_automl.search.allocator import FamilyAllocator
from autonomous_automl.search.budget import BudgetManager, BudgetPhase
from autonomous_automl.search.controller import SearchController
from autonomous_automl.search.fidelity import FidelityPolicy, FidelityScheduler
from autonomous_automl.search.optimizer import SearchCandidate
from autonomous_automl.tracking.store import ExperimentStore, RetryableTrial
from autonomous_automl.validation import ValidationPlanner, audit_validation


class ManualClock:
    """Monotonic test clock advanced only by the fake evaluator."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


@dataclass(slots=True)
class FakeOutcome:
    trial_result: TrialResult


@dataclass(frozen=True, slots=True)
class EvaluationCall:
    trial_id: str
    family: str
    fidelity_level: int
    timeout_seconds: float | None


class RecordingEvaluator:
    """PipelineEvaluator-shaped test double returning real TrialResult contracts."""

    def __init__(
        self,
        scores: Sequence[float],
        *,
        clock: ManualClock | None = None,
        elapsed_seconds: float = 0.0,
        raising_calls: set[int] | None = None,
        interrupting_calls: set[int] | None = None,
    ) -> None:
        self.scores = list(scores)
        self.clock = clock
        self.elapsed_seconds = elapsed_seconds
        self.raising_calls = set(raising_calls or set())
        self.interrupting_calls = set(interrupting_calls or set())
        self.calls: list[EvaluationCall] = []

    def evaluate(
        self,
        trial_id: str,
        pipeline_spec: PipelineSpec,
        fidelity: FidelitySpec,
        dataset: LoadedDataset,
        profile: DatasetProfile,
        validation_plan: ValidationPlan,
        metric: MetricName | str = MetricName.AUTO,
        *,
        timeout_seconds: float | None = None,
    ) -> FakeOutcome:
        del dataset, profile, validation_plan
        call_index = len(self.calls)
        self.calls.append(
            EvaluationCall(
                trial_id,
                pipeline_spec.family,
                fidelity.level,
                timeout_seconds,
            )
        )
        if self.clock is not None:
            self.clock.advance(self.elapsed_seconds)
        if call_index in self.interrupting_calls:
            raise KeyboardInterrupt
        if call_index in self.raising_calls:
            raise RuntimeError(f"synthetic pipeline failure {call_index}")
        score = self.scores[call_index]
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
                std_score=0.0,
                fold_scores=[score],
                fit_seconds=self.elapsed_seconds,
                predict_seconds=0.0,
                started_at=now,
                finished_at=now,
            )
        )


class RecordingOptimizer:
    """Persistent-identity ask/tell double with SearchCandidate contracts."""

    def __init__(
        self,
        family: str,
        template: PipelineSpec,
        *,
        known_candidates: Iterable[SearchCandidate] = (),
    ) -> None:
        self.family = family
        self.template = template
        self.ask_count = 0
        self.told: list[tuple[SearchCandidate, TrialResult]] = []
        self.reconciled: list[TrialResult] = []
        self.candidates = {candidate.candidate_id: candidate for candidate in known_candidates}

    def ask(self, fidelity: FidelitySpec) -> SearchCandidate:
        number = self.ask_count
        self.ask_count += 1
        specification = self.template.model_copy(
            update={"random_seed": self.template.random_seed + number},
            deep=True,
        )
        candidate = SearchCandidate(
            candidate_id=f"{self.family}-candidate-{number}",
            family=self.family,
            model_name=specification.model_name,
            study_name=f"test:{self.family}",
            trial_number=number,
            template_fingerprint=f"template-{self.family}",
            pipeline_spec=specification,
            fidelity=fidelity,
        )
        self.candidates[candidate.candidate_id] = candidate
        return candidate

    def tell(self, candidate: SearchCandidate, result: TrialResult) -> None:
        assert candidate.candidate_id == result.trial_id
        assert candidate.pipeline_spec == result.pipeline_spec
        assert candidate.fidelity == result.fidelity
        self.told.append((candidate, result))

    def reconcile_results(self, results: Iterable[TrialResult]) -> None:
        self.reconciled.extend(results)

    def candidate_by_id(
        self,
        candidate_id: str,
        pipeline_spec: PipelineSpec,
        fidelity: FidelitySpec,
    ) -> SearchCandidate | None:
        candidate = self.candidates.get(candidate_id)
        if candidate is None:
            return None
        assert candidate.pipeline_spec == pipeline_spec
        assert candidate.fidelity == fidelity
        return candidate


class FailingAskOptimizer(RecordingOptimizer):
    """Optimizer whose generation failure must release allocator pending state."""

    def ask(self, fidelity: FidelitySpec) -> SearchCandidate:
        del fidelity
        self.ask_count += 1
        raise ValueError("sensitive/user/path.csv must not reach persisted failures")


@dataclass(frozen=True, slots=True)
class SearchProblem:
    run_id: str
    dataset: LoadedDataset
    profile: DatasetProfile
    plan: ValidationPlan
    store: ExperimentStore
    templates: dict[str, PipelineSpec]


def _problem(tmp_path: Path, *, run_id: str = "controller-run") -> SearchProblem:
    source = tmp_path / f"{run_id}.csv"
    frame = pd.DataFrame(
        {
            "number": [float(index % 7) for index in range(30)],
            "category": [f"c-{index % 3}" for index in range(30)],
            "target": [index % 2 for index in range(30)],
        }
    )
    frame.to_csv(source, index=False)
    config = AutoMLConfig(
        target="target",
        task=TaskType.BINARY_CLASSIFICATION,
        metric=MetricName.ACCURACY,
        budget_seconds=100,
        random_seed=42,
        output_dir=tmp_path / run_id,
    )
    dataset = load_dataset(source, config)
    profile = profile_dataset(dataset, TaskType.BINARY_CLASSIFICATION)
    leakage = detect_leakage(dataset, profile)
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
        leakage_report=leakage,
        validation_plan=plan,
        validation_audit=audit_validation(dataset, plan),
        dependency_versions={"python": "3.12"},
        random_seed=42,
    )
    store = ExperimentStore.open(tmp_path / run_id / "registry.sqlite3")
    store.create_run(manifest)
    common = {
        "task": TaskType.BINARY_CLASSIFICATION,
        "numeric_imputer": "median",
        "numeric_scaler": "standard",
        "categorical_imputer": "constant",
        "categorical_encoder": "one_hot",
        "datetime_transformer": "drop",
        "feature_selector": None,
        "model_params": {},
        "excluded_columns": [],
        "random_seed": 100,
    }
    templates = {
        "baseline": PipelineSpec(
            family="baseline",
            model_name="dummy",
            **common,
        ),
        "linear": PipelineSpec(
            family="linear",
            model_name="logistic_regression",
            **common,
        ),
    }
    return SearchProblem(run_id, dataset, profile, plan, store, templates)


def _scheduler(problem: SearchProblem, *, reduction_factor: int = 3) -> FidelityScheduler:
    return FidelityScheduler(
        FidelityPolicy(
            available_folds=len(problem.plan.folds),
            random_seed=42,
            confirmation_seed_count=2,
        ),
        reduction_factor=reduction_factor,
    )


def _scheduler_with_low_promotion(problem: SearchProblem) -> FidelityScheduler:
    scheduler = _scheduler(problem, reduction_factor=2)
    fidelity = scheduler.initial_level()
    now = datetime.now(UTC)
    for index, score in enumerate((0.9, 0.7)):
        specification = problem.templates["linear"].model_copy(
            update={"random_seed": 300 + index},
            deep=True,
        )
        scheduler.observe(
            TrialResult(
                trial_id=f"seed-low-{index}",
                family="linear",
                pipeline_spec=specification,
                fidelity=fidelity,
                status=TrialStatus.COMPLETED,
                primary_metric=MetricName.ACCURACY,
                mean_score=score,
                std_score=0.0,
                fold_scores=[score],
                started_at=now,
                finished_at=now,
            )
        )
    assert scheduler.pending_promotions()
    return scheduler


def _controller(
    problem: SearchProblem,
    optimizers: dict[str, RecordingOptimizer],
    allocator: FamilyAllocator,
    scheduler: FidelityScheduler,
    evaluator: RecordingEvaluator,
    budget: BudgetManager,
    *,
    clock: ManualClock,
    retryable_trials: Sequence[RetryableTrial] = (),
) -> SearchController:
    return SearchController(
        run_id=problem.run_id,
        dataset=problem.dataset,
        profile=problem.profile,
        validation_plan=problem.plan,
        metric=MetricName.ACCURACY,
        optimizers=optimizers,
        allocator=allocator,
        scheduler=scheduler,
        evaluator=evaluator,
        store=problem.store,
        budget=budget,
        budget_clock=clock,
        retryable_trials=retryable_trials,
    )


def test_controller_explores_baseline_and_each_family_at_low_fidelity(
    tmp_path: Path,
) -> None:
    problem = _problem(tmp_path)
    clock = ManualClock()
    optimizers = {
        name: RecordingOptimizer(name, problem.templates[name]) for name in ("baseline", "linear")
    }
    allocator = FamilyAllocator(["baseline", "linear"])
    scheduler = _scheduler(problem, reduction_factor=10)
    evaluator = RecordingEvaluator([0.4, 0.7], clock=clock)
    budget = BudgetManager(100, clock=clock)

    results = _controller(
        problem,
        optimizers,
        allocator,
        scheduler,
        evaluator,
        budget,
        clock=clock,
    ).run(max_trials=2)

    assert [call.family for call in evaluator.calls] == ["baseline", "linear"]
    assert [call.fidelity_level for call in evaluator.calls] == [0, 0]
    assert [result.status for result in results] == [
        TrialStatus.COMPLETED,
        TrialStatus.COMPLETED,
    ]
    assert len(optimizers["baseline"].told) == 1
    assert len(optimizers["linear"].told) == 1
    states = problem.store.load_component_states(problem.run_id)
    assert states == {
        "allocator": allocator.snapshot(),
        "budget": budget.snapshot(),
        "scheduler": scheduler.snapshot(),
    }
    assert problem.store.get_run_progress(problem.run_id).checkpoint_revision == 2


def test_pending_promotion_runs_before_another_optimizer_suggestion(tmp_path: Path) -> None:
    problem = _problem(tmp_path)
    clock = ManualClock()
    optimizer = RecordingOptimizer("linear", problem.templates["linear"])
    allocator = FamilyAllocator(["linear"])
    scheduler = _scheduler(problem, reduction_factor=2)
    evaluator = RecordingEvaluator([0.9, 0.5, 0.8], clock=clock)

    results = _controller(
        problem,
        {"linear": optimizer},
        allocator,
        scheduler,
        evaluator,
        BudgetManager(100, clock=clock),
        clock=clock,
    ).run(max_trials=3)

    assert [call.fidelity_level for call in evaluator.calls] == [0, 0, 1]
    assert optimizer.ask_count == 2
    assert len(optimizer.told) == 2
    assert results[2].trial_id.startswith("promotion-")
    assert results[2].pipeline_spec == results[0].pipeline_spec


def test_failed_pipeline_is_recorded_and_does_not_stop_the_search(tmp_path: Path) -> None:
    problem = _problem(tmp_path)
    clock = ManualClock()
    optimizer = RecordingOptimizer("linear", problem.templates["linear"])
    evaluator = RecordingEvaluator(
        [0.8, 0.9],
        clock=clock,
        raising_calls={0},
    )

    results = _controller(
        problem,
        {"linear": optimizer},
        FamilyAllocator(["linear"]),
        _scheduler(problem, reduction_factor=10),
        evaluator,
        BudgetManager(100, clock=clock),
        clock=clock,
    ).run(max_trials=2)

    assert [result.status for result in results] == [
        TrialStatus.FAILED,
        TrialStatus.COMPLETED,
    ]
    assert results[0].failure_type == "RuntimeError"
    assert results[0].failure_message == "pipeline evaluation failed (RuntimeError)"
    assert "synthetic pipeline failure" not in (results[0].failure_message or "")
    assert len(evaluator.calls) == 2
    assert len(optimizer.told) == 2
    assert problem.store.list_trials(problem.run_id) == results


def test_confirmation_promotion_can_use_confirmation_but_not_final_reserve(
    tmp_path: Path,
) -> None:
    problem = _problem(tmp_path)
    clock = ManualClock()
    scheduler = _scheduler(problem, reduction_factor=2)
    full = scheduler.policy.level(2)
    now = datetime.now(UTC)
    for index, score in enumerate((0.9, 0.8)):
        specification = problem.templates["linear"].model_copy(
            update={"random_seed": 200 + index},
            deep=True,
        )
        scheduler.observe(
            TrialResult(
                trial_id=f"full-{index}",
                family="linear",
                pipeline_spec=specification,
                fidelity=full,
                status=TrialStatus.COMPLETED,
                primary_metric=MetricName.ACCURACY,
                mean_score=score,
                std_score=0.0,
                fold_scores=[score],
                started_at=now,
                finished_at=now,
            )
        )
    assert scheduler.pending_promotions()[0].to_fidelity.level == 3
    optimizer = RecordingOptimizer("linear", problem.templates["linear"])
    evaluator = RecordingEvaluator([0.85], clock=clock)
    budget = BudgetManager(
        10,
        confirmation_fraction=0.1,
        finalization_fraction=0.2,
        clock=clock,
    )

    results = _controller(
        problem,
        {"linear": optimizer},
        FamilyAllocator(["linear"]),
        scheduler,
        evaluator,
        budget,
        clock=clock,
    ).run(max_trials=1)

    assert len(results) == 1
    assert evaluator.calls[0].fidelity_level == 3
    assert evaluator.calls[0].timeout_seconds == pytest.approx(8.0)
    assert evaluator.calls[0].timeout_seconds > budget.available_for(BudgetPhase.EXPLORATION)
    assert len(optimizer.told) == 0


def test_interrupted_ask_retries_then_terminal_resume_does_not_repeat_it(
    tmp_path: Path,
) -> None:
    problem = _problem(tmp_path)
    initial_scheduler = _scheduler(problem)
    initial_optimizer = RecordingOptimizer("linear", problem.templates["linear"])
    candidate = initial_optimizer.ask(initial_scheduler.initial_level())
    problem.store.reserve_trial(
        problem.run_id,
        candidate.candidate_id,
        candidate.pipeline_spec,
        candidate.fidelity,
        MetricName.ACCURACY,
    )
    problem.store.mark_stale_trials_interrupted(problem.run_id, stale_after_seconds=0)
    retryable = problem.store.list_retryable_trials(problem.run_id)
    assert [trial.trial_id for trial in retryable] == [candidate.candidate_id]

    retry_clock = ManualClock()
    retry_optimizer = RecordingOptimizer(
        "linear",
        problem.templates["linear"],
        known_candidates=[candidate],
    )
    retry_allocator = FamilyAllocator(["linear"])
    retry_scheduler = _scheduler(problem)
    retry_evaluator = RecordingEvaluator([0.75], clock=retry_clock)
    retried = _controller(
        problem,
        {"linear": retry_optimizer},
        retry_allocator,
        retry_scheduler,
        retry_evaluator,
        BudgetManager(100, clock=retry_clock),
        clock=retry_clock,
        retryable_trials=retryable,
    ).run(max_trials=1)

    assert [result.trial_id for result in retried] == [candidate.candidate_id]
    assert [call.trial_id for call in retry_evaluator.calls] == [candidate.candidate_id]
    assert len(retry_optimizer.told) == 1
    assert problem.store.list_retryable_trials(problem.run_id) == []
    persisted_states = problem.store.load_component_states(problem.run_id)

    resumed_clock = ManualClock()
    resumed_optimizer = RecordingOptimizer(
        "linear",
        problem.templates["linear"],
        known_candidates=[candidate],
    )
    resumed_allocator = FamilyAllocator(["linear"])
    resumed_scheduler = _scheduler(problem)
    resumed_evaluator = RecordingEvaluator([0.99], clock=resumed_clock)
    resumed_controller = _controller(
        problem,
        {"linear": resumed_optimizer},
        resumed_allocator,
        resumed_scheduler,
        resumed_evaluator,
        BudgetManager(100, clock=resumed_clock),
        clock=resumed_clock,
    )

    assert resumed_controller.run(max_trials=1) == []
    assert resumed_evaluator.calls == []
    assert resumed_optimizer.reconciled == retried
    assert resumed_allocator.snapshot() == persisted_states["allocator"]
    assert resumed_scheduler.snapshot() == persisted_states["scheduler"]
    assert resumed_controller.budget.snapshot() == persisted_states["budget"]


def test_interrupted_promotion_retries_before_queue_and_is_not_told_to_optuna(
    tmp_path: Path,
) -> None:
    problem = _problem(tmp_path)
    first_clock = ManualClock()
    first_scheduler = _scheduler_with_low_promotion(problem)
    first_optimizer = RecordingOptimizer("linear", problem.templates["linear"])
    interrupted_evaluator = RecordingEvaluator(
        [0.8],
        clock=first_clock,
        interrupting_calls={0},
    )
    first_controller = _controller(
        problem,
        {"linear": first_optimizer},
        FamilyAllocator(["linear"]),
        first_scheduler,
        interrupted_evaluator,
        BudgetManager(100, clock=first_clock),
        clock=first_clock,
    )

    with pytest.raises(KeyboardInterrupt):
        first_controller.run(max_trials=1)
    problem.store.mark_stale_trials_interrupted(problem.run_id, stale_after_seconds=0)
    retryable = problem.store.list_retryable_trials(problem.run_id)
    assert len(retryable) == 1
    assert retryable[0].fidelity.level == 1

    retry_clock = ManualClock()
    retry_scheduler = _scheduler_with_low_promotion(problem)
    retry_optimizer = RecordingOptimizer("linear", problem.templates["linear"])
    retry_evaluator = RecordingEvaluator([0.82], clock=retry_clock)
    retried = _controller(
        problem,
        {"linear": retry_optimizer},
        FamilyAllocator(["linear"]),
        retry_scheduler,
        retry_evaluator,
        BudgetManager(100, clock=retry_clock),
        clock=retry_clock,
        retryable_trials=retryable,
    ).run(max_trials=1)

    assert [result.trial_id for result in retried] == [retryable[0].trial_id]
    assert [call.fidelity_level for call in retry_evaluator.calls] == [1]
    assert retry_scheduler.pending_promotions() == ()
    assert retry_optimizer.told == []


def test_restored_budget_accepts_an_explicitly_increased_total(tmp_path: Path) -> None:
    problem = _problem(tmp_path)
    first_clock = ManualClock()
    first_optimizer = RecordingOptimizer("linear", problem.templates["linear"])
    _controller(
        problem,
        {"linear": first_optimizer},
        FamilyAllocator(["linear"]),
        _scheduler(problem, reduction_factor=10),
        RecordingEvaluator([0.7], clock=first_clock),
        BudgetManager(100, clock=first_clock),
        clock=first_clock,
    ).run(max_trials=1)
    progress = problem.store.add_budget(problem.run_id, 50)
    assert progress.budget_seconds == 150

    resumed_clock = ManualClock()
    resumed = _controller(
        problem,
        {"linear": RecordingOptimizer("linear", problem.templates["linear"])},
        FamilyAllocator(["linear"]),
        _scheduler(problem, reduction_factor=10),
        RecordingEvaluator([0.9], clock=resumed_clock),
        BudgetManager(150, clock=resumed_clock),
        clock=resumed_clock,
    )

    assert resumed.budget.total_seconds == 150
    assert resumed.budget.consumed_seconds == pytest.approx(0.0)
    assert resumed.run(max_trials=1) == []


def test_ask_failure_releases_pending_and_tries_another_family(tmp_path: Path) -> None:
    problem = _problem(tmp_path)
    clock = ManualClock()
    failing = FailingAskOptimizer("baseline", problem.templates["baseline"])
    working = RecordingOptimizer("linear", problem.templates["linear"])
    allocator = FamilyAllocator(["baseline", "linear"])
    evaluator = RecordingEvaluator([0.8], clock=clock)

    results = _controller(
        problem,
        {"baseline": failing, "linear": working},
        allocator,
        _scheduler(problem, reduction_factor=10),
        evaluator,
        BudgetManager(100, clock=clock),
        clock=clock,
    ).run(max_trials=1)

    assert [result.family for result in results] == ["linear"]
    assert allocator.state_for("baseline").pending == 0
    assert failing.ask_count == 1
    assert working.ask_count == 1
    assert [call.family for call in evaluator.calls] == ["linear"]


def test_duplicate_terminal_candidate_is_reconciled_then_search_continues(
    tmp_path: Path,
) -> None:
    problem = _problem(tmp_path)
    first_clock = ManualClock()
    first_optimizer = RecordingOptimizer("linear", problem.templates["linear"])
    _controller(
        problem,
        {"linear": first_optimizer},
        FamilyAllocator(["linear"]),
        _scheduler(problem, reduction_factor=10),
        RecordingEvaluator([0.6], clock=first_clock),
        BudgetManager(100, clock=first_clock),
        clock=first_clock,
    ).run(max_trials=1)

    resumed_clock = ManualClock()
    resumed_optimizer = RecordingOptimizer("linear", problem.templates["linear"])
    resumed_evaluator = RecordingEvaluator([0.8], clock=resumed_clock)
    results = _controller(
        problem,
        {"linear": resumed_optimizer},
        FamilyAllocator(["linear"]),
        _scheduler(problem, reduction_factor=10),
        resumed_evaluator,
        BudgetManager(100, clock=resumed_clock),
        clock=resumed_clock,
    ).run(max_trials=2)

    assert len(results) == 1
    assert results[0].trial_id == "linear-candidate-1"
    assert [call.trial_id for call in resumed_evaluator.calls] == ["linear-candidate-1"]
    assert [candidate.candidate_id for candidate, _ in resumed_optimizer.told] == [
        "linear-candidate-0",
        "linear-candidate-1",
    ]

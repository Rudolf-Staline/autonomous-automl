"""Sequential search orchestration across allocation, fidelity, budget, and tracking."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, cast

from autonomous_automl.contracts import (
    DatasetProfile,
    FidelitySpec,
    MetricName,
    PipelineSpec,
    SearchStopReason,
    TaskType,
    TrialResult,
    TrialStatus,
    ValidationPlan,
)
from autonomous_automl.data import LoadedDataset
from autonomous_automl.evaluation.metrics import resolve_metric
from autonomous_automl.search.allocator import FamilyAllocator
from autonomous_automl.search.budget import BudgetManager, BudgetPhase
from autonomous_automl.search.fidelity import FidelityScheduler, Promotion
from autonomous_automl.search.optimizer import SearchCandidate
from autonomous_automl.tracking.store import ExperimentStore, RetryableTrial
from autonomous_automl.utils.errors import ResumeError
from autonomous_automl.utils.hashing import sha256_json
from autonomous_automl.utils.json import JsonValue

_CONTROLLER_COMPONENTS = frozenset({"allocator", "scheduler", "budget"})
_PROMOTION_PREFIX = "promotion-"
_OPTUNA_STATUSES = {TrialStatus.COMPLETED, TrialStatus.FAILED, TrialStatus.PRUNED}
TrialCompletedCallback = Callable[[TrialResult, BudgetManager], None]


class EvaluationOutcomeLike(Protocol):
    """Structural subset returned by :class:`PipelineEvaluator`."""

    trial_result: TrialResult


class TrialEvaluator(Protocol):
    """Injectable evaluator boundary used by the search loop and its tests."""

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
    ) -> EvaluationOutcomeLike: ...


class FamilyOptimizer(Protocol):
    """Ask/tell surface implemented by ``OptunaFamilyOptimizer``."""

    family: str

    def ask(self, fidelity: FidelitySpec) -> SearchCandidate: ...

    def tell(self, candidate: SearchCandidate, result: TrialResult) -> None: ...

    def reconcile_results(self, results: Iterable[TrialResult]) -> None: ...

    def candidate_by_id(
        self,
        candidate_id: str,
        pipeline_spec: PipelineSpec,
        fidelity: FidelitySpec,
    ) -> SearchCandidate | None: ...


@dataclass(frozen=True, slots=True)
class _WorkItem:
    proposed_trial_id: str
    pipeline_spec: PipelineSpec
    fidelity: FidelitySpec
    candidate: SearchCandidate | None


class _CandidateGenerationError(RuntimeError):
    def __init__(self, family: str) -> None:
        super().__init__(f"candidate generation failed for family {family!r}")
        self.family = family


class SearchController:
    """Run a crash-safe, budgeted, multi-family successive-halving search."""

    def __init__(
        self,
        *,
        run_id: str,
        dataset: LoadedDataset,
        profile: DatasetProfile,
        validation_plan: ValidationPlan,
        metric: MetricName | str,
        optimizers: Mapping[str, FamilyOptimizer],
        allocator: FamilyAllocator,
        scheduler: FidelityScheduler,
        evaluator: TrialEvaluator,
        store: ExperimentStore,
        budget: BudgetManager,
        trial_timeout_seconds: float | None = None,
        estimated_trial_seconds: float = 0.0,
        safety_margin_seconds: float = 0.0,
        lease_epoch: int | None = None,
        restore_state: bool = True,
        budget_clock: Callable[[], float] | None = None,
        retryable_trials: Sequence[RetryableTrial] = (),
        trial_completed_callback: TrialCompletedCallback | None = None,
    ) -> None:
        if not run_id.strip():
            raise ValueError("run_id cannot be blank")
        if trial_timeout_seconds is not None and trial_timeout_seconds <= 0:
            raise ValueError("trial_timeout_seconds must be positive")
        if estimated_trial_seconds < 0 or safety_margin_seconds < 0:
            raise ValueError("trial estimate and safety margin must be non-negative")
        if not optimizers:
            raise ValueError("at least one family optimizer is required")

        self.run_id = run_id
        self.dataset = dataset
        self.profile = profile
        self.validation_plan = validation_plan
        self.metric = resolve_metric(metric, profile.inferred_task, profile).name
        self.optimizers = dict(optimizers)
        self.allocator = allocator
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.store = store
        self.budget = budget
        self.trial_timeout_seconds = trial_timeout_seconds
        self.estimated_trial_seconds = estimated_trial_seconds
        self.safety_margin_seconds = safety_margin_seconds
        self._budget_clock = budget_clock
        self._retryable_trials = sorted(retryable_trials, key=lambda trial: trial.sequence_number)
        self.trial_completed_callback = trial_completed_callback
        self.stop_reason: SearchStopReason | None = None

        self._validate_components()
        self._validate_retries()
        progress = self.store.get_run_progress(run_id)
        if lease_epoch is not None and lease_epoch != progress.lease_epoch:
            raise ResumeError("controller lease epoch differs from the run registry")
        self.lease_epoch = progress.lease_epoch if lease_epoch is None else lease_epoch
        if restore_state:
            self._restore()

    def run(self, *, max_trials: int | None = None) -> list[TrialResult]:
        """Execute until the total trial cap or the current phase budget is exhausted."""
        if max_trials is not None and max_trials < 0:
            raise ValueError("max_trials must be non-negative")
        existing_count = len(self.store.list_trials(self.run_id))
        completed_here: list[TrialResult] = []
        duplicate_keys: set[str] = set()
        generation_failures = 0
        unavailable_families: set[str] = set()
        generation_failure_limit = len(self.optimizers) * self.allocator.failure_threshold
        self.stop_reason = None

        while max_trials is None or existing_count + len(completed_here) < max_trials:
            retry = self._retryable_trials[0] if self._retryable_trials else None
            pending = self.scheduler.pending_promotions()
            promotion = pending[0] if retry is None and pending else None
            next_fidelity = (
                retry.fidelity
                if retry is not None
                else (
                    promotion.to_fidelity
                    if promotion is not None
                    else self.scheduler.initial_level()
                )
            )
            phase = _phase_for_fidelity(next_fidelity)
            timeout = self._timeout_for(phase)
            if timeout is None:
                self.stop_reason = SearchStopReason.BUDGET_EXHAUSTED
                break

            if retry is not None:
                self._retryable_trials.pop(0)
                work = self._retry_work(retry)
            elif promotion is not None:
                popped = self.scheduler.pop_next_promotion()
                if popped != promotion:
                    raise RuntimeError("fidelity promotion queue changed unexpectedly")
                work = _promotion_work(promotion)
            else:
                available = [
                    family for family in self.optimizers if family not in unavailable_families
                ]
                if not available:
                    unavailable_families.clear()
                    available = list(self.optimizers)
                try:
                    work = self._suggest_low_fidelity(available)
                except _CandidateGenerationError as error:
                    generation_failures += 1
                    unavailable_families.add(error.family)
                    if generation_failures >= generation_failure_limit:
                        raise RuntimeError(
                            "candidate generation failed repeatedly across optimizer families"
                        ) from None
                    continue
                generation_failures = 0
                unavailable_families.clear()

            reservation = self.store.reserve_trial(
                self.run_id,
                work.proposed_trial_id,
                work.pipeline_spec,
                work.fidelity,
                self.metric,
            )
            if not reservation.reserved:
                self._reconcile_duplicate(work, reservation.trial_id)
                existing_count = len(self.store.list_trials(self.run_id))
                if reservation.candidate_key in duplicate_keys:
                    self.stop_reason = SearchStopReason.SEARCH_SPACE_EXHAUSTED
                    break
                duplicate_keys.add(reservation.candidate_key)
                continue
            duplicate_keys.clear()
            if work.candidate is not None and reservation.trial_id != work.candidate.candidate_id:
                raise ResumeError(
                    "an interrupted Optuna candidate cannot be rebound to a different ask trial"
                )

            consumed_before = self.budget.consumed_seconds
            result = self._evaluate(
                reservation.trial_id,
                work.pipeline_spec,
                work.fidelity,
                timeout,
            )
            self.allocator.update(result)
            self.scheduler.observe(result)
            snapshots = self._snapshots()
            budget_snapshot = _state_object(snapshots["budget"], "budget")
            consumed_after = float(cast(float | int, budget_snapshot["consumed_seconds"]))
            consumed_delta = max(0.0, consumed_after - consumed_before)
            self.store.record_trial_result(
                result,
                attempt_token=reservation.attempt_token,
                lease_epoch=self.lease_epoch,
                component_states=snapshots,
                consumed_seconds=consumed_delta,
            )
            if work.candidate is not None:
                self.optimizers[work.candidate.family].tell(work.candidate, result)
            completed_here.append(result)
            if self.trial_completed_callback is not None:
                self.trial_completed_callback(result, self.budget)

        if self.stop_reason is None:
            self.stop_reason = SearchStopReason.CONTROLLER_STOPPED
        return completed_here

    def _retry_work(self, retry: RetryableTrial) -> _WorkItem:
        family = retry.pipeline_spec.family
        optimizer = self.optimizers[family]
        candidate = optimizer.candidate_by_id(
            retry.trial_id,
            retry.pipeline_spec,
            retry.fidelity,
        )
        if retry.fidelity.level == 0 and candidate is None:
            raise ResumeError("low-fidelity retry has no matching persisted Optuna candidate")
        if retry.fidelity.level > 0 and candidate is not None:
            raise ResumeError("promoted retry must not reference an Optuna ask candidate")
        if retry.fidelity.level == 0 and self.allocator.state_for(family).pending == 0:
            selected = self.allocator.select_next([family])
            if selected != family:
                raise RuntimeError("allocator selected the wrong retry family")
        elif retry.fidelity.level > 0:
            _consume_matching_promotion(self.scheduler, retry)
        return _WorkItem(
            proposed_trial_id=retry.trial_id,
            pipeline_spec=retry.pipeline_spec,
            fidelity=retry.fidelity,
            candidate=candidate,
        )

    def _suggest_low_fidelity(self, available_families: Sequence[str]) -> _WorkItem:
        family = self.allocator.select_next(available_families)
        optimizer = self.optimizers[family]
        fidelity = self.scheduler.initial_level()
        try:
            candidate = optimizer.ask(fidelity)
            if candidate.family != family or candidate.pipeline_spec.family != family:
                raise ValueError("optimizer returned a candidate from a different family")
            if candidate.fidelity != fidelity:
                raise ValueError("new optimizer candidates must start at low fidelity")
        except Exception:
            self.allocator.cancel_pending(family)
            raise _CandidateGenerationError(family) from None
        return _WorkItem(
            proposed_trial_id=candidate.candidate_id,
            pipeline_spec=candidate.pipeline_spec,
            fidelity=candidate.fidelity,
            candidate=candidate,
        )

    def _evaluate(
        self,
        trial_id: str,
        pipeline_spec: PipelineSpec,
        fidelity: FidelitySpec,
        timeout_seconds: float,
    ) -> TrialResult:
        started_at = datetime.now(UTC)
        try:
            outcome = self.evaluator.evaluate(
                trial_id,
                pipeline_spec,
                fidelity,
                self.dataset,
                self.profile,
                self.validation_plan,
                self.metric,
                timeout_seconds=timeout_seconds,
            )
            result = outcome.trial_result
        except Exception as error:
            failure_type = _safe_failure_type(error)
            result = TrialResult(
                trial_id=trial_id,
                family=pipeline_spec.family,
                pipeline_spec=pipeline_spec,
                fidelity=fidelity,
                status=TrialStatus.FAILED,
                primary_metric=self.metric,
                failure_type=failure_type,
                failure_message=f"pipeline evaluation failed ({failure_type})",
                started_at=started_at,
                finished_at=datetime.now(UTC),
            )
        _validate_result_identity(result, trial_id, pipeline_spec, fidelity, self.metric)
        return result

    def _timeout_for(self, phase: BudgetPhase) -> float | None:
        if not self.budget.can_start(
            phase,
            estimated_seconds=self.estimated_trial_seconds,
            safety_margin_seconds=self.safety_margin_seconds,
        ):
            return None
        timeout = self.budget.effective_timeout(
            phase,
            self.trial_timeout_seconds,
            safety_margin_seconds=self.safety_margin_seconds,
        )
        return timeout if timeout > 0.0 else None

    def _snapshots(self) -> dict[str, JsonValue]:
        return {
            "allocator": self.allocator.snapshot(),
            "scheduler": self.scheduler.snapshot(),
            "budget": self.budget.snapshot(),
        }

    def _restore(self) -> None:
        terminal = self.store.list_trials(self.run_id)
        states = self.store.load_component_states(self.run_id)
        present = _CONTROLLER_COMPONENTS.intersection(states)
        if present and present != _CONTROLLER_COMPONENTS:
            raise ResumeError("controller checkpoint is missing allocator, scheduler, or budget")

        if present:
            self.allocator.restore(_state_object(states["allocator"], "allocator"))
            self.scheduler.restore(_state_object(states["scheduler"], "scheduler"))
            self.budget = _restore_budget(
                self.budget,
                _state_object(states["budget"], "budget"),
                self._budget_clock,
            )
        elif terminal:
            for result in terminal:
                self.allocator.update(result)
                self.scheduler.observe(result)

        for family, optimizer in self.optimizers.items():
            asked_results: list[TrialResult] = []
            for result in terminal:
                if result.family != family or result.status not in _OPTUNA_STATUSES:
                    continue
                candidate = optimizer.candidate_by_id(
                    result.trial_id,
                    result.pipeline_spec,
                    result.fidelity,
                )
                if candidate is not None:
                    asked_results.append(result)
            optimizer.reconcile_results(asked_results)

    def _reconcile_duplicate(self, work: _WorkItem, stored_trial_id: str) -> None:
        stored = self.store.get_trial(stored_trial_id)
        if stored is None:
            raise ResumeError("candidate is already running in another controller")
        if work.candidate is not None:
            replayed = stored.model_copy(
                update={"trial_id": work.candidate.candidate_id},
                deep=True,
            )
            self.optimizers[work.candidate.family].tell(work.candidate, replayed)
            if self.allocator.state_for(work.candidate.family).pending > 0:
                self.allocator.cancel_pending(work.candidate.family)

    def _validate_components(self) -> None:
        optimizer_families = tuple(self.optimizers)
        if not self.allocator.families:
            self.allocator.register(optimizer_families)
        if set(self.allocator.families) != set(optimizer_families):
            raise ValueError("allocator and optimizer families differ")
        for family, optimizer in self.optimizers.items():
            if optimizer.family != family:
                raise ValueError("optimizer mapping key differs from optimizer.family")
        if self.profile.inferred_task is TaskType.AUTO:
            raise ValueError("search controller requires a resolved dataset task")
        if self.validation_plan.dataset_hash is not None and (
            self.validation_plan.dataset_hash != self.profile.dataset_hash
        ):
            raise ValueError("validation plan does not match the dataset profile")

    def _validate_retries(self) -> None:
        trial_ids = [trial.trial_id for trial in self._retryable_trials]
        if len(trial_ids) != len(set(trial_ids)):
            raise ValueError("retryable trials must have unique trial ids")
        for retry in self._retryable_trials:
            if retry.pipeline_spec.family not in self.optimizers:
                raise ValueError("retryable trial references an unknown family")
            if retry.primary_metric is not self.metric:
                raise ValueError("retryable trial primary metric differs from the controller")
            if retry.fidelity != self.scheduler.policy.level(retry.fidelity.level):
                raise ValueError("retryable trial fidelity differs from the scheduler policy")


def _promotion_work(promotion: Promotion) -> _WorkItem:
    digest = sha256_json(
        {
            "kind": "fidelity_promotion",
            "parent_trial_id": promotion.parent_trial_id,
            "pipeline_spec": promotion.pipeline_spec.to_json_value(),
            "fidelity": promotion.to_fidelity.to_json_value(),
        }
    )
    return _WorkItem(
        proposed_trial_id=f"{_PROMOTION_PREFIX}{digest}",
        pipeline_spec=promotion.pipeline_spec,
        fidelity=promotion.to_fidelity,
        candidate=None,
    )


def _phase_for_fidelity(fidelity: FidelitySpec) -> BudgetPhase:
    return BudgetPhase.CONFIRMATION if fidelity.level == 3 else BudgetPhase.EXPLORATION


def _consume_matching_promotion(
    scheduler: FidelityScheduler,
    retry: RetryableTrial,
) -> None:
    pending = scheduler.pending_promotions()
    if not pending:
        raise ResumeError("retryable promoted trial is absent from the scheduler queue")
    expected = pending[0]
    if expected.pipeline_spec != retry.pipeline_spec or expected.to_fidelity != retry.fidelity:
        raise ResumeError("retryable promoted trial differs from the pending scheduler queue")
    if scheduler.pop_next_promotion() != expected:
        raise RuntimeError("fidelity promotion queue changed unexpectedly")


def _state_object(value: JsonValue, component: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ResumeError(f"{component} checkpoint must be a JSON object")
    return cast(dict[str, JsonValue], value)


def _restore_budget(
    current: BudgetManager,
    snapshot: Mapping[str, JsonValue],
    clock: Callable[[], float] | None,
) -> BudgetManager:
    restored = (
        BudgetManager.restore(snapshot)
        if clock is None
        else BudgetManager.restore(snapshot, clock=clock)
    )
    if (
        restored.confirmation_fraction != current.confirmation_fraction
        or restored.finalization_fraction != current.finalization_fraction
    ):
        raise ResumeError("budget reserve fractions differ from the persisted checkpoint")
    consumed = float(cast(float | int, snapshot["consumed_seconds"]))
    if current.total_seconds < restored.total_seconds or current.total_seconds < consumed:
        raise ResumeError("current total budget is below the persisted budget state")
    if clock is None:
        return BudgetManager(
            current.total_seconds,
            consumed_seconds=consumed,
            confirmation_fraction=current.confirmation_fraction,
            finalization_fraction=current.finalization_fraction,
        )
    return BudgetManager(
        current.total_seconds,
        consumed_seconds=consumed,
        confirmation_fraction=current.confirmation_fraction,
        finalization_fraction=current.finalization_fraction,
        clock=clock,
    )


def _safe_failure_type(error: Exception) -> str:
    name = type(error).__name__
    sanitized = "".join(character for character in name if character.isalnum() or character == "_")
    return sanitized[:80] or "PipelineEvaluationError"


def _validate_result_identity(
    result: TrialResult,
    trial_id: str,
    pipeline_spec: PipelineSpec,
    fidelity: FidelitySpec,
    metric: MetricName,
) -> None:
    if result.trial_id != trial_id:
        raise ValueError("evaluator returned a different trial id")
    if result.pipeline_spec != pipeline_spec or result.family != pipeline_spec.family:
        raise ValueError("evaluator returned a different PipelineSpec or family")
    if result.fidelity != fidelity:
        raise ValueError("evaluator returned a different fidelity")
    if result.primary_metric is not metric:
        raise ValueError("evaluator returned a different primary metric")
    if result.status not in _OPTUNA_STATUSES:
        raise ValueError("controller evaluator must return completed, failed, or pruned")


__all__ = [
    "EvaluationOutcomeLike",
    "FamilyOptimizer",
    "SearchController",
    "TrialCompletedCallback",
    "TrialEvaluator",
]

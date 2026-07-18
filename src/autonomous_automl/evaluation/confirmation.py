"""Crash-safe final-fidelity evaluation of a deterministic finalist shortlist."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from autonomous_automl.contracts import (
    DatasetProfile,
    FidelitySpec,
    MetricName,
    PipelineSpec,
    TrialResult,
    TrialStatus,
    ValidationPlan,
)
from autonomous_automl.data import LoadedDataset
from autonomous_automl.evaluation.metrics import resolve_metric
from autonomous_automl.pipelines import pipeline_fingerprint
from autonomous_automl.search.budget import BudgetManager, BudgetPhase
from autonomous_automl.search.fidelity import FidelityPolicy
from autonomous_automl.tracking.store import ExperimentStore
from autonomous_automl.utils.errors import ResumeError
from autonomous_automl.utils.hashing import sha256_json
from autonomous_automl.utils.json import JsonValue

_EVALUATOR_STATUSES = {
    TrialStatus.COMPLETED,
    TrialStatus.FAILED,
    TrialStatus.PRUNED,
}


class EvaluationOutcomeLike(Protocol):
    """Structural evaluator result used without coupling to its concrete class."""

    trial_result: TrialResult


class FinalistEvaluator(Protocol):
    """Pipeline-evaluator surface required by :class:`FinalistConfirmer`."""

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


ComponentStateSupplier = Callable[[], Mapping[str, JsonValue]]


@dataclass(frozen=True, slots=True)
class _Attempt:
    result: TrialResult | None
    executed: bool
    budget_exhausted: bool


class FinalistConfirmer:
    """Evaluate distinct top candidates at full then multi-seed fidelity.

    Every reservation and terminal result is persisted through the normal
    exactly-once registry boundary. Both levels consume only the confirmation
    phase, so the finalization reserve remains unavailable to this service.
    """

    def __init__(
        self,
        *,
        run_id: str,
        evaluator: FinalistEvaluator,
        dataset: LoadedDataset,
        profile: DatasetProfile,
        validation_plan: ValidationPlan,
        store: ExperimentStore,
        budget: BudgetManager,
        metric: MetricName | str,
        lease_epoch: int,
        fidelity_policy: FidelityPolicy | None = None,
        top_n: int = 3,
        uncertainty_weight: float = 1.0,
        trial_timeout_seconds: float | None = None,
        estimated_trial_seconds: float = 0.0,
        safety_margin_seconds: float = 0.0,
        component_states_supplier: ComponentStateSupplier | None = None,
    ) -> None:
        if not run_id.strip():
            raise ValueError("run_id cannot be blank")
        if top_n < 1:
            raise ValueError("top_n must be positive")
        numeric_values = (
            uncertainty_weight,
            estimated_trial_seconds,
            safety_margin_seconds,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("confirmation numeric options must be finite")
        if uncertainty_weight < 0.0:
            raise ValueError("uncertainty_weight must be non-negative")
        if estimated_trial_seconds < 0.0 or safety_margin_seconds < 0.0:
            raise ValueError("trial estimate and safety margin must be non-negative")
        if trial_timeout_seconds is not None and (
            not math.isfinite(trial_timeout_seconds) or trial_timeout_seconds <= 0.0
        ):
            raise ValueError("trial_timeout_seconds must be finite and positive")
        if validation_plan.dataset_hash is not None and (
            validation_plan.dataset_hash != profile.dataset_hash
        ):
            raise ValueError("validation plan does not match the dataset profile")

        policy = fidelity_policy or FidelityPolicy(
            available_folds=validation_plan.n_splits,
            random_seed=validation_plan.random_seed or 0,
        )
        if policy.available_folds != validation_plan.n_splits:
            raise ValueError("fidelity policy fold count differs from the validation plan")

        progress = store.get_run_progress(run_id)
        if progress.lease_epoch != lease_epoch:
            raise ResumeError("confirmer lease epoch differs from the run registry")

        self.run_id = run_id
        self.evaluator = evaluator
        self.dataset = dataset
        self.profile = profile
        self.validation_plan = validation_plan
        self.store = store
        self.budget = budget
        self.metric = resolve_metric(metric, profile.inferred_task, profile).name
        self.lease_epoch = lease_epoch
        self.fidelity_policy = policy
        self.top_n = top_n
        self.uncertainty_weight = uncertainty_weight
        self.trial_timeout_seconds = trial_timeout_seconds
        self.estimated_trial_seconds = estimated_trial_seconds
        self.safety_margin_seconds = safety_margin_seconds
        self.component_states_supplier = component_states_supplier

    def run(self, trials: Iterable[TrialResult]) -> list[TrialResult]:
        """Execute missing full/confirmation work and return only work done now."""

        specifications = self._shortlist(trials)
        persisted = self._persisted_results()
        executed: list[TrialResult] = []

        for specification in specifications:
            fingerprint = pipeline_fingerprint(specification)
            full_key = (fingerprint, 2)
            full_result = persisted.get(full_key)
            if full_result is None:
                attempt = self._attempt(specification, self.fidelity_policy.level(2))
                if attempt.budget_exhausted:
                    break
                full_result = attempt.result
                if full_result is None:
                    continue
                persisted[full_key] = full_result
                if attempt.executed:
                    executed.append(full_result)

            self._validate_persisted_result(
                full_result,
                specification,
                expected_fidelity=self.fidelity_policy.level(2),
            )
            if full_result.status is not TrialStatus.COMPLETED:
                continue

            confirmation_key = (fingerprint, 3)
            confirmation = persisted.get(confirmation_key)
            if confirmation is not None:
                self._validate_persisted_result(
                    confirmation,
                    specification,
                    expected_fidelity=self.fidelity_policy.level(3),
                )
                continue

            attempt = self._attempt(specification, self.fidelity_policy.level(3))
            if attempt.budget_exhausted:
                break
            if attempt.result is None:
                continue
            persisted[confirmation_key] = attempt.result
            if attempt.executed:
                executed.append(attempt.result)

        return executed

    def _shortlist(self, trials: Iterable[TrialResult]) -> tuple[PipelineSpec, ...]:
        representatives: dict[str, TrialResult] = {}
        for result in trials:
            if result.status is not TrialStatus.COMPLETED:
                continue
            if result.primary_metric is not self.metric:
                raise ValueError("completed finalist metric differs from confirmer metric")
            fingerprint = pipeline_fingerprint(result.pipeline_spec)
            current = representatives.get(fingerprint)
            if current is None or self._ranking_key(result) < self._ranking_key(current):
                representatives[fingerprint] = result

        ranked = sorted(representatives.values(), key=self._ranking_key)
        selected = ranked[: self.top_n]
        baselines = [result for result in ranked if _is_baseline(result)]
        if baselines:
            baseline = baselines[0]
            selected_fingerprints = {
                pipeline_fingerprint(result.pipeline_spec) for result in selected
            }
            if pipeline_fingerprint(baseline.pipeline_spec) not in selected_fingerprints:
                selected.append(baseline)
        return tuple(result.pipeline_spec for result in selected)

    def _ranking_key(self, result: TrialResult) -> tuple[float, int, str]:
        if result.mean_score is None or result.std_score is None:
            raise ValueError("completed finalist lacks aggregate scores")
        adjusted = result.mean_score - self.uncertainty_weight * result.std_score
        return (-adjusted, -result.fidelity.level, result.trial_id)

    def _persisted_results(self) -> dict[tuple[str, int], TrialResult]:
        selected: dict[tuple[str, int], TrialResult] = {}
        expected = {
            2: self.fidelity_policy.level(2),
            3: self.fidelity_policy.level(3),
        }
        for result in self.store.list_trials(self.run_id):
            if result.fidelity != expected.get(result.fidelity.level):
                continue
            key = (pipeline_fingerprint(result.pipeline_spec), result.fidelity.level)
            current = selected.get(key)
            if current is None or _persisted_key(result) < _persisted_key(current):
                selected[key] = result
        return selected

    def _attempt(self, specification: PipelineSpec, fidelity: FidelitySpec) -> _Attempt:
        timeout = self._timeout_for_confirmation()
        if timeout is None:
            return _Attempt(None, False, True)

        proposed_id = _trial_id(self.run_id, specification, fidelity)
        reservation = self.store.reserve_trial(
            self.run_id,
            proposed_id,
            specification,
            fidelity,
            self.metric,
        )
        if reservation.lease_epoch != self.lease_epoch:
            raise ResumeError("confirmation reservation was created under a stale lease")
        if not reservation.reserved:
            return _Attempt(self.store.get_trial(reservation.trial_id), False, False)

        consumed_before = self.budget.consumed_seconds
        result = self._evaluate(
            reservation.trial_id,
            specification,
            fidelity,
            timeout,
        )
        consumed_delta = max(0.0, self.budget.consumed_seconds - consumed_before)
        self.store.record_trial_result(
            result,
            attempt_token=reservation.attempt_token,
            lease_epoch=self.lease_epoch,
            component_states=self._component_states(),
            consumed_seconds=consumed_delta,
        )
        return _Attempt(result, True, False)

    def _evaluate(
        self,
        trial_id: str,
        specification: PipelineSpec,
        fidelity: FidelitySpec,
        timeout: float,
    ) -> TrialResult:
        started_at = datetime.now(UTC)
        try:
            outcome = self.evaluator.evaluate(
                trial_id,
                specification,
                fidelity,
                self.dataset,
                self.profile,
                self.validation_plan,
                self.metric,
                timeout_seconds=timeout,
            )
            result = outcome.trial_result
            _validate_result_identity(
                result,
                trial_id,
                specification,
                fidelity,
                self.metric,
            )
            return result
        except Exception as error:
            failure_type = _safe_failure_type(error)
            return TrialResult(
                trial_id=trial_id,
                family=specification.family,
                pipeline_spec=specification,
                fidelity=fidelity,
                status=TrialStatus.FAILED,
                primary_metric=self.metric,
                failure_type=failure_type,
                failure_message=f"finalist evaluation failed ({failure_type})",
                started_at=started_at,
                finished_at=datetime.now(UTC),
            )

    def _timeout_for_confirmation(self) -> float | None:
        if not self.budget.can_start(
            BudgetPhase.CONFIRMATION,
            estimated_seconds=self.estimated_trial_seconds,
            safety_margin_seconds=self.safety_margin_seconds,
        ):
            return None
        timeout = self.budget.effective_timeout(
            BudgetPhase.CONFIRMATION,
            self.trial_timeout_seconds,
            safety_margin_seconds=self.safety_margin_seconds,
        )
        return timeout if timeout > 0.0 else None

    def _component_states(self) -> dict[str, JsonValue]:
        states = (
            {} if self.component_states_supplier is None else dict(self.component_states_supplier())
        )
        states["budget"] = self.budget.snapshot()
        return states

    def _validate_persisted_result(
        self,
        result: TrialResult,
        specification: PipelineSpec,
        *,
        expected_fidelity: FidelitySpec,
    ) -> None:
        if result.pipeline_spec != specification or result.fidelity != expected_fidelity:
            raise ResumeError("persisted finalist result identity is inconsistent")
        if result.primary_metric is not self.metric:
            raise ResumeError("persisted finalist metric differs from confirmer metric")


def _is_baseline(result: TrialResult) -> bool:
    return (
        result.family.casefold() == "baseline"
        or result.pipeline_spec.model_name.casefold() == "dummy"
    )


def _persisted_key(result: TrialResult) -> tuple[int, str, str]:
    return (
        0 if result.status is TrialStatus.COMPLETED else 1,
        result.fidelity.to_json(),
        result.trial_id,
    )


def _trial_id(run_id: str, specification: PipelineSpec, fidelity: FidelitySpec) -> str:
    digest = sha256_json(
        {
            "kind": "finalist_confirmation",
            "run_id": run_id,
            "pipeline_spec": specification.to_json_value(),
            "fidelity": fidelity.to_json_value(),
        }
    )
    return f"finalist-l{fidelity.level}-{digest}"


def _safe_failure_type(error: Exception) -> str:
    name = type(error).__name__
    sanitized = "".join(character for character in name if character.isalnum() or character == "_")
    return sanitized[:80] or "FinalistEvaluationError"


def _validate_result_identity(
    result: TrialResult,
    trial_id: str,
    specification: PipelineSpec,
    fidelity: FidelitySpec,
    metric: MetricName,
) -> None:
    if result.trial_id != trial_id:
        raise ValueError("evaluator returned a different trial id")
    if result.pipeline_spec != specification or result.family != specification.family:
        raise ValueError("evaluator returned a different PipelineSpec or family")
    if result.fidelity != fidelity:
        raise ValueError("evaluator returned a different fidelity")
    if result.primary_metric is not metric:
        raise ValueError("evaluator returned a different primary metric")
    if result.status not in _EVALUATOR_STATUSES:
        raise ValueError("confirmation evaluator returned a non-terminal status")


__all__ = [
    "ComponentStateSupplier",
    "EvaluationOutcomeLike",
    "FinalistConfirmer",
    "FinalistEvaluator",
]

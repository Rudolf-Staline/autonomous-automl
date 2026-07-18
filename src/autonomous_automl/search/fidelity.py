"""Deterministic successive-halving policy and promotion state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from autonomous_automl.contracts import FidelitySpec, PipelineSpec, TrialResult, TrialStatus
from autonomous_automl.utils.errors import ResumeError
from autonomous_automl.utils.json import JsonValue, canonical_json_dumps
from autonomous_automl.utils.seeds import derive_seed

_TERMINAL_STATUSES = {
    TrialStatus.COMPLETED,
    TrialStatus.FAILED,
    TrialStatus.PRUNED,
    TrialStatus.INTERRUPTED,
}


@dataclass(frozen=True, slots=True)
class FidelityPolicy:
    """Four reproducible fidelity levels adapted to available validation folds."""

    available_folds: int
    random_seed: int
    confirmation_seed_count: int = 3
    low_sample_fraction: float = 0.25
    medium_sample_fraction: float = 0.60
    low_max_iterations: int = 50
    medium_max_iterations: int = 150
    full_max_iterations: int = 400

    def __post_init__(self) -> None:
        if self.available_folds < 2:
            raise ValueError("available_folds must be at least two")
        if not 0 <= self.random_seed <= 4_294_967_295:
            raise ValueError("random_seed must fit an unsigned 32-bit integer")
        if self.confirmation_seed_count < 2:
            raise ValueError("confirmation_seed_count must be at least two")
        if not 0.0 < self.low_sample_fraction < self.medium_sample_fraction < 1.0:
            raise ValueError("sample fractions must satisfy 0 < low < medium < 1")
        if (
            min(
                self.low_max_iterations,
                self.medium_max_iterations,
                self.full_max_iterations,
            )
            <= 0
        ):
            raise ValueError("fidelity iteration limits must be positive")

    @property
    def levels(self) -> tuple[FidelitySpec, FidelitySpec, FidelitySpec, FidelitySpec]:
        """Materialize low, medium, full, and confirmation specifications."""

        full_folds = min(5, self.available_folds)
        base_seeds = [derive_seed(self.random_seed, "fidelity", level) for level in range(3)]
        confirmation_seeds = [
            derive_seed(self.random_seed, "confirmation", index)
            for index in range(self.confirmation_seed_count)
        ]
        return (
            FidelitySpec(
                level=0,
                sample_fraction=self.low_sample_fraction,
                n_folds=min(2, self.available_folds),
                max_iterations=self.low_max_iterations,
                seeds=[base_seeds[0]],
            ),
            FidelitySpec(
                level=1,
                sample_fraction=self.medium_sample_fraction,
                n_folds=min(3, self.available_folds),
                max_iterations=self.medium_max_iterations,
                seeds=[base_seeds[1]],
            ),
            FidelitySpec(
                level=2,
                sample_fraction=1.0,
                n_folds=full_folds,
                max_iterations=self.full_max_iterations,
                seeds=[base_seeds[2]],
            ),
            FidelitySpec(
                level=3,
                sample_fraction=1.0,
                n_folds=full_folds,
                max_iterations=self.full_max_iterations,
                seeds=confirmation_seeds,
            ),
        )

    def level(self, level: int) -> FidelitySpec:
        """Return one configured level or reject an out-of-range index."""

        if not 0 <= level <= 3:
            raise ValueError("fidelity level must be between zero and three")
        return self.levels[level]

    def to_json_value(self) -> dict[str, JsonValue]:
        """Return the complete versioned policy configuration."""

        return {
            "schema_version": 1,
            "available_folds": self.available_folds,
            "random_seed": self.random_seed,
            "confirmation_seed_count": self.confirmation_seed_count,
            "low_sample_fraction": self.low_sample_fraction,
            "medium_sample_fraction": self.medium_sample_fraction,
            "low_max_iterations": self.low_max_iterations,
            "medium_max_iterations": self.medium_max_iterations,
            "full_max_iterations": self.full_max_iterations,
        }


@dataclass(frozen=True, slots=True)
class Promotion:
    """Promotion of the exact same PipelineSpec to the next fidelity level."""

    parent_trial_id: str
    pipeline_spec: PipelineSpec
    from_fidelity: FidelitySpec
    to_fidelity: FidelitySpec
    adjusted_score: float

    def __post_init__(self) -> None:
        if not self.parent_trial_id:
            raise ValueError("parent_trial_id must not be empty")
        if self.to_fidelity.level != self.from_fidelity.level + 1:
            raise ValueError("a promotion must advance exactly one fidelity level")
        canonical_json_dumps(self.to_json_value())

    def to_json_value(self) -> dict[str, JsonValue]:
        """Return a deterministic checkpoint representation."""

        return {
            "parent_trial_id": self.parent_trial_id,
            "pipeline_spec": self.pipeline_spec.to_json_value(),
            "from_fidelity": self.from_fidelity.to_json_value(),
            "to_fidelity": self.to_fidelity.to_json_value(),
            "adjusted_score": self.adjusted_score,
        }

    @classmethod
    def from_json_value(cls, value: object) -> Promotion:
        """Validate a promotion restored from scheduler state."""

        if not isinstance(value, dict):
            raise ResumeError("promotion checkpoint must be a JSON object")
        try:
            return cls(
                parent_trial_id=cast(str, value["parent_trial_id"]),
                pipeline_spec=PipelineSpec.model_validate(value["pipeline_spec"]),
                from_fidelity=FidelitySpec.model_validate(value["from_fidelity"]),
                to_fidelity=FidelitySpec.model_validate(value["to_fidelity"]),
                adjusted_score=float(cast(float | int, value["adjusted_score"])),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ResumeError("invalid promotion checkpoint") from error


class FidelityScheduler:
    """Track fidelity rungs and queue diverse top-tier promotions."""

    def __init__(
        self,
        policy: FidelityPolicy,
        *,
        reduction_factor: int = 3,
        uncertainty_weight: float = 1.0,
    ) -> None:
        if reduction_factor < 2:
            raise ValueError("reduction_factor must be at least two")
        if uncertainty_weight < 0.0:
            raise ValueError("uncertainty_weight must be non-negative")
        self.policy = policy
        self.reduction_factor = reduction_factor
        self.uncertainty_weight = uncertainty_weight
        self._results: dict[str, TrialResult] = {}
        self._promoted_ids: set[str] = set()
        self._pending: list[Promotion] = []

    def initial_level(self) -> FidelitySpec:
        """Return the low-fidelity entry level."""

        return self.policy.level(0)

    def next_level(self, current: FidelitySpec) -> FidelitySpec | None:
        """Return the next configured level, if one exists."""

        self._validate_fidelity(current)
        if current.level == 3:
            return None
        return self.policy.level(current.level + 1)

    def observe(self, result: TrialResult) -> tuple[Promotion, ...]:
        """Record one terminal result and enqueue newly earned promotions."""

        self._validate_result(result)
        existing = self._results.get(result.trial_id)
        if existing is not None:
            if existing != result:
                raise ResumeError("trial id was observed with conflicting results")
            return ()
        self._results[result.trial_id] = result
        if result.status is not TrialStatus.COMPLETED or result.fidelity.level == 3:
            return ()

        new_promotions = self._refresh_rung(result.fidelity.level)
        self._pending.extend(new_promotions)
        return tuple(new_promotions)

    def should_promote(
        self,
        candidate: TrialResult,
        peers: Sequence[TrialResult],
    ) -> bool:
        """Return whether a completed candidate belongs to its diverse top tier."""

        if candidate.status is not TrialStatus.COMPLETED or candidate.fidelity.level == 3:
            return False
        self._validate_result(candidate)
        comparable = {
            result.trial_id: result
            for result in [*peers, candidate]
            if result.status is TrialStatus.COMPLETED and result.fidelity == candidate.fidelity
        }
        if len(comparable) < self.reduction_factor:
            return False
        slots = len(comparable) // self.reduction_factor
        selected = self._select_diverse_top(list(comparable.values()), slots, {})
        return any(result.trial_id == candidate.trial_id for result in selected)

    def pending_promotions(self) -> tuple[Promotion, ...]:
        """Return the promotion queue without consuming it."""

        return tuple(self._pending)

    def pop_next_promotion(self) -> Promotion | None:
        """Pop the oldest pending promotion."""

        return None if not self._pending else self._pending.pop(0)

    def snapshot(self) -> dict[str, JsonValue]:
        """Return a deterministic, exact JSON checkpoint."""

        value: dict[str, JsonValue] = {
            "schema_version": 1,
            "policy": self.policy.to_json_value(),
            "reduction_factor": self.reduction_factor,
            "uncertainty_weight": self.uncertainty_weight,
            "results": [
                result.to_json_value()
                for result in sorted(self._results.values(), key=lambda item: item.trial_id)
            ],
            "promoted_trial_ids": cast(list[JsonValue], sorted(self._promoted_ids)),
            "pending_promotions": [promotion.to_json_value() for promotion in self._pending],
        }
        canonical_json_dumps(value)
        return value

    def restore(self, snapshot: dict[str, JsonValue]) -> None:
        """Restore a compatible checkpoint without recomputing promotions."""

        try:
            if snapshot.get("schema_version") != 1:
                raise ResumeError("unsupported fidelity scheduler checkpoint version")
            if snapshot.get("policy") != self.policy.to_json_value():
                raise ResumeError("fidelity policy differs from the persisted checkpoint")
            if snapshot.get("reduction_factor") != self.reduction_factor:
                raise ResumeError("reduction factor differs from the persisted checkpoint")
            if snapshot.get("uncertainty_weight") != self.uncertainty_weight:
                raise ResumeError("uncertainty weight differs from the persisted checkpoint")

            raw_results = cast(list[object], snapshot["results"])
            results = [TrialResult.model_validate(value) for value in raw_results]
            restored_results = {result.trial_id: result for result in results}
            if len(restored_results) != len(results):
                raise ResumeError("fidelity checkpoint contains duplicate trial ids")
            for result in results:
                self._validate_result(result)

            promoted_ids = set(cast(list[str], snapshot["promoted_trial_ids"]))
            if not promoted_ids.issubset(restored_results):
                raise ResumeError("promoted trial is missing from fidelity results")
            pending = [
                Promotion.from_json_value(value)
                for value in cast(list[object], snapshot["pending_promotions"])
            ]
            if any(promotion.parent_trial_id not in promoted_ids for promotion in pending):
                raise ResumeError("pending promotion is not marked as promoted")
        except KeyError as error:
            raise ResumeError("incomplete fidelity scheduler checkpoint") from error

        self._results = restored_results
        self._promoted_ids = promoted_ids
        self._pending = pending
        if self.snapshot() != snapshot:
            raise ResumeError("fidelity scheduler checkpoint is not canonical")

    def _refresh_rung(self, level: int) -> list[Promotion]:
        completed = [
            result
            for result in self._results.values()
            if result.status is TrialStatus.COMPLETED and result.fidelity.level == level
        ]
        allowed = len(completed) // self.reduction_factor
        already = [result for result in completed if result.trial_id in self._promoted_ids]
        available_slots = allowed - len(already)
        if available_slots <= 0:
            return []

        group_counts: dict[tuple[str, str], int] = {}
        for result in already:
            key = (result.family, result.pipeline_spec.model_name)
            group_counts[key] = group_counts.get(key, 0) + 1
        eligible = [result for result in completed if result.trial_id not in self._promoted_ids]
        selected = self._select_diverse_top(eligible, available_slots, group_counts)
        next_fidelity = self.policy.level(level + 1)
        promotions = [
            Promotion(
                parent_trial_id=result.trial_id,
                pipeline_spec=result.pipeline_spec,
                from_fidelity=result.fidelity,
                to_fidelity=next_fidelity,
                adjusted_score=self._adjusted_score(result),
            )
            for result in selected
        ]
        self._promoted_ids.update(promotion.parent_trial_id for promotion in promotions)
        return promotions

    def _select_diverse_top(
        self,
        candidates: list[TrialResult],
        slots: int,
        group_counts: dict[tuple[str, str], int],
    ) -> list[TrialResult]:
        remaining = list(candidates)
        selected: list[TrialResult] = []
        counts = dict(group_counts)
        while remaining and len(selected) < slots:
            choice = min(
                remaining,
                key=lambda result: (
                    counts.get((result.family, result.pipeline_spec.model_name), 0),
                    -self._adjusted_score(result),
                    result.trial_id,
                ),
            )
            remaining.remove(choice)
            selected.append(choice)
            key = (choice.family, choice.pipeline_spec.model_name)
            counts[key] = counts.get(key, 0) + 1
        return selected

    def _adjusted_score(self, result: TrialResult) -> float:
        if result.mean_score is None or result.std_score is None:
            raise ValueError("completed trial requires mean_score and std_score")
        return result.mean_score - self.uncertainty_weight * result.std_score

    def _validate_fidelity(self, fidelity: FidelitySpec) -> None:
        if fidelity != self.policy.level(fidelity.level):
            raise ValueError("result fidelity does not match the scheduler policy")

    def _validate_result(self, result: TrialResult) -> None:
        if result.status not in _TERMINAL_STATUSES:
            raise ValueError("scheduler can observe terminal TrialResult values only")
        self._validate_fidelity(result.fidelity)


__all__ = ["FidelityPolicy", "FidelityScheduler", "Promotion"]

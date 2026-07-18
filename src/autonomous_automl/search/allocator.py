"""Deterministic cost-aware UCB allocation across compatible families."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import cast

from autonomous_automl.contracts import TrialResult, TrialStatus
from autonomous_automl.utils.errors import ResumeError
from autonomous_automl.utils.json import JsonValue, canonical_json_dumps


@dataclass(slots=True)
class FamilyState:
    """Online statistics and temporary availability for one pipeline family."""

    attempts: int = 0
    successes: int = 0
    failures: int = 0
    pending: int = 0
    consecutive_failures: int = 0
    best_score: float | None = None
    mean_score: float | None = None
    score_m2: float = 0.0
    mean_cost_seconds: float = 0.0
    recent_improvement: float = 0.0
    last_selected_step: int | None = None
    suspended_until_step: int | None = None
    models_seen: set[str] = field(default_factory=set)

    @property
    def score_variance(self) -> float:
        """Return the sample variance of successful rewards."""

        return self.score_m2 / (self.successes - 1) if self.successes > 1 else 0.0

    @property
    def exploration_count(self) -> int:
        """Count completed attempts and candidates currently in flight."""

        return self.attempts + self.pending

    def to_json_value(self) -> dict[str, JsonValue]:
        """Return a finite JSON representation suitable for checkpoints."""

        return {
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "pending": self.pending,
            "consecutive_failures": self.consecutive_failures,
            "best_score": self.best_score,
            "mean_score": self.mean_score,
            "score_m2": self.score_m2,
            "mean_cost_seconds": self.mean_cost_seconds,
            "recent_improvement": self.recent_improvement,
            "last_selected_step": self.last_selected_step,
            "suspended_until_step": self.suspended_until_step,
            "models_seen": cast(list[JsonValue], sorted(self.models_seen)),
        }


class FamilyAllocator:
    """Choose compatible families using deterministic cost-aware UCB priorities."""

    def __init__(
        self,
        families: Sequence[str] | None = None,
        *,
        minimum_exploration_trials: int = 1,
        failure_threshold: int = 3,
        cooldown_selections: int = 2,
        performance_weight: float = 1.0,
        improvement_weight: float = 0.25,
        cost_weight: float = 0.20,
        uncertainty_weight: float = 0.35,
        diversity_weight: float = 0.15,
        diversity_window: int = 5,
    ) -> None:
        if minimum_exploration_trials < 1:
            raise ValueError("minimum_exploration_trials must be positive")
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be positive")
        if cooldown_selections < 1:
            raise ValueError("cooldown_selections must be positive")
        if diversity_window < 1:
            raise ValueError("diversity_window must be positive")
        weights = (
            performance_weight,
            improvement_weight,
            cost_weight,
            uncertainty_weight,
            diversity_weight,
        )
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise ValueError("allocator weights must be finite and non-negative")

        self.minimum_exploration_trials = minimum_exploration_trials
        self.failure_threshold = failure_threshold
        self.cooldown_selections = cooldown_selections
        self.performance_weight = performance_weight
        self.improvement_weight = improvement_weight
        self.cost_weight = cost_weight
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        self.diversity_window = diversity_window
        self._order: list[str] = []
        self._states: dict[str, FamilyState] = {}
        self._selection_step = 0
        self._recent_selections: list[str] = []
        if families is not None:
            self.register(families)

    @property
    def families(self) -> tuple[str, ...]:
        """Return compatible families in deterministic registration order."""

        return tuple(self._order)

    @property
    def selection_step(self) -> int:
        """Return the number of allocation decisions made so far."""

        return self._selection_step

    def register(self, families: Sequence[str]) -> None:
        """Register compatible families without resetting existing statistics."""

        if isinstance(families, str):
            raise TypeError("families must be a sequence of names, not a string")
        names = list(families)
        if not names:
            raise ValueError("at least one compatible family is required")
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("family names must be non-empty strings")
        if len(names) != len(set(names)):
            raise ValueError("family names must be unique")
        for name in names:
            if name not in self._states:
                self._order.append(name)
                self._states[name] = FamilyState()

    def state_for(self, family: str) -> FamilyState:
        """Return a detached copy of one family's runtime state."""

        state = self._state(family)
        return replace(state, models_seen=set(state.models_seen))

    def select_next(self, available_families: Sequence[str] | None = None) -> str:
        """Select the next family, honoring exploration and temporary suspension."""

        candidates = self._available_names(available_families)
        next_step = self._selection_step + 1
        active = self._reactivated(candidates, next_step)
        if not active:
            suspension_steps = [
                self._states[name].suspended_until_step
                for name in candidates
                if self._states[name].suspended_until_step is not None
            ]
            if not suspension_steps:
                raise RuntimeError("no family is available for allocation")
            next_step = min(cast(list[int], suspension_steps))
            active = self._reactivated(candidates, next_step)

        unexplored = [
            name
            for name in active
            if self._states[name].exploration_count < self.minimum_exploration_trials
        ]
        if unexplored:
            selected = min(
                unexplored,
                key=lambda name: (self._states[name].exploration_count, self._index(name)),
            )
        else:
            priorities = self.priorities(active)
            selected = min(active, key=lambda name: (-priorities[name], self._index(name)))

        self._selection_step = next_step
        state = self._states[selected]
        state.pending += 1
        state.last_selected_step = next_step
        self._recent_selections.append(selected)
        self._recent_selections = self._recent_selections[-self.diversity_window :]
        return selected

    def cancel_pending(self, family: str) -> None:
        """Release a selection whose candidate could not be constructed."""

        state = self._state(family)
        if state.pending <= 0:
            raise RuntimeError("family has no pending selection to cancel")
        state.pending -= 1

    def priorities(self, families: Sequence[str] | None = None) -> dict[str, float]:
        """Return current UCB priorities for diagnostics and deterministic tests."""

        names = self._available_names(families)
        means = {
            name: self._states[name].mean_score
            for name in names
            if self._states[name].mean_score is not None
        }
        improvements = {name: self._states[name].recent_improvement for name in names}
        costs = {name: self._states[name].mean_cost_seconds for name in names}
        normalized_means = _normalize_optional(means, names)
        normalized_improvements = _normalize_non_negative(improvements)
        normalized_costs = _normalize_non_negative(costs)
        total_observations = sum(self._states[name].exploration_count for name in names)
        recent_counts = {name: self._recent_selections.count(name) for name in names}

        scores: dict[str, float] = {}
        for name in names:
            state = self._states[name]
            uncertainty = math.sqrt(
                math.log(total_observations + 2.0) / max(1, state.exploration_count)
            )
            diversity = 1.0 / (1.0 + recent_counts[name])
            scores[name] = (
                self.performance_weight * normalized_means[name]
                + self.improvement_weight * normalized_improvements[name]
                - self.cost_weight * normalized_costs[name]
                + self.uncertainty_weight * uncertainty
                + self.diversity_weight * diversity
            )
        return scores

    def update(self, result: TrialResult) -> None:
        """Update online rewards, costs, failures, and model diversity."""

        state = self._state(result.family)
        if result.status not in {
            TrialStatus.COMPLETED,
            TrialStatus.FAILED,
            TrialStatus.PRUNED,
            TrialStatus.INTERRUPTED,
        }:
            raise ValueError("allocator can update terminal TrialResult values only")
        if state.pending > 0:
            state.pending -= 1
        state.attempts += 1
        cost = result.fit_seconds + result.predict_seconds
        state.mean_cost_seconds += (cost - state.mean_cost_seconds) / state.attempts
        state.models_seen.add(result.pipeline_spec.model_name)

        if result.status is TrialStatus.COMPLETED:
            if result.mean_score is None:
                raise ValueError("completed TrialResult requires mean_score")
            previous_best = state.best_score
            state.successes += 1
            if state.mean_score is None:
                state.mean_score = result.mean_score
            else:
                delta = result.mean_score - state.mean_score
                state.mean_score += delta / state.successes
                state.score_m2 += delta * (result.mean_score - state.mean_score)
            improvement = (
                0.0 if previous_best is None else max(0.0, result.mean_score - previous_best)
            )
            state.recent_improvement = 0.5 * state.recent_improvement + 0.5 * improvement
            state.best_score = (
                result.mean_score
                if previous_best is None
                else max(previous_best, result.mean_score)
            )
            state.consecutive_failures = 0
            return

        if result.status is TrialStatus.FAILED:
            state.failures += 1
            state.consecutive_failures += 1
            if state.consecutive_failures >= self.failure_threshold:
                state.suspended_until_step = self._selection_step + self.cooldown_selections + 1
                state.consecutive_failures = 0
            return

        state.consecutive_failures = 0

    def snapshot(self) -> dict[str, JsonValue]:
        """Return a deterministic versioned JSON checkpoint."""

        value: dict[str, JsonValue] = {
            "schema_version": 1,
            "configuration": self._configuration(),
            "families": cast(list[JsonValue], list(self._order)),
            "selection_step": self._selection_step,
            "recent_selections": cast(list[JsonValue], list(self._recent_selections)),
            "states": {name: self._states[name].to_json_value() for name in self._order},
        }
        canonical_json_dumps(value)
        return value

    def restore(self, snapshot: dict[str, JsonValue]) -> None:
        """Strictly restore a canonical checkpoint and reject incompatible state."""

        if set(snapshot) != {
            "schema_version",
            "configuration",
            "families",
            "selection_step",
            "recent_selections",
            "states",
        }:
            raise ResumeError("allocator checkpoint fields are invalid")
        if snapshot["schema_version"] != 1:
            raise ResumeError("unsupported allocator checkpoint version")
        if snapshot["configuration"] != self._configuration():
            raise ResumeError("allocator configuration differs from checkpoint")
        try:
            families = _string_list(snapshot["families"], "families")
            if not self._order:
                self.register(families)
            elif families != self._order:
                raise ResumeError("registered families differ from checkpoint")
            selection_step = _non_negative_int(snapshot["selection_step"], "selection_step")
            recent = _string_list(snapshot["recent_selections"], "recent_selections")
            raw_snapshot_states = snapshot["states"]
            if not isinstance(raw_snapshot_states, dict):
                raise ResumeError("allocator checkpoint states must be an object")
            raw_states = cast(dict[str, object], raw_snapshot_states)
            if set(raw_states) != set(families):
                raise ResumeError("allocator checkpoint family states are incomplete")
            restored = {name: _restore_family_state(raw_states[name]) for name in families}
            if any(name not in restored for name in recent):
                raise ResumeError("allocator recent selection references an unknown family")
            if len(recent) > self.diversity_window:
                raise ResumeError("allocator recent selection window is too large")
        except (TypeError, ValueError) as error:
            if isinstance(error, ResumeError):
                raise
            raise ResumeError("allocator checkpoint values are invalid") from error

        self._states = restored
        self._selection_step = selection_step
        self._recent_selections = list(recent)
        if self.snapshot() != snapshot:
            raise ResumeError("allocator checkpoint is not canonical")

    def _configuration(self) -> dict[str, JsonValue]:
        return {
            "minimum_exploration_trials": self.minimum_exploration_trials,
            "failure_threshold": self.failure_threshold,
            "cooldown_selections": self.cooldown_selections,
            "performance_weight": self.performance_weight,
            "improvement_weight": self.improvement_weight,
            "cost_weight": self.cost_weight,
            "uncertainty_weight": self.uncertainty_weight,
            "diversity_weight": self.diversity_weight,
            "diversity_window": self.diversity_window,
        }

    def _available_names(self, requested: Sequence[str] | None) -> list[str]:
        if not self._order:
            raise RuntimeError("no compatible family has been registered")
        if requested is None:
            return list(self._order)
        if isinstance(requested, str):
            raise TypeError("available_families must be a sequence, not a string")
        names = list(requested)
        if not names:
            raise ValueError("available_families must not be empty")
        if len(names) != len(set(names)):
            raise ValueError("available_families must be unique")
        unknown = set(names).difference(self._states)
        if unknown:
            raise ValueError(f"unregistered families requested: {sorted(unknown)}")
        return sorted(names, key=self._index)

    def _reactivated(self, names: list[str], step: int) -> list[str]:
        active: list[str] = []
        for name in names:
            state = self._states[name]
            if state.suspended_until_step is None or step >= state.suspended_until_step:
                state.suspended_until_step = None
                active.append(name)
        return active

    def _state(self, family: str) -> FamilyState:
        try:
            return self._states[family]
        except KeyError as error:
            raise ValueError(f"unregistered family: {family}") from error

    def _index(self, family: str) -> int:
        return self._order.index(family)


def _normalize_optional(values: dict[str, float | None], names: list[str]) -> dict[str, float]:
    present = [value for value in values.values() if value is not None]
    if not present:
        return dict.fromkeys(names, 0.0)
    minimum = min(present)
    maximum = max(present)
    if maximum == minimum:
        return {name: 0.5 if values.get(name) is not None else 0.0 for name in names}
    return {
        name: (
            0.0
            if values.get(name) is None
            else (cast(float, values[name]) - minimum) / (maximum - minimum)
        )
        for name in names
    }


def _normalize_non_negative(values: dict[str, float]) -> dict[str, float]:
    maximum = max(values.values(), default=0.0)
    return {name: (value / maximum if maximum > 0.0 else 0.0) for name, value in values.items()}


def _restore_family_state(value: object) -> FamilyState:
    expected = {
        "attempts",
        "successes",
        "failures",
        "pending",
        "consecutive_failures",
        "best_score",
        "mean_score",
        "score_m2",
        "mean_cost_seconds",
        "recent_improvement",
        "last_selected_step",
        "suspended_until_step",
        "models_seen",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ResumeError("allocator family state fields are invalid")
    try:
        state = FamilyState(
            attempts=_non_negative_int(value["attempts"], "attempts"),
            successes=_non_negative_int(value["successes"], "successes"),
            failures=_non_negative_int(value["failures"], "failures"),
            pending=_non_negative_int(value["pending"], "pending"),
            consecutive_failures=_non_negative_int(
                value["consecutive_failures"], "consecutive_failures"
            ),
            best_score=_optional_float(value["best_score"]),
            mean_score=_optional_float(value["mean_score"]),
            score_m2=float(value["score_m2"]),
            mean_cost_seconds=float(value["mean_cost_seconds"]),
            recent_improvement=float(value["recent_improvement"]),
            last_selected_step=_optional_int(value["last_selected_step"]),
            suspended_until_step=_optional_int(value["suspended_until_step"]),
            models_seen=set(_string_list(value["models_seen"], "models_seen")),
        )
    except (TypeError, ValueError) as error:
        raise ResumeError("allocator family state values are invalid") from error
    numeric = (
        state.attempts,
        state.successes,
        state.failures,
        state.pending,
        state.consecutive_failures,
    )
    floats = (state.score_m2, state.mean_cost_seconds, state.recent_improvement)
    if any(item < 0 for item in numeric) or any(
        not math.isfinite(item) or item < 0 for item in floats
    ):
        raise ResumeError("allocator family state contains invalid statistics")
    if state.successes + state.failures > state.attempts:
        raise ResumeError("allocator family outcome counts exceed attempts")
    if state.successes == 0 and (state.mean_score is not None or state.best_score is not None):
        raise ResumeError("allocator family reward statistics are inconsistent")
    if state.successes > 0 and (state.mean_score is None or state.best_score is None):
        raise ResumeError("allocator family reward statistics are inconsistent")
    return state


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("allocator reward values must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("non-finite allocator value")
    return converted


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("allocator step values must be integers")
    if value < 0:
        raise ValueError("allocator step values must be non-negative")
    return value


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResumeError(f"allocator {field_name} must be a non-negative integer")
    return value


def _string_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ResumeError(f"allocator {field_name} must contain non-empty strings")
    if len(value) != len(set(value)) and field_name in {"families", "models_seen"}:
        raise ResumeError(f"allocator {field_name} must not contain duplicates")
    return list(value)


__all__ = ["FamilyAllocator", "FamilyState"]

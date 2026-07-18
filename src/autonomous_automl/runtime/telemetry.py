"""Observation-only search and finalization telemetry."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import cast

from autonomous_automl.contracts import RuntimeTelemetry, SearchStopReason, TrialResult, TrialStatus
from autonomous_automl.utils.json import JsonValue

RUNTIME_STATE_KEY = "trust_runtime"
RUNTIME_TELEMETRY_PATH = "runtime_telemetry.json"


class RuntimeTelemetryRecorder:
    """Measure active phases without changing launch-budget decisions."""

    def __init__(
        self,
        configured_search_budget_seconds: float,
        *,
        previous: RuntimeTelemetry | None = None,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if configured_search_budget_seconds <= 0:
            raise ValueError("configured search budget must be positive")
        self.configured_search_budget_seconds = float(configured_search_budget_seconds)
        self._clock = clock
        self._now = now or (lambda: datetime.now(UTC))
        self._base_search_elapsed = (
            0.0
            if previous is None or previous.search_elapsed_seconds is None
            else previous.search_elapsed_seconds
        )
        self.search_started_at = (
            previous.search_started_at
            if previous is not None and previous.search_started_at is not None
            else self._now()
        )
        self._search_segment_started = clock()
        self._search_elapsed: float | None = None
        self._search_finished_at: datetime | None = None
        self._finalization_started: float | None = None
        self._stop_reason: SearchStopReason | None = None
        self._remaining_at_stop: float | None = None

    @property
    def search_stopped(self) -> bool:
        """Whether the current active search segment has been closed."""

        return self._search_elapsed is not None

    @property
    def finalization_started(self) -> bool:
        """Whether post-selection timing has begun."""

        return self._finalization_started is not None

    @classmethod
    def from_scheduler_state(
        cls,
        configured_search_budget_seconds: float,
        scheduler_state: Mapping[str, JsonValue],
        *,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] | None = None,
    ) -> RuntimeTelemetryRecorder:
        """Restore accumulated active time from an optional interrupted checkpoint."""

        raw = scheduler_state.get(RUNTIME_STATE_KEY)
        previous = None
        if isinstance(raw, dict):
            previous = RuntimeTelemetry.model_validate(raw)
        return cls(
            configured_search_budget_seconds,
            previous=previous,
            clock=clock,
            now=now,
        )

    def stop_search(
        self,
        reason: SearchStopReason,
        *,
        budget_remaining_seconds: float,
    ) -> None:
        """Close the active search segment exactly once."""

        if self._search_elapsed is not None:
            raise RuntimeError("search telemetry has already stopped")
        if budget_remaining_seconds < 0:
            raise ValueError("remaining budget must be non-negative")
        self._search_elapsed = self._base_search_elapsed + max(
            0.0,
            self._clock() - self._search_segment_started,
        )
        self._search_finished_at = self._now()
        self._stop_reason = reason
        self._remaining_at_stop = min(
            self.configured_search_budget_seconds,
            float(budget_remaining_seconds),
        )

    def start_finalization(self) -> None:
        """Start post-selection work without touching the search budget."""

        if self._search_elapsed is None:
            raise RuntimeError("search telemetry must stop before finalization")
        if self._finalization_started is not None:
            raise RuntimeError("finalization telemetry has already started")
        self._finalization_started = self._clock()

    def mark_fatal_error(self) -> None:
        """Record that a later essential operation made the run fail."""

        if self._search_elapsed is None:
            raise RuntimeError("search telemetry must stop before marking a fatal error")
        self._stop_reason = SearchStopReason.FATAL_ERROR

    def snapshot(
        self,
        trials: Sequence[TrialResult],
        *,
        include_finalization: bool,
    ) -> RuntimeTelemetry:
        """Build a validated snapshot from measured clocks and persisted trials."""

        if (
            self._search_elapsed is None
            or self._search_finished_at is None
            or self._stop_reason is None
            or self._remaining_at_stop is None
        ):
            raise RuntimeError("search telemetry is incomplete")
        finalization = 0.0
        if include_finalization:
            if self._finalization_started is None:
                raise RuntimeError("finalization telemetry has not started")
            finalization = max(0.0, self._clock() - self._finalization_started)
        return RuntimeTelemetry(
            configured_search_budget_seconds=self.configured_search_budget_seconds,
            search_started_at=self.search_started_at,
            search_finished_at=self._search_finished_at,
            search_elapsed_seconds=self._search_elapsed,
            finalization_elapsed_seconds=finalization,
            total_runtime_seconds=self._search_elapsed + finalization,
            budget_remaining_at_search_stop_seconds=self._remaining_at_stop,
            budget_overshoot_seconds=max(
                0.0,
                self._search_elapsed - self.configured_search_budget_seconds,
            ),
            trials_started=len(trials),
            trials_completed=sum(trial.status is TrialStatus.COMPLETED for trial in trials),
            trials_failed=sum(trial.status is TrialStatus.FAILED for trial in trials),
            stop_reason=self._stop_reason,
        )


def runtime_from_scheduler_state(
    scheduler_state: Mapping[str, JsonValue],
) -> RuntimeTelemetry | None:
    """Load optional telemetry without making old manifests unreadable."""

    value = scheduler_state.get(RUNTIME_STATE_KEY)
    if not isinstance(value, dict):
        return None
    return RuntimeTelemetry.model_validate(cast(dict[str, JsonValue], value))


__all__ = [
    "RUNTIME_STATE_KEY",
    "RUNTIME_TELEMETRY_PATH",
    "RuntimeTelemetryRecorder",
    "runtime_from_scheduler_state",
]

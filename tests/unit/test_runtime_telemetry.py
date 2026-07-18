"""Runtime telemetry observes phases without changing budget decisions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from autonomous_automl.contracts import RuntimeTelemetry, SearchStopReason
from autonomous_automl.runtime import RUNTIME_STATE_KEY, RuntimeTelemetryRecorder


class FakeTime:
    def __init__(self) -> None:
        self.monotonic = 100.0
        self.wall = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)

    def clock(self) -> float:
        return self.monotonic

    def now(self) -> datetime:
        return self.wall

    def advance(self, seconds: float) -> None:
        self.monotonic += seconds
        self.wall += timedelta(seconds=seconds)


def _recorder(fake: FakeTime, budget: float = 10) -> RuntimeTelemetryRecorder:
    return RuntimeTelemetryRecorder(budget, clock=fake.clock, now=fake.now)


def test_budget_exhaustion_and_finalization_are_persisted_separately() -> None:
    fake = FakeTime()
    recorder = _recorder(fake)
    fake.advance(12)
    recorder.stop_search(SearchStopReason.BUDGET_EXHAUSTED, budget_remaining_seconds=0)
    recorder.start_finalization()
    fake.advance(3)

    telemetry = recorder.snapshot([], include_finalization=True)

    assert telemetry.search_elapsed_seconds == pytest.approx(12)
    assert telemetry.finalization_elapsed_seconds == pytest.approx(3)
    assert telemetry.total_runtime_seconds == pytest.approx(15)
    assert telemetry.budget_overshoot_seconds == pytest.approx(2)
    assert telemetry.stop_reason is SearchStopReason.BUDGET_EXHAUSTED


@pytest.mark.parametrize(
    "reason",
    [
        SearchStopReason.SEARCH_SPACE_EXHAUSTED,
        SearchStopReason.USER_INTERRUPTED,
        SearchStopReason.CONTROLLER_STOPPED,
        SearchStopReason.FATAL_ERROR,
    ],
)
def test_every_declared_stop_reason_round_trips(reason: SearchStopReason) -> None:
    fake = FakeTime()
    recorder = _recorder(fake)
    fake.advance(1)
    recorder.stop_search(reason, budget_remaining_seconds=9)

    restored = RuntimeTelemetry.from_json(
        recorder.snapshot([], include_finalization=False).to_json()
    )

    assert restored.stop_reason is reason


def test_resume_accumulates_active_search_time_without_charging_downtime() -> None:
    first_time = FakeTime()
    first = _recorder(first_time, budget=20)
    first_time.advance(4)
    first.stop_search(SearchStopReason.USER_INTERRUPTED, budget_remaining_seconds=16)
    partial = first.snapshot([], include_finalization=False)

    resumed_time = FakeTime()
    resumed_time.monotonic = 10_000
    resumed_time.wall += timedelta(days=2)
    resumed = RuntimeTelemetryRecorder.from_scheduler_state(
        20,
        {RUNTIME_STATE_KEY: partial.to_json_value()},
        clock=resumed_time.clock,
        now=resumed_time.now,
    )
    resumed_time.advance(3)
    resumed.stop_search(SearchStopReason.SEARCH_SPACE_EXHAUSTED, budget_remaining_seconds=13)
    resumed.start_finalization()
    resumed_time.advance(2)

    final = resumed.snapshot([], include_finalization=True)

    assert final.search_elapsed_seconds == pytest.approx(7)
    assert final.finalization_elapsed_seconds == pytest.approx(2)
    assert final.total_runtime_seconds == pytest.approx(9)
    assert final.search_started_at == partial.search_started_at


def test_finalization_after_budget_is_not_reported_as_search_overshoot() -> None:
    fake = FakeTime()
    recorder = _recorder(fake, budget=8)
    fake.advance(7.5)
    recorder.stop_search(SearchStopReason.CONTROLLER_STOPPED, budget_remaining_seconds=0.5)
    recorder.start_finalization()
    fake.advance(5)

    telemetry = recorder.snapshot([], include_finalization=True)

    assert telemetry.total_runtime_seconds == pytest.approx(12.5)
    assert telemetry.budget_overshoot_seconds == 0


def test_runtime_contract_rejects_invented_totals_and_trial_counts() -> None:
    with pytest.raises(ValidationError, match="total runtime"):
        RuntimeTelemetry(
            configured_search_budget_seconds=10,
            search_elapsed_seconds=4,
            finalization_elapsed_seconds=2,
            total_runtime_seconds=99,
            budget_overshoot_seconds=0,
            trials_started=1,
            trials_completed=1,
            trials_failed=0,
            stop_reason="controller_stopped",
        )

    with pytest.raises(ValidationError, match="cannot exceed"):
        RuntimeTelemetry(
            configured_search_budget_seconds=10,
            trials_started=1,
            trials_completed=1,
            trials_failed=1,
            stop_reason="controller_stopped",
        )

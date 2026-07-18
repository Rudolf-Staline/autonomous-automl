"""Protected monotonic budget tests."""

from __future__ import annotations

import pytest

from autonomous_automl.search.budget import BudgetManager, BudgetPhase


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_exploration_and_confirmation_cannot_consume_final_reserve() -> None:
    clock = FakeClock()
    budget = BudgetManager(100, clock=clock)

    assert budget.available_for(BudgetPhase.EXPLORATION) == pytest.approx(70)
    clock.advance(69)
    assert not budget.can_start(BudgetPhase.EXPLORATION, estimated_seconds=2)
    assert budget.available_for(BudgetPhase.CONFIRMATION) == pytest.approx(11)
    assert budget.remaining_seconds == pytest.approx(31)
    clock.advance(11)
    assert budget.available_for(BudgetPhase.CONFIRMATION) == 0
    assert budget.available_for(BudgetPhase.FINALIZATION) == pytest.approx(20)


def test_timeout_is_capped_by_phase_and_safety_margin() -> None:
    clock = FakeClock()
    budget = BudgetManager(20, clock=clock)
    clock.advance(10)

    assert budget.effective_timeout(BudgetPhase.CONFIRMATION, 30) == pytest.approx(6)
    assert budget.effective_timeout(
        BudgetPhase.CONFIRMATION,
        None,
        safety_margin_seconds=1,
    ) == pytest.approx(5)


def test_snapshot_restore_does_not_charge_process_downtime() -> None:
    first_clock = FakeClock()
    budget = BudgetManager(100, consumed_seconds=5, clock=first_clock)
    first_clock.advance(12)
    snapshot = budget.snapshot()
    restored_clock = FakeClock()
    restored_clock.now = 10_000

    restored = BudgetManager.restore(snapshot, clock=restored_clock)

    assert restored.consumed_seconds == pytest.approx(17)
    restored_clock.advance(3)
    assert restored.consumed_seconds == pytest.approx(20)
    assert restored.finalization_reserve_seconds == pytest.approx(20)

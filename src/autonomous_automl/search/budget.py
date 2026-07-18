"""Monotonic wall-clock budget with protected confirmation and final reserves."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import cast

from autonomous_automl.utils.json import JsonValue


class BudgetPhase(StrEnum):
    EXPLORATION = "exploration"
    CONFIRMATION = "confirmation"
    FINALIZATION = "finalization"


class BudgetManager:
    """Protect later phases while excluding process downtime after restoration."""

    def __init__(
        self,
        total_seconds: float,
        *,
        consumed_seconds: float = 0.0,
        confirmation_fraction: float = 0.10,
        finalization_fraction: float = 0.20,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        values = (total_seconds, consumed_seconds, confirmation_fraction, finalization_fraction)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("budget values must be finite")
        if total_seconds <= 0 or consumed_seconds < 0:
            raise ValueError("total budget must be positive and consumed time non-negative")
        if not 0 <= confirmation_fraction < 1 or not 0 < finalization_fraction < 1:
            raise ValueError("budget reserve fractions are invalid")
        if confirmation_fraction + finalization_fraction >= 1:
            raise ValueError("confirmation and finalization cannot reserve the whole budget")
        self.total_seconds = float(total_seconds)
        self._base_consumed_seconds = float(consumed_seconds)
        self.confirmation_fraction = float(confirmation_fraction)
        self.finalization_fraction = float(finalization_fraction)
        self._clock = clock
        self._started_at = clock()

    @property
    def finalization_reserve_seconds(self) -> float:
        return self.total_seconds * self.finalization_fraction

    @property
    def confirmation_reserve_seconds(self) -> float:
        return self.total_seconds * self.confirmation_fraction

    @property
    def consumed_seconds(self) -> float:
        return self._base_consumed_seconds + max(0.0, self._clock() - self._started_at)

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.total_seconds - self.consumed_seconds)

    def available_for(self, phase: BudgetPhase) -> float:
        """Return time usable now without consuming reserves of later phases."""
        if phase is BudgetPhase.EXPLORATION:
            phase_limit = self.total_seconds * (
                1.0 - self.confirmation_fraction - self.finalization_fraction
            )
        elif phase is BudgetPhase.CONFIRMATION:
            phase_limit = self.total_seconds * (1.0 - self.finalization_fraction)
        else:
            phase_limit = self.total_seconds
        return max(0.0, phase_limit - self.consumed_seconds)

    def can_start(
        self,
        phase: BudgetPhase,
        *,
        estimated_seconds: float = 0.0,
        safety_margin_seconds: float = 0.0,
    ) -> bool:
        if estimated_seconds < 0 or safety_margin_seconds < 0:
            raise ValueError("estimated time and safety margin must be non-negative")
        return self.available_for(phase) >= estimated_seconds + safety_margin_seconds

    def effective_timeout(
        self,
        phase: BudgetPhase,
        requested_seconds: float | None,
        *,
        safety_margin_seconds: float = 0.0,
    ) -> float:
        available = max(0.0, self.available_for(phase) - safety_margin_seconds)
        if requested_seconds is None:
            return available
        if not math.isfinite(requested_seconds) or requested_seconds <= 0:
            raise ValueError("requested timeout must be finite and positive")
        return min(float(requested_seconds), available)

    def snapshot(self) -> dict[str, JsonValue]:
        return {
            "schema_version": 1,
            "total_seconds": self.total_seconds,
            "consumed_seconds": min(self.consumed_seconds, self.total_seconds),
            "confirmation_fraction": self.confirmation_fraction,
            "finalization_fraction": self.finalization_fraction,
        }

    @classmethod
    def restore(
        cls,
        snapshot: Mapping[str, JsonValue],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> BudgetManager:
        if snapshot.get("schema_version") != 1:
            raise ValueError("unsupported budget snapshot version")
        try:
            return cls(
                total_seconds=float(cast(float | int, snapshot["total_seconds"])),
                consumed_seconds=float(cast(float | int, snapshot["consumed_seconds"])),
                confirmation_fraction=float(cast(float | int, snapshot["confirmation_fraction"])),
                finalization_fraction=float(cast(float | int, snapshot["finalization_fraction"])),
                clock=clock,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid budget snapshot") from error


__all__ = ["BudgetManager", "BudgetPhase"]

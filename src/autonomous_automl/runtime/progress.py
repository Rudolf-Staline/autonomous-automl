"""Safe progress events shared by the Python API and the CLI."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from autonomous_automl.utils.json import JsonValue, canonical_json_dumps


class RunStage(StrEnum):
    LOADING = "loading"
    PROFILING = "profiling"
    LEAKAGE = "leakage"
    VALIDATION = "validation"
    SEARCH = "search"
    CONFIRMATION = "confirmation"
    FINAL_TRAINING = "final_training"
    ARTIFACTS = "artifacts"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RunProgressEvent:
    """One sanitized progress snapshot without user data values."""

    stage: RunStage
    message: str
    elapsed_seconds: float
    budget_seconds: float
    remaining_seconds: float
    completed_trials: int = 0
    failed_trials: int = 0
    interrupted_trials: int = 0
    best_internal_score: float | None = None

    def to_json_value(self) -> dict[str, JsonValue]:
        return {
            "stage": self.stage.value,
            "message": self.message,
            "elapsed_seconds": self.elapsed_seconds,
            "budget_seconds": self.budget_seconds,
            "remaining_seconds": self.remaining_seconds,
            "completed_trials": self.completed_trials,
            "failed_trials": self.failed_trials,
            "interrupted_trials": self.interrupted_trials,
            "best_internal_score": self.best_internal_score,
        }


ProgressCallback = Callable[[RunProgressEvent], None]


class ProgressReporter:
    """Build progress snapshots and optionally persist JSON Lines."""

    def __init__(
        self,
        budget_seconds: float,
        *,
        callback: ProgressCallback | None = None,
        log_path: str | Path | None = None,
        started_clock: float | None = None,
    ) -> None:
        if budget_seconds <= 0:
            raise ValueError("progress budget must be positive")
        self.budget_seconds = float(budget_seconds)
        self.callback = callback
        self.log_path = None if log_path is None else Path(log_path)
        self.started_clock = time.monotonic() if started_clock is None else started_clock

    def emit(
        self,
        stage: RunStage,
        message: str,
        *,
        remaining_seconds: float | None = None,
        completed_trials: int = 0,
        failed_trials: int = 0,
        interrupted_trials: int = 0,
        best_internal_score: float | None = None,
    ) -> RunProgressEvent:
        elapsed = max(0.0, time.monotonic() - self.started_clock)
        remaining = (
            max(0.0, self.budget_seconds - elapsed)
            if remaining_seconds is None
            else max(0.0, remaining_seconds)
        )
        event = RunProgressEvent(
            stage=stage,
            message=message,
            elapsed_seconds=elapsed,
            budget_seconds=self.budget_seconds,
            remaining_seconds=remaining,
            completed_trials=completed_trials,
            failed_trials=failed_trials,
            interrupted_trials=interrupted_trials,
            best_internal_score=best_internal_score,
        )
        if self.log_path is not None:
            self._append(event)
        if self.callback is not None:
            self.callback(event)
        return event

    def _append(self, event: RunProgressEvent) -> None:
        assert self.log_path is not None
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_json_dumps(event.to_json_value()) + "\n"
        with self.log_path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())


__all__ = [
    "ProgressCallback",
    "ProgressReporter",
    "RunProgressEvent",
    "RunStage",
]

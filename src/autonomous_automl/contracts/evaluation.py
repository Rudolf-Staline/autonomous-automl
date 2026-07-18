"""Contracts produced by fold and trial evaluation."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from autonomous_automl.contracts.base import ContractModel
from autonomous_automl.contracts.enums import MetricName, TrialStatus
from autonomous_automl.contracts.pipeline import FidelitySpec, PipelineSpec

_TERMINAL_STATUSES = {
    TrialStatus.COMPLETED,
    TrialStatus.FAILED,
    TrialStatus.PRUNED,
    TrialStatus.INTERRUPTED,
}


class FoldResult(ContractModel):
    """Metrics and timings for one validation fold."""

    fold_index: int = Field(ge=0)
    seed: int | None = Field(default=None, ge=0, le=4_294_967_295)
    score: float
    user_metric_value: float
    fit_seconds: float = Field(ge=0)
    predict_seconds: float = Field(ge=0)
    n_train_rows: int = Field(gt=0)
    n_validation_rows: int = Field(gt=0)


class TrialResult(ContractModel):
    """Versioned outcome of evaluating one PipelineSpec at one fidelity."""

    schema_version: Literal[1] = 1
    trial_id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    pipeline_spec: PipelineSpec
    fidelity: FidelitySpec
    status: TrialStatus
    primary_metric: MetricName
    mean_score: float | None = None
    std_score: float | None = Field(default=None, ge=0)
    fold_scores: list[float] = Field(default_factory=list)
    fold_results: list[FoldResult] = Field(default_factory=list)
    fit_seconds: float = Field(default=0.0, ge=0)
    predict_seconds: float = Field(default=0.0, ge=0)
    peak_memory_mb: float | None = Field(default=None, ge=0)
    failure_type: str | None = None
    failure_message: str | None = None
    started_at: datetime
    finished_at: datetime | None = None

    @model_validator(mode="after")
    def _validate_outcome(self) -> TrialResult:
        if self.family != self.pipeline_spec.family:
            raise ValueError("trial family must match PipelineSpec.family")
        if self.status in _TERMINAL_STATUSES and self.finished_at is None:
            raise ValueError("terminal trial statuses require finished_at")
        if self.status not in _TERMINAL_STATUSES and self.finished_at is not None:
            raise ValueError("non-terminal trial statuses cannot have finished_at")
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished_at cannot precede started_at")

        if self.status == TrialStatus.COMPLETED:
            if self.mean_score is None or self.std_score is None or not self.fold_scores:
                raise ValueError("completed trials require aggregate and fold scores")
            if self.failure_type is not None or self.failure_message is not None:
                raise ValueError("completed trials cannot contain failure details")
            if self.fold_results and len(self.fold_results) != len(self.fold_scores):
                raise ValueError("fold_results and fold_scores must have the same length")
        elif self.status == TrialStatus.FAILED:
            if not self.failure_type or not self.failure_message:
                raise ValueError("failed trials require failure_type and failure_message")
            if self.mean_score is not None:
                raise ValueError("failed trials cannot contain an aggregate score")
        elif self.failure_type is not None or self.failure_message is not None:
            if self.status not in {TrialStatus.PRUNED, TrialStatus.INTERRUPTED}:
                raise ValueError("failure details require a failed, pruned, or interrupted status")
        return self

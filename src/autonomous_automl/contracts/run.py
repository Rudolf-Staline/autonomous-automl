"""Run manifest, notebook validation, and public result contracts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from autonomous_automl.contracts.base import ContractModel
from autonomous_automl.contracts.config import AutoMLConfig
from autonomous_automl.contracts.dataset import DatasetBundle, DatasetProfile
from autonomous_automl.contracts.enums import (
    MetricName,
    NotebookValidationStatus,
    RunStatus,
)
from autonomous_automl.contracts.leakage import LeakageReport
from autonomous_automl.contracts.pipeline import PipelineSpec
from autonomous_automl.contracts.validation import ValidationAudit, ValidationPlan
from autonomous_automl.utils.json import JsonValue


class NotebookValidation(ContractModel):
    """Persisted proof that the generated notebook reproduced predictions."""

    status: NotebookValidationStatus = NotebookValidationStatus.PENDING
    executed_at: datetime | None = None
    prediction_match: bool | None = None
    absolute_tolerance: float = Field(default=1e-10, ge=0)
    relative_tolerance: float = Field(default=1e-8, ge=0)
    failure_message: str | None = None

    @model_validator(mode="after")
    def _validate_status(self) -> NotebookValidation:
        if self.status == NotebookValidationStatus.PENDING:
            if self.executed_at is not None or self.prediction_match is not None:
                raise ValueError("pending notebook validation cannot have execution results")
        elif self.executed_at is None:
            raise ValueError("finished notebook validation requires executed_at")

        if self.status == NotebookValidationStatus.PASSED:
            if self.prediction_match is not True:
                raise ValueError("passed notebook validation requires matching predictions")
            if self.failure_message is not None:
                raise ValueError("passed notebook validation cannot have a failure message")
        if self.status == NotebookValidationStatus.FAILED and not self.failure_message:
            raise ValueError("failed notebook validation requires a failure message")
        return self


class RunManifest(ContractModel):
    """Authoritative inventory of a run and all reconstructible decisions."""

    manifest_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    package_version: str = Field(min_length=1)
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    configuration: AutoMLConfig
    source_hashes: dict[str, str]
    dataset: DatasetBundle
    dataset_profile: DatasetProfile
    leakage_report: LeakageReport
    validation_plan: ValidationPlan
    validation_audit: ValidationAudit | None = None
    dependency_versions: dict[str, str]
    random_seed: int = Field(ge=0, le=4_294_967_295)
    best_pipeline: PipelineSpec | None = None
    finalist_trial_ids: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    scheduler_state: dict[str, JsonValue] = Field(default_factory=dict)
    notebook_validation: NotebookValidation = Field(default_factory=NotebookValidation)

    @model_validator(mode="after")
    def _validate_run_state(self) -> RunManifest:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        if self.completed_at is not None and self.completed_at < self.created_at:
            raise ValueError("completed_at cannot precede created_at")
        if self.source_hashes != self.dataset.source_hashes:
            raise ValueError("manifest source_hashes must match DatasetBundle")
        if self.random_seed != self.configuration.random_seed:
            raise ValueError("manifest random_seed must match configuration")
        if self.status == RunStatus.COMPLETED:
            if self.completed_at is None:
                raise ValueError("completed runs require completed_at")
            if self.best_pipeline is None:
                raise ValueError("completed runs require best_pipeline")
        elif self.completed_at is not None:
            raise ValueError("only completed runs may contain completed_at")
        return self


class RunResult(ContractModel):
    """Small public API result containing paths to durable run artifacts."""

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    status: RunStatus
    output_dir: Path
    manifest_path: Path
    best_pipeline_path: Path
    best_pipeline_spec_path: Path
    leaderboard_path: Path
    notebook_path: Path | None = None
    report_path: Path
    primary_metric: MetricName
    best_score: float
    oof_predictions_path: Path | None = None
    predictions_path: Path | None = None

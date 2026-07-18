"""Validated user configuration contract."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from autonomous_automl.contracts.base import ContractModel
from autonomous_automl.contracts.enums import MetricName, OptimizationProfile, TaskType

_CLASSIFICATION_METRICS = {
    MetricName.ROC_AUC,
    MetricName.AVERAGE_PRECISION,
    MetricName.LOG_LOSS,
    MetricName.ACCURACY,
    MetricName.BALANCED_ACCURACY,
    MetricName.F1,
    MetricName.MACRO_F1,
}
_REGRESSION_METRICS = {MetricName.RMSE, MetricName.MAE, MetricName.R2}


class AutoMLConfig(ContractModel):
    """Complete, serializable configuration for one AutoML run."""

    schema_version: Literal[1] = 1
    target: str = Field(min_length=1)
    task: TaskType = TaskType.AUTO
    metric: MetricName = MetricName.AUTO
    budget_seconds: int = Field(
        gt=0,
        description=(
            "Search-launch budget. New trials stop launching when exhausted; finalization "
            "and an already-running native operation may extend total wall-clock runtime."
        ),
    )
    random_seed: int = Field(default=42, ge=0, le=4_294_967_295)
    n_jobs: int = Field(default=1, ge=1)
    output_dir: Path = Path("runs/automl")
    test_path: Path | None = None
    group_column: str | None = None
    time_column: str | None = None
    id_column: str | None = None
    predefined_fold_column: str | None = None
    optimization_profile: OptimizationProfile = OptimizationProfile.BALANCED
    enable_gpu: bool = False
    memory_limit_mb: int | None = Field(default=None, ge=128)
    trial_timeout_seconds: int | None = Field(default=None, gt=0)
    compute_trust_gap: bool = False
    trust_gap_timeout_seconds: int = Field(default=30, gt=0)

    @field_validator(
        "target",
        "group_column",
        "time_column",
        "id_column",
        "predefined_fold_column",
    )
    @classmethod
    def _reject_blank_column_names(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("column names must not be blank")
        return value

    @model_validator(mode="after")
    def _validate_cross_field_constraints(self) -> AutoMLConfig:
        role_columns = [
            self.group_column,
            self.time_column,
            self.id_column,
            self.predefined_fold_column,
        ]
        configured_roles = [column for column in role_columns if column is not None]
        if self.target in configured_roles:
            raise ValueError("target cannot also be a group, time, id, or fold column")
        if len(configured_roles) != len(set(configured_roles)):
            raise ValueError("group, time, id, and fold columns must be distinct")

        if self.task == TaskType.REGRESSION and self.metric not in {
            MetricName.AUTO,
            *_REGRESSION_METRICS,
        }:
            raise ValueError(f"metric {self.metric.value!r} is incompatible with regression")
        if self.task in {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        } and self.metric not in {MetricName.AUTO, *_CLASSIFICATION_METRICS}:
            raise ValueError(f"metric {self.metric.value!r} is incompatible with classification")
        if self.task == TaskType.MULTICLASS_CLASSIFICATION and self.metric in {
            MetricName.ROC_AUC,
            MetricName.AVERAGE_PRECISION,
            MetricName.F1,
        }:
            raise ValueError(f"metric {self.metric.value!r} requires binary classification in V1")
        return self

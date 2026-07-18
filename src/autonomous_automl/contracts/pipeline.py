"""Pipeline and fidelity descriptions, independent from executable estimators."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from autonomous_automl.contracts.base import ContractModel
from autonomous_automl.contracts.enums import TaskType
from autonomous_automl.utils.json import JsonValue


class PipelineSpec(ContractModel):
    """The sole source of truth for reconstructing a candidate pipeline."""

    spec_version: Literal[1] = 1
    family: str = Field(min_length=1)
    task: TaskType
    numeric_imputer: str = Field(min_length=1)
    numeric_scaler: str = Field(min_length=1)
    categorical_imputer: str = Field(min_length=1)
    categorical_encoder: str = Field(min_length=1)
    datetime_transformer: str = Field(min_length=1)
    feature_selector: str | None = None
    model_name: str = Field(min_length=1)
    model_params: dict[str, JsonValue] = Field(default_factory=dict)
    excluded_columns: list[str] = Field(default_factory=list)
    random_seed: int = Field(ge=0, le=4_294_967_295)

    @field_validator("task")
    @classmethod
    def _require_resolved_task(cls, value: TaskType) -> TaskType:
        if value == TaskType.AUTO:
            raise ValueError("PipelineSpec requires a resolved task")
        return value

    @model_validator(mode="after")
    def _validate_columns(self) -> PipelineSpec:
        if len(self.excluded_columns) != len(set(self.excluded_columns)):
            raise ValueError("excluded_columns must not contain duplicates")
        return self


class FidelitySpec(ContractModel):
    """Reproducible data, fold, iteration, and seed fidelity level."""

    spec_version: Literal[1] = 1
    level: int = Field(ge=0, le=3)
    sample_fraction: float = Field(gt=0.0, le=1.0)
    n_folds: int = Field(ge=2)
    max_iterations: int | None = Field(default=None, gt=0)
    seeds: list[int] = Field(min_length=1)

    @field_validator("seeds")
    @classmethod
    def _validate_seeds(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("fidelity seeds must be unique")
        if any(seed < 0 or seed > 4_294_967_295 for seed in value):
            raise ValueError("fidelity seeds must fit an unsigned 32-bit integer")
        return value

    @model_validator(mode="after")
    def _validate_level(self) -> FidelitySpec:
        if self.level >= 2 and self.sample_fraction != 1.0:
            raise ValueError("full and confirmation fidelity require all rows")
        if self.level == 3 and len(self.seeds) < 2:
            raise ValueError("confirmation fidelity requires multiple seeds")
        return self

"""Serializable dataset descriptions and profiles."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from autonomous_automl.contracts.base import ContractModel
from autonomous_automl.contracts.enums import TaskType
from autonomous_automl.utils.json import JsonValue

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class DatasetBundle(ContractModel):
    """Persistent description of loaded data, never the raw DataFrames.

    Runtime DataFrames live in ``data.LoadedDataset``. Keeping them out of this
    contract makes manifests safe, compact, and reconstructible from explicit
    sources and hashes.
    """

    schema_version: Literal[1] = 1
    train_paths: list[Path] = Field(min_length=1)
    test_path: Path | None = None
    target: str = Field(min_length=1)
    feature_columns: list[str] = Field(min_length=1)
    excluded_columns: list[str] = Field(default_factory=list)
    n_rows: int = Field(gt=0)
    n_test_rows: int | None = Field(default=None, ge=0)
    row_id_column: str | None = None
    source_hashes: dict[str, str] = Field(min_length=1)

    @field_validator("source_hashes")
    @classmethod
    def _validate_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        invalid = [name for name, digest in value.items() if not _SHA256_PATTERN.fullmatch(digest)]
        if invalid:
            raise ValueError(f"source hashes must be lowercase SHA-256 digests: {invalid}")
        return value

    @model_validator(mode="after")
    def _validate_schema(self) -> DatasetBundle:
        if len(self.train_paths) != len(set(self.train_paths)):
            raise ValueError("train_paths must not contain duplicates")
        if len(self.feature_columns) != len(set(self.feature_columns)):
            raise ValueError("feature_columns must not contain duplicates")
        if self.target in self.feature_columns:
            raise ValueError("target must never be present in feature_columns")
        if set(self.feature_columns).intersection(self.excluded_columns):
            raise ValueError("a column cannot be both a feature and excluded")
        if self.test_path is None and self.n_test_rows is not None:
            raise ValueError("n_test_rows requires a test_path")
        if self.test_path is not None and self.n_test_rows is None:
            raise ValueError("test_path requires n_test_rows")
        return self


class DatasetProfile(ContractModel):
    """Data-derived metadata used to plan validation and compatible pipelines."""

    schema_version: Literal[1] = 1
    dataset_hash: str
    n_rows: int = Field(gt=0)
    n_features: int = Field(ge=1)
    inferred_task: TaskType
    inferred_types: dict[str, str]
    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    boolean_columns: list[str] = Field(default_factory=list)
    datetime_columns: list[str] = Field(default_factory=list)
    text_columns: list[str] = Field(default_factory=list)
    constant_columns: list[str] = Field(default_factory=list)
    near_constant_columns: list[str] = Field(default_factory=list)
    id_candidates: list[str] = Field(default_factory=list)
    time_candidates: list[str] = Field(default_factory=list)
    group_candidates: list[str] = Field(default_factory=list)
    missing_ratios: dict[str, float] = Field(default_factory=dict)
    cardinalities: dict[str, int] = Field(default_factory=dict)
    unique_ratios: dict[str, float] = Field(default_factory=dict)
    numeric_skewness: dict[str, float] = Field(default_factory=dict)
    infinite_counts: dict[str, int] = Field(default_factory=dict)
    target_profile: dict[str, JsonValue] = Field(default_factory=dict)
    landmarks: dict[str, float] = Field(default_factory=dict)
    duplicate_row_count: int = Field(default=0, ge=0)
    potential_cross_fold_duplicate_count: int = Field(default=0, ge=0)
    estimated_memory_mb: float = Field(ge=0)

    @field_validator("dataset_hash")
    @classmethod
    def _validate_dataset_hash(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("dataset_hash must be a lowercase SHA-256 digest")
        return value

    @field_validator("missing_ratios", "unique_ratios")
    @classmethod
    def _validate_ratios(cls, value: dict[str, float]) -> dict[str, float]:
        invalid = [name for name, ratio in value.items() if not 0.0 <= ratio <= 1.0]
        if invalid:
            raise ValueError(f"ratios must be between zero and one: {invalid}")
        return value

    @field_validator("cardinalities", "infinite_counts")
    @classmethod
    def _validate_counts(cls, value: dict[str, int]) -> dict[str, int]:
        invalid = [name for name, count in value.items() if count < 0]
        if invalid:
            raise ValueError(f"counts must be non-negative: {invalid}")
        return value

    @model_validator(mode="after")
    def _validate_column_roles(self) -> DatasetProfile:
        primary_groups = [
            self.numeric_columns,
            self.categorical_columns,
            self.boolean_columns,
            self.datetime_columns,
            self.text_columns,
        ]
        flattened = [column for group in primary_groups for column in group]
        if len(flattened) != len(set(flattened)):
            raise ValueError("primary inferred column roles must be disjoint")
        if len(flattened) != self.n_features:
            raise ValueError("primary inferred column roles must cover every feature")
        if set(flattened) != set(self.inferred_types):
            raise ValueError("inferred_types must contain exactly the feature columns")
        return self

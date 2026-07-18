"""Serializable validation strategy and materialized fold metadata."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from autonomous_automl.contracts.base import ContractModel


class FoldAssignment(ContractModel):
    """One fold expressed as stable row positions, not index labels."""

    fold_index: int = Field(ge=0)
    train_positions: list[int] = Field(min_length=1)
    validation_positions: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_positions(self) -> FoldAssignment:
        train = set(self.train_positions)
        validation = set(self.validation_positions)
        if len(train) != len(self.train_positions):
            raise ValueError("train_positions must not contain duplicates")
        if len(validation) != len(self.validation_positions):
            raise ValueError("validation_positions must not contain duplicates")
        if min(train | validation) < 0:
            raise ValueError("fold positions must be non-negative")
        if train.intersection(validation):
            raise ValueError("training and validation positions must be disjoint")
        return self


class ValidationPlan(ContractModel):
    """Explainable and reproducible validation strategy."""

    schema_version: Literal[1] = 1
    splitter_name: str = Field(min_length=1)
    n_splits: int = Field(ge=2)
    shuffle: bool
    random_seed: int | None = Field(default=None, ge=0, le=4_294_967_295)
    group_column: str | None = None
    time_column: str | None = None
    predefined_fold_column: str | None = None
    rationale: list[str] = Field(min_length=1)
    dataset_hash: str | None = None
    folds: list[FoldAssignment] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_strategy(self) -> ValidationPlan:
        normalized = self.splitter_name.lower()
        if "group" in normalized and self.group_column is None:
            raise ValueError("a group splitter requires group_column")
        if "timeseries" in normalized or "time_series" in normalized:
            if self.time_column is None:
                raise ValueError("a time-series splitter requires time_column")
            if self.shuffle:
                raise ValueError("time-series validation cannot shuffle rows")
        if "predefined" in normalized and self.predefined_fold_column is None:
            raise ValueError("a predefined splitter requires predefined_fold_column")
        if self.shuffle and self.random_seed is None:
            raise ValueError("shuffled validation requires random_seed")
        if not self.shuffle and self.random_seed is not None:
            raise ValueError("random_seed must be null when shuffle is false")
        if self.folds and len(self.folds) != self.n_splits:
            raise ValueError("materialized folds must match n_splits")
        if self.folds:
            fold_indices = [fold.fold_index for fold in self.folds]
            if fold_indices != list(range(self.n_splits)):
                raise ValueError("materialized folds must be ordered from zero to n_splits - 1")
        return self


class ValidationAudit(ContractModel):
    """Persisted fold-safety checks, with duplicates treated as warnings."""

    schema_version: Literal[1] = 1
    valid: bool
    n_rows: int = Field(gt=0)
    validation_coverage_ratio: float = Field(ge=0.0, le=1.0)
    train_validation_overlap_counts: list[int]
    group_overlap_counts: list[int]
    temporal_violation_counts: list[int]
    duplicate_overlap_counts: list[int]
    out_of_bounds_count: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_audit(self) -> ValidationAudit:
        count_lists = (
            self.train_validation_overlap_counts,
            self.group_overlap_counts,
            self.temporal_violation_counts,
            self.duplicate_overlap_counts,
        )
        if not count_lists[0]:
            raise ValueError("a validation audit requires at least one fold")
        if len({len(counts) for counts in count_lists}) != 1:
            raise ValueError("all per-fold audit counts must have the same length")
        if any(count < 0 for counts in count_lists for count in counts):
            raise ValueError("audit counts must be non-negative")
        hard_failures = self.out_of_bounds_count + sum(
            self.train_validation_overlap_counts
            + self.group_overlap_counts
            + self.temporal_violation_counts
        )
        if self.valid != (hard_failures == 0):
            raise ValueError("valid must reflect all hard fold-safety checks")
        return self

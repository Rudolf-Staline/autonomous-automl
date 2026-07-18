"""Post-materialization validation safety checks."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

from autonomous_automl.contracts import ValidationAudit, ValidationPlan
from autonomous_automl.data import LoadedDataset


def audit_validation(dataset: LoadedDataset, plan: ValidationPlan) -> ValidationAudit:
    """Audit fold bounds, disjointness, group/time safety, and duplicate crossings."""
    n_rows = len(dataset.X)
    hasher = cast(Callable[..., pd.Series], hash_pandas_object)
    row_hashes = np.asarray(hasher(dataset.X, index=False), dtype=np.uint64)
    validation_positions: set[int] = set()
    overlaps: list[int] = []
    group_overlaps: list[int] = []
    temporal_violations: list[int] = []
    duplicate_overlaps: list[int] = []
    out_of_bounds = 0

    for fold in plan.folds:
        train_all = fold.train_positions
        validation_all = fold.validation_positions
        out_of_bounds += sum(position < 0 or position >= n_rows for position in train_all)
        out_of_bounds += sum(position < 0 or position >= n_rows for position in validation_all)
        train = [position for position in train_all if 0 <= position < n_rows]
        validation = [position for position in validation_all if 0 <= position < n_rows]
        validation_positions.update(validation)

        overlaps.append(len(set(train).intersection(validation)))
        group_overlaps.append(_group_overlap_count(dataset, plan, train, validation))
        temporal_violations.append(_temporal_violation_count(dataset, plan, train, validation))
        duplicate_overlaps.append(_duplicate_overlap_count(row_hashes, train, validation))

    hard_failures = out_of_bounds + sum(overlaps) + sum(group_overlaps) + sum(temporal_violations)
    warnings: list[str] = []
    duplicate_fold_count = sum(count > 0 for count in duplicate_overlaps)
    if duplicate_fold_count:
        warnings.append(
            f"exact feature duplicates cross train/validation in {duplicate_fold_count} folds"
        )
    coverage = len(validation_positions) / n_rows
    if coverage < 1.0:
        warnings.append(
            "validation does not cover every row; this is expected for time series or "
            "training-only predefined rows"
        )
    return ValidationAudit(
        valid=hard_failures == 0,
        n_rows=n_rows,
        validation_coverage_ratio=coverage,
        train_validation_overlap_counts=overlaps,
        group_overlap_counts=group_overlaps,
        temporal_violation_counts=temporal_violations,
        duplicate_overlap_counts=duplicate_overlaps,
        out_of_bounds_count=out_of_bounds,
        warnings=warnings,
    )


def _group_overlap_count(
    dataset: LoadedDataset,
    plan: ValidationPlan,
    train: list[int],
    validation: list[int],
) -> int:
    if plan.group_column is None:
        return 0
    if dataset.groups is None:
        return max(1, len(validation))
    train_groups = {_normalized_group(dataset.groups.iloc[position]) for position in train}
    validation_groups = {
        _normalized_group(dataset.groups.iloc[position]) for position in validation
    }
    return len(train_groups.intersection(validation_groups))


def _normalized_group(value: object) -> tuple[str, str]:
    if bool(pd.isna(value)):
        return ("missing", "")
    return (type(value).__name__, str(value))


def _temporal_violation_count(
    dataset: LoadedDataset,
    plan: ValidationPlan,
    train: list[int],
    validation: list[int],
) -> int:
    if plan.time_column is None:
        return 0
    if dataset.time_values is None or not train or not validation:
        return 1
    numeric_time = _comparable_time_values(dataset.time_values)
    train_max = np.max(numeric_time[np.asarray(train, dtype=int)])
    validation_min = np.min(numeric_time[np.asarray(validation, dtype=int)])
    return int(train_max >= validation_min)


def _comparable_time_values(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series.dtype):
        numeric = cast(pd.Series, pd.to_numeric(series, errors="raise"))
        return np.asarray(numeric, dtype=np.float64)
    parsed = cast(
        pd.Series,
        pd.to_datetime(series, errors="raise", utc=True, format="mixed"),
    )
    return np.asarray(parsed.astype("int64"), dtype=np.int64)


def _duplicate_overlap_count(
    row_hashes: np.ndarray,
    train: list[int],
    validation: list[int],
) -> int:
    train_hashes = set(row_hashes[np.asarray(train, dtype=int)].tolist())
    validation_hashes = row_hashes[np.asarray(validation, dtype=int)].tolist()
    return sum(value in train_hashes for value in validation_hashes)


__all__ = ["audit_validation"]

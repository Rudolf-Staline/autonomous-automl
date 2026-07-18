"""Leakage-safe and deterministic row sampling for fidelity levels."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from autonomous_automl.contracts import (
    FidelitySpec,
    FoldAssignment,
    TaskType,
    ValidationPlan,
)
from autonomous_automl.data import LoadedDataset

PositionArray = NDArray[np.int64]
_CLASSIFICATION_TASKS = {
    TaskType.BINARY_CLASSIFICATION,
    TaskType.MULTICLASS_CLASSIFICATION,
}
_MAX_SEED = 4_294_967_295


class FidelitySampler:
    """Select training rows without crossing validation-strategy boundaries."""

    @staticmethod
    def sample_training_positions(
        dataset: LoadedDataset,
        validation_plan: ValidationPlan,
        fold: FoldAssignment,
        fidelity: FidelitySpec,
        seed: int,
        task: TaskType,
    ) -> PositionArray:
        """Return a deterministic subset of one fold's training row positions.

        Sampling is driven exclusively by training-side runtime fields. In
        particular, ``test_X`` and its metadata are deliberately never read.
        """
        resolved_task = TaskType(task)
        _validate_inputs(dataset, validation_plan, fold, seed, resolved_task)
        positions = np.asarray(fold.train_positions, dtype=np.int64)

        if fidelity.sample_fraction == 1.0:
            return positions.copy()
        if validation_plan.time_column is not None:
            return _sample_time_prefix(
                dataset,
                fold,
                positions,
                fraction=fidelity.sample_fraction,
                task=resolved_task,
            )
        if validation_plan.group_column is not None:
            return _sample_whole_groups(
                dataset,
                positions,
                fraction=fidelity.sample_fraction,
                seed=seed,
                task=resolved_task,
            )
        if resolved_task in _CLASSIFICATION_TASKS:
            return _sample_stratified(
                positions,
                dataset.y,
                fraction=fidelity.sample_fraction,
                seed=seed,
            )
        return _sample_random(
            positions,
            fraction=fidelity.sample_fraction,
            seed=seed,
        )


def _validate_inputs(
    dataset: LoadedDataset,
    validation_plan: ValidationPlan,
    fold: FoldAssignment,
    seed: int,
    task: TaskType,
) -> None:
    if task is TaskType.AUTO:
        raise ValueError("fidelity sampling requires a resolved task")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= _MAX_SEED:
        raise ValueError("sampling seed must fit an unsigned 32-bit integer")
    if not validation_plan.folds:
        raise ValueError("validation plan must contain materialized folds")
    if fold.fold_index >= len(validation_plan.folds):
        raise ValueError("fold is absent from the validation plan")
    if validation_plan.folds[fold.fold_index] != fold:
        raise ValueError("fold differs from its materialized validation-plan assignment")

    n_rows = len(dataset.X)
    if dataset.bundle.n_rows != n_rows:
        raise ValueError("runtime dataset row count differs from DatasetBundle")
    all_positions = np.asarray(
        [*fold.train_positions, *fold.validation_positions],
        dtype=np.int64,
    )
    if bool((all_positions >= n_rows).any()):
        raise ValueError("fold contains an out-of-bounds row position")

    roles = (
        validation_plan.group_column is not None,
        validation_plan.time_column is not None,
        validation_plan.predefined_fold_column is not None,
    )
    if sum(roles) > 1:
        raise ValueError("validation plan has ambiguous sampling roles")
    if validation_plan.group_column is not None and (
        dataset.groups is None or len(dataset.groups) != n_rows
    ):
        raise ValueError("group validation requires one group value per training row")
    if validation_plan.time_column is not None and (
        dataset.time_values is None or len(dataset.time_values) != n_rows
    ):
        raise ValueError("time validation requires one time value per training row")
    if validation_plan.predefined_fold_column is not None and (
        dataset.predefined_folds is None or len(dataset.predefined_folds) != n_rows
    ):
        raise ValueError("predefined validation requires one fold value per training row")


def _sample_random(
    positions: PositionArray,
    *,
    fraction: float,
    seed: int,
) -> PositionArray:
    desired = _desired_row_count(len(positions), fraction)
    generator = np.random.default_rng(seed)
    selected_offsets = np.sort(generator.choice(len(positions), size=desired, replace=False))
    return positions[selected_offsets].astype(np.int64, copy=False)


def _sample_stratified(
    positions: PositionArray,
    target: pd.Series,
    *,
    fraction: float,
    seed: int,
) -> PositionArray:
    labels = target.iloc[positions].reset_index(drop=True)
    codes, unique_labels = pd.factorize(labels, sort=False, use_na_sentinel=False)
    class_count = len(unique_labels)
    desired = max(_desired_row_count(len(positions), fraction), class_count)
    desired = min(len(positions), desired)
    counts = np.bincount(codes, minlength=class_count).astype(np.int64, copy=False)
    allocation = _stratified_allocation(counts, desired)

    generator = np.random.default_rng(seed)
    selected = np.zeros(len(positions), dtype=np.bool_)
    for class_code, take in enumerate(allocation.tolist()):
        offsets = np.flatnonzero(codes == class_code)
        chosen = generator.choice(offsets, size=take, replace=False)
        selected[chosen] = True
    return positions[selected].astype(np.int64, copy=False)


def _stratified_allocation(counts: PositionArray, desired: int) -> PositionArray:
    proportions = counts.astype(np.float64) * (float(desired) / float(counts.sum()))
    allocation = np.floor(proportions).astype(np.int64)
    allocation = np.minimum(counts, np.maximum(allocation, 1))

    while int(allocation.sum()) < desired:
        candidates = np.flatnonzero(allocation < counts).tolist()
        selected = max(
            candidates,
            key=lambda index: (
                float(proportions[index] - allocation[index]),
                int(counts[index] - allocation[index]),
                -index,
            ),
        )
        allocation[selected] += 1
    while int(allocation.sum()) > desired:
        candidates = np.flatnonzero(allocation > 1).tolist()
        selected = max(
            candidates,
            key=lambda index: (
                float(allocation[index] - proportions[index]),
                int(allocation[index]),
                -index,
            ),
        )
        allocation[selected] -= 1
    return allocation


def _sample_whole_groups(
    dataset: LoadedDataset,
    positions: PositionArray,
    *,
    fraction: float,
    seed: int,
    task: TaskType,
) -> PositionArray:
    assert dataset.groups is not None
    group_values = dataset.groups.iloc[positions].reset_index(drop=True)
    if bool(group_values.isna().any()):
        raise ValueError("group values must not be missing")
    group_codes, unique_groups = pd.factorize(group_values, sort=False)
    generator = np.random.default_rng(seed)
    shuffled_codes = generator.permutation(len(unique_groups)).astype(np.int64).tolist()
    desired_rows = _desired_row_count(len(positions), fraction)
    selected_codes: list[int] = []
    selected_set: set[int] = set()

    if task in _CLASSIFICATION_TASKS:
        labels = dataset.y.iloc[positions].reset_index(drop=True)
        class_codes, unique_classes = pd.factorize(labels, sort=False, use_na_sentinel=False)
        uncovered = set(range(len(unique_classes)))
        while uncovered:
            best_group: int | None = None
            best_coverage: set[int] = set()
            for group_code in shuffled_codes:
                if group_code in selected_set:
                    continue
                coverage = set(class_codes[group_codes == group_code].tolist()).intersection(
                    uncovered
                )
                if len(coverage) > len(best_coverage):
                    best_group = group_code
                    best_coverage = coverage
            if best_group is None:
                break
            selected_codes.append(best_group)
            selected_set.add(best_group)
            uncovered.difference_update(best_coverage)

    selected_rows = int(np.isin(group_codes, selected_codes).sum())
    for group_code in shuffled_codes:
        if selected_rows >= desired_rows and selected_codes:
            break
        if group_code in selected_set:
            continue
        selected_codes.append(group_code)
        selected_set.add(group_code)
        selected_rows += int((group_codes == group_code).sum())

    mask = np.isin(group_codes, selected_codes)
    return positions[mask].astype(np.int64, copy=False)


def _sample_time_prefix(
    dataset: LoadedDataset,
    fold: FoldAssignment,
    positions: PositionArray,
    *,
    fraction: float,
    task: TaskType,
) -> PositionArray:
    assert dataset.time_values is not None
    train_values = _sortable_time_values(dataset.time_values.iloc[positions])
    validation_positions = np.asarray(fold.validation_positions, dtype=np.int64)
    validation_values = _sortable_time_values(dataset.time_values.iloc[validation_positions])
    if train_values.max() > validation_values.min():
        raise ValueError("time fold contains training rows from the validation future")

    order = np.argsort(train_values, kind="stable")
    ordered_positions = positions[order]
    desired = _desired_row_count(len(ordered_positions), fraction)
    if task in _CLASSIFICATION_TASKS:
        labels = dataset.y.iloc[ordered_positions].reset_index(drop=True)
        class_codes, unique_classes = pd.factorize(labels, sort=False, use_na_sentinel=False)
        first_occurrences = [
            int(np.flatnonzero(class_codes == code)[0]) for code in range(len(unique_classes))
        ]
        desired = max(desired, max(first_occurrences) + 1)
    return ordered_positions[:desired].astype(np.int64, copy=False)


def _sortable_time_values(series: pd.Series) -> NDArray[Any]:
    if bool(series.isna().any()):
        raise ValueError("time values must not be missing")
    try:
        if pd.api.types.is_numeric_dtype(series.dtype):
            numeric = cast(pd.Series, pd.to_numeric(series, errors="raise"))
            values = np.asarray(numeric, dtype=np.float64)
            if not bool(np.isfinite(values).all()):
                raise ValueError("time values must be finite")
            return values
        parsed = cast(
            pd.Series,
            pd.to_datetime(series, errors="raise", utc=True, format="mixed"),
        )
        return np.asarray(parsed.astype("int64"), dtype=np.int64)
    except (TypeError, ValueError) as error:
        raise ValueError("time values cannot be ordered") from error


def _desired_row_count(size: int, fraction: float) -> int:
    return min(size, max(2, math.ceil(size * fraction)))


__all__ = ["FidelitySampler", "PositionArray"]

"""Deterministic and leakage-safe fidelity sampling tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autonomous_automl.contracts import (
    DatasetBundle,
    FidelitySpec,
    FoldAssignment,
    TaskType,
    ValidationPlan,
)
from autonomous_automl.data import LoadedDataset
from autonomous_automl.search.sampling import FidelitySampler

_DIGEST = "0" * 64


def _dataset(
    target: list[object],
    *,
    groups: list[object] | None = None,
    times: list[object] | None = None,
    predefined: list[int] | None = None,
) -> LoadedDataset:
    n_rows = len(target)
    features = pd.DataFrame({"feature": np.arange(n_rows, dtype=np.int64)})
    bundle = DatasetBundle(
        train_paths=[Path("/tmp/fidelity-sampling.csv")],
        target="target",
        feature_columns=["feature"],
        n_rows=n_rows,
        source_hashes={"/tmp/fidelity-sampling.csv": _DIGEST},
    )
    return LoadedDataset(
        X=features,
        y=pd.Series(target, name="target"),
        test_X=None,
        row_ids=None,
        test_row_ids=None,
        groups=None if groups is None else pd.Series(groups, name="group"),
        time_values=None if times is None else pd.Series(times, name="time"),
        predefined_folds=(None if predefined is None else pd.Series(predefined, name="fold")),
        bundle=bundle,
    )


def _plan(
    fold: FoldAssignment,
    *,
    splitter_name: str = "KFold",
    group_column: str | None = None,
    time_column: str | None = None,
    predefined_fold_column: str | None = None,
) -> ValidationPlan:
    companion = FoldAssignment(
        fold_index=1,
        train_positions=list(fold.train_positions),
        validation_positions=list(fold.validation_positions),
    )
    shuffled = time_column is None and predefined_fold_column is None
    return ValidationPlan(
        splitter_name=splitter_name,
        n_splits=2,
        shuffle=shuffled,
        random_seed=42 if shuffled else None,
        group_column=group_column,
        time_column=time_column,
        predefined_fold_column=predefined_fold_column,
        rationale=["unit-test materialized folds"],
        folds=[fold, companion],
    )


def _fidelity(fraction: float) -> FidelitySpec:
    return FidelitySpec(
        level=2 if fraction == 1.0 else 0,
        sample_fraction=fraction,
        n_folds=2,
        max_iterations=10,
        seeds=[42],
    )


def test_full_fraction_preserves_exact_fold_order_without_reading_test_x() -> None:
    dataset = _dataset([0, 1, 0, 1, 0, 1])
    fold = FoldAssignment(
        fold_index=0,
        train_positions=[4, 1, 3, 0],
        validation_positions=[2, 5],
    )
    plan = _plan(fold)
    # A non-DataFrame sentinel makes any accidental test_X inspection fail loudly.
    dataset.test_X = object()  # type: ignore[assignment]

    sampled = FidelitySampler.sample_training_positions(
        dataset,
        plan,
        fold,
        _fidelity(1.0),
        7,
        TaskType.BINARY_CLASSIFICATION,
    )

    assert sampled.dtype == np.int64
    assert sampled.tolist() == fold.train_positions


def test_classification_sampling_is_seeded_stratified_and_keeps_a_rare_class() -> None:
    dataset = _dataset([0] * 19 + [1, 0])
    fold = FoldAssignment(
        fold_index=0,
        train_positions=list(range(20)),
        validation_positions=[20],
    )
    plan = _plan(fold)
    fidelity = _fidelity(0.1)

    first = FidelitySampler.sample_training_positions(
        dataset, plan, fold, fidelity, 123, TaskType.BINARY_CLASSIFICATION
    )
    repeated = FidelitySampler.sample_training_positions(
        dataset, plan, fold, fidelity, 123, TaskType.BINARY_CLASSIFICATION
    )
    another_seed = FidelitySampler.sample_training_positions(
        dataset, plan, fold, fidelity, 321, TaskType.BINARY_CLASSIFICATION
    )

    assert first.tolist() == repeated.tolist()
    assert first.tolist() != another_seed.tolist()
    assert len(first) == 2
    assert set(dataset.y.iloc[first]) == {0, 1}
    assert 19 in first


def test_multiclass_sampling_may_exceed_tiny_fraction_to_retain_every_class() -> None:
    dataset = _dataset(["common"] * 12 + ["rare-a", "rare-b", "held-out"])
    fold = FoldAssignment(
        fold_index=0,
        train_positions=list(range(14)),
        validation_positions=[14],
    )
    plan = _plan(fold)

    sampled = FidelitySampler.sample_training_positions(
        dataset,
        plan,
        fold,
        _fidelity(0.01),
        42,
        TaskType.MULTICLASS_CLASSIFICATION,
    )

    assert len(sampled) == 3
    assert set(dataset.y.iloc[sampled]) == {"common", "rare-a", "rare-b"}


def test_regression_sampling_is_random_seeded_and_confined_to_training_pool() -> None:
    dataset = _dataset([float(index) for index in range(30)])
    fold = FoldAssignment(
        fold_index=0,
        train_positions=list(range(25)),
        validation_positions=list(range(25, 30)),
    )
    plan = _plan(fold)
    fidelity = _fidelity(0.4)

    first = FidelitySampler.sample_training_positions(
        dataset, plan, fold, fidelity, 11, TaskType.REGRESSION
    )
    repeated = FidelitySampler.sample_training_positions(
        dataset, plan, fold, fidelity, 11, TaskType.REGRESSION
    )
    another_seed = FidelitySampler.sample_training_positions(
        dataset, plan, fold, fidelity, 12, TaskType.REGRESSION
    )

    assert first.tolist() == repeated.tolist()
    assert first.tolist() != another_seed.tolist()
    assert len(first) == 10
    assert set(first).issubset(fold.train_positions)
    assert set(first).isdisjoint(fold.validation_positions)


def test_group_sampling_selects_complete_groups_and_preserves_rare_class_group() -> None:
    groups = ["a"] * 3 + ["b"] * 2 + ["c"] * 4 + ["rare", "validation", "validation"]
    target = [0] * 9 + [1, 0, 1]
    dataset = _dataset(target, groups=groups)
    fold = FoldAssignment(
        fold_index=0,
        train_positions=list(range(10)),
        validation_positions=[10, 11],
    )
    plan = _plan(
        fold,
        splitter_name="StratifiedGroupKFold",
        group_column="group",
    )

    sampled = FidelitySampler.sample_training_positions(
        dataset,
        plan,
        fold,
        _fidelity(0.2),
        9,
        TaskType.BINARY_CLASSIFICATION,
    )

    assert 9 in sampled
    selected = set(sampled.tolist())
    for group in {groups[position] for position in fold.train_positions}:
        group_pool = {position for position in fold.train_positions if groups[position] == group}
        assert group_pool.issubset(selected) or group_pool.isdisjoint(selected)


def test_time_sampling_returns_an_ordered_historical_prefix() -> None:
    day = pd.Timestamp("2024-01-01")
    offsets = [4, 1, 3, 0, 2, 5, 6, 7, 8, 9]
    dataset = _dataset(
        [float(index) for index in range(10)],
        times=[day + pd.Timedelta(days=offset) for offset in offsets],
    )
    fold = FoldAssignment(
        fold_index=0,
        train_positions=list(range(8)),
        validation_positions=[8, 9],
    )
    plan = _plan(fold, splitter_name="TimeSeriesSplit", time_column="time")

    sampled = FidelitySampler.sample_training_positions(
        dataset,
        plan,
        fold,
        _fidelity(0.5),
        999,
        TaskType.REGRESSION,
    )

    assert sampled.tolist() == [3, 1, 4, 2]
    sampled_times = dataset.time_values.iloc[sampled]
    assert sampled_times.is_monotonic_increasing
    assert sampled_times.max() <= dataset.time_values.iloc[fold.validation_positions].min()


def test_time_sampling_rejects_a_fold_with_future_training_rows() -> None:
    dataset = _dataset(
        [float(index) for index in range(6)],
        times=[0, 1, 2, 10, 4, 5],
    )
    fold = FoldAssignment(
        fold_index=0,
        train_positions=[0, 1, 2, 3],
        validation_positions=[4, 5],
    )
    plan = _plan(fold, splitter_name="TimeSeriesSplit", time_column="time")

    with pytest.raises(ValueError, match="validation future"):
        FidelitySampler.sample_training_positions(
            dataset,
            plan,
            fold,
            _fidelity(0.5),
            42,
            TaskType.REGRESSION,
        )


def test_predefined_sampling_never_leaves_the_materialized_training_pool() -> None:
    dataset = _dataset(
        [float(index) for index in range(8)],
        predefined=[-1, 0, -1, 0, 1, 1, 2, 2],
    )
    fold = FoldAssignment(
        fold_index=0,
        train_positions=[0, 2, 4, 5, 6, 7],
        validation_positions=[1, 3],
    )
    plan = _plan(
        fold,
        splitter_name="PredefinedSplit",
        predefined_fold_column="fold",
    )

    sampled = FidelitySampler.sample_training_positions(
        dataset,
        plan,
        fold,
        _fidelity(0.34),
        17,
        TaskType.REGRESSION,
    )

    assert len(sampled) == 3
    assert set(sampled).issubset(fold.train_positions)
    assert set(sampled).isdisjoint(fold.validation_positions)

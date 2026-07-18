"""M2 tests for deterministic, training-only dataset profiling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autonomous_automl import AutoMLConfig
from autonomous_automl.contracts import TaskType
from autonomous_automl.data import LoadedDataset, load_dataset
from autonomous_automl.profiling import (
    DatasetProfiler,
    InferredType,
    infer_column_type,
    profile_dataset,
)
from autonomous_automl.utils.errors import DataValidationError


def _load_frame(
    tmp_path: Path,
    frame: pd.DataFrame,
    *,
    task: str = "auto",
) -> LoadedDataset:
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    config = AutoMLConfig(target="target", task=task, budget_seconds=60)
    return load_dataset(path, config)


def _mixed_frame() -> pd.DataFrame:
    n_rows = 40
    return pd.DataFrame(
        {
            "numeric": [*range(38), np.inf, np.nan],
            "category": (["a", "b", "c", None] * 10),
            "flag": ([True, False] * 20),
            "event_date": pd.date_range("2024-01-01", periods=n_rows).strftime("%Y-%m-%d"),
            "description": [
                f"A deliberately long free-text observation number {index:03d} with details"
                for index in range(n_rows)
            ],
            "all_missing": [None] * n_rows,
            "constant": ["same"] * n_rows,
            "almost_constant": [0] * 39 + [1],
            "customer_id": [f"C-{index:04d}" for index in range(n_rows)],
            "group_code": [f"group-{index % 4}" for index in range(n_rows)],
            "target": [0] * 30 + [1] * 10,
        }
    )


@pytest.mark.parametrize(
    ("series", "expected"),
    [
        (pd.Series([1.0, 2.0, np.nan]), InferredType.NUMERIC),
        (pd.Series([True, False, None]), InferredType.BOOLEAN),
        (pd.Series(["2024-01-01", "2024-02-03"]), InferredType.DATETIME),
        (
            pd.Series(
                [
                    "A long and unique free-text sentence for the first row.",
                    "Another long and unique free-text sentence for row two.",
                ]
            ),
            InferredType.TEXT,
        ),
        (pd.Series([None, None]), InferredType.ALL_MISSING),
        (pd.Series(["small", "labels", "small"]), InferredType.CATEGORICAL),
    ],
)
def test_column_type_inference(series: pd.Series, expected: InferredType) -> None:
    assert infer_column_type(series) is expected


def test_profiler_computes_required_mixed_statistics(tmp_path: Path) -> None:
    profile = profile_dataset(_load_frame(tmp_path, _mixed_frame()))

    assert profile.inferred_task is TaskType.BINARY_CLASSIFICATION
    assert profile.n_rows == 40
    assert profile.n_features == 10
    assert "numeric" in profile.numeric_columns
    assert "category" in profile.categorical_columns
    assert "all_missing" in profile.categorical_columns
    assert "flag" in profile.boolean_columns
    assert "event_date" in profile.datetime_columns
    assert "description" in profile.text_columns
    assert profile.inferred_types["all_missing"] == "all_missing"
    assert set(profile.constant_columns) >= {"all_missing", "constant"}
    assert "almost_constant" in profile.near_constant_columns
    assert "customer_id" in profile.id_candidates
    assert "event_date" in profile.time_candidates
    assert "group_code" in profile.group_candidates
    assert profile.missing_ratios["numeric"] == pytest.approx(0.025)
    assert profile.cardinalities["category"] == 3
    assert profile.unique_ratios["customer_id"] == 1.0
    assert profile.infinite_counts["numeric"] == 1
    assert np.isfinite(profile.numeric_skewness["numeric"])
    assert profile.estimated_memory_mb > 0


def test_profiler_target_profile_and_landmarks_are_aggregated(tmp_path: Path) -> None:
    profile = profile_dataset(_load_frame(tmp_path, _mixed_frame()))

    assert profile.target_profile["class_counts"] == {"0": 30, "1": 10}
    assert profile.target_profile["minority_class_fraction"] == 0.25
    assert profile.target_profile["imbalance_ratio"] == 3.0
    assert profile.landmarks["missing_fraction"] > 0
    assert profile.landmarks["categorical_fraction"] > 0
    assert profile.landmarks["feature_to_row_ratio"] == 0.25
    assert all(np.isfinite(value) for value in profile.landmarks.values())


def test_profiler_detects_duplicate_feature_rows(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "number": [1, 1, 2, 2],
            "category": ["a", "a", "b", "b"],
            "target": [0, 1, 0, 1],
        }
    )

    profile = profile_dataset(_load_frame(tmp_path, frame))

    assert profile.duplicate_row_count == 2
    assert profile.potential_cross_fold_duplicate_count == 4
    assert profile.landmarks["duplicate_ratio"] == 0.5


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ([0, 1] * 15, TaskType.BINARY_CLASSIFICATION),
        (["a", "b", "c"] * 10, TaskType.MULTICLASS_CLASSIFICATION),
        (list(np.linspace(0.1, 9.9, 30)), TaskType.REGRESSION),
    ],
)
def test_auto_task_inference(tmp_path: Path, target: list[object], expected: TaskType) -> None:
    frame = pd.DataFrame({"feature": range(30), "target": target})

    profile = profile_dataset(_load_frame(tmp_path, frame))

    assert profile.inferred_task is expected


def test_explicit_regression_records_distribution(tmp_path: Path) -> None:
    frame = pd.DataFrame({"feature": range(5), "target": [1.5, 2.5, 3.5, 4.5, 5.5]})

    profile = profile_dataset(_load_frame(tmp_path, frame, task="regression"), TaskType.REGRESSION)

    assert profile.target_profile["mean"] == 3.5
    assert profile.target_profile["median"] == 3.5
    assert profile.target_profile["minimum"] == 1.5
    assert profile.target_profile["maximum"] == 5.5


def test_profile_is_deterministic_and_json_round_trippable(tmp_path: Path) -> None:
    dataset = _load_frame(tmp_path, _mixed_frame())

    first = profile_dataset(dataset)
    second = profile_dataset(dataset)

    assert first == second
    assert first.from_json(first.to_json()) == first


def test_profile_hash_does_not_depend_on_test_file(tmp_path: Path) -> None:
    frame = _mixed_frame()
    train = tmp_path / "train.csv"
    first_test = tmp_path / "test-one.csv"
    second_test = tmp_path / "test-two.csv"
    frame.to_csv(train, index=False)
    frame.drop(columns="target").iloc[:2].to_csv(first_test, index=False)
    changed = frame.drop(columns="target").iloc[:2].copy()
    changed.loc[:, "category"] = "never-seen"
    changed.to_csv(second_test, index=False)

    first = load_dataset(
        train,
        AutoMLConfig(target="target", budget_seconds=60, test_path=first_test),
    )
    second = load_dataset(
        train,
        AutoMLConfig(target="target", budget_seconds=60, test_path=second_test),
    )

    assert profile_dataset(first) == profile_dataset(second)


@pytest.mark.parametrize(
    ("target", "task", "message"),
    [
        ([1, 1, 1], TaskType.AUTO, "at least two"),
        ([0, 1, 2], TaskType.BINARY_CLASSIFICATION, "exactly two"),
        (["a", "b", "c"], TaskType.REGRESSION, "numeric"),
    ],
)
def test_profiler_rejects_incompatible_targets(
    tmp_path: Path,
    target: list[object],
    task: TaskType,
    message: str,
) -> None:
    dataset = _load_frame(
        tmp_path,
        pd.DataFrame({"feature": range(len(target)), "target": target}),
    )

    with pytest.raises(DataValidationError, match=message):
        profile_dataset(dataset, task)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("near_constant_threshold", 0.5),
        ("near_constant_threshold", 1.01),
        ("id_unique_ratio_threshold", 0.5),
        ("id_unique_ratio_threshold", 1.01),
    ],
)
def test_profiler_rejects_invalid_thresholds(keyword: str, value: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        DatasetProfiler(**{keyword: value})

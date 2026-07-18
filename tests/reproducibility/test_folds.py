"""M3 acceptance tests for safe and reproducible materialized folds."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from autonomous_automl.contracts import AutoMLConfig, ValidationPlan
from autonomous_automl.data import LoadedDataset, load_dataset
from autonomous_automl.profiling import profile_dataset
from autonomous_automl.validation import ValidationPlanner, audit_validation, plan_validation


def load_frame(
    tmp_path: Path,
    frame: pd.DataFrame,
    *,
    random_seed: int = 42,
    group_column: str | None = None,
    time_column: str | None = None,
) -> tuple[LoadedDataset, AutoMLConfig]:
    train_path = tmp_path / f"train-{random_seed}.csv"
    frame.to_csv(train_path, index=False)
    config = AutoMLConfig(
        target="target",
        task="binary_classification",
        metric="roc_auc",
        budget_seconds=30,
        random_seed=random_seed,
        output_dir=tmp_path / f"run-{random_seed}",
        group_column=group_column,
        time_column=time_column,
    )
    return load_dataset(train_path, config), config


def test_e4_group_validation_never_shares_a_group_between_train_and_validation(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "signal": [float(index % 7) for index in range(30)],
            "group_id": [f"group-{index // 3}" for index in range(30)],
            "target": [index % 2 for index in range(30)],
        }
    )
    dataset, config = load_frame(tmp_path, frame, group_column="group_id")

    plan = plan_validation(dataset, profile_dataset(dataset), config)
    audit = audit_validation(dataset, plan)

    assert "GroupKFold" in plan.splitter_name
    assert plan.group_column == "group_id"
    assert dataset.groups is not None
    assert "group_id" not in dataset.X.columns
    for fold in plan.folds:
        train_groups = set(dataset.groups.iloc[fold.train_positions])
        validation_groups = set(dataset.groups.iloc[fold.validation_positions])
        assert train_groups.isdisjoint(validation_groups)
    assert audit.valid
    assert audit.group_overlap_counts == [0] * plan.n_splits


def test_time_validation_uses_only_strictly_earlier_training_rows(tmp_path: Path) -> None:
    chronological = pd.date_range("2024-01-01", periods=18, freq="D")
    source_order = [8, 0, 17, 4, 12, 2, 15, 6, 10, 1, 14, 3, 16, 5, 13, 7, 11, 9]
    frame = pd.DataFrame(
        {
            "signal": [float(index % 5) for index in range(18)],
            "event_time": [chronological[index].isoformat() for index in source_order],
            "target": [index % 2 for index in range(18)],
        }
    )
    dataset, config = load_frame(tmp_path, frame, time_column="event_time")

    plan = ValidationPlanner(max_splits=4).plan(dataset, profile_dataset(dataset), config)
    audit = audit_validation(dataset, plan)

    assert plan.splitter_name == "TimeSeriesSplit"
    assert plan.shuffle is False
    assert plan.random_seed is None
    assert dataset.time_values is not None
    comparable_time = pd.to_datetime(dataset.time_values, utc=True)
    for fold in plan.folds:
        latest_training_time = comparable_time.iloc[fold.train_positions].max()
        earliest_validation_time = comparable_time.iloc[fold.validation_positions].min()
        assert latest_training_time < earliest_validation_time
    assert audit.valid
    assert audit.temporal_violation_counts == [0] * plan.n_splits


def test_seeded_folds_are_identical_for_same_seed_and_change_for_another_seed(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "signal": [float(index) for index in range(60)],
            "category": [f"category-{index % 4}" for index in range(60)],
            "target": [index % 2 for index in range(60)],
        }
    )
    dataset_42, config_42 = load_frame(tmp_path, frame, random_seed=42)
    dataset_43, config_43 = load_frame(tmp_path, frame, random_seed=43)
    profile_42 = profile_dataset(dataset_42)
    profile_43 = profile_dataset(dataset_43)
    planner = ValidationPlanner(max_splits=5)

    first = planner.plan(dataset_42, profile_42, config_42)
    repeated = planner.plan(dataset_42, profile_42, config_42)
    different_seed = planner.plan(dataset_43, profile_43, config_43)

    assert first.folds == repeated.folds
    assert first.to_json() == repeated.to_json()
    assert first.random_seed == 42
    assert different_seed.random_seed == 43
    assert first.folds != different_seed.folds


def test_materialized_plan_round_trips_through_exported_json(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "signal": [float(index) for index in range(20)],
            "target": [index % 2 for index in range(20)],
        }
    )
    dataset, config = load_frame(tmp_path, frame)
    profile = profile_dataset(dataset)
    plan = plan_validation(dataset, profile, config)
    destination = tmp_path / "artifacts" / "validation_plan.json"

    plan.write_json(destination)
    restored = ValidationPlan.read_json(destination)

    assert restored == plan
    assert restored.schema_version == 1
    assert restored.dataset_hash == profile.dataset_hash
    assert len(restored.folds) == restored.n_splits
    assert [fold.fold_index for fold in restored.folds] == list(range(restored.n_splits))
    assert audit_validation(dataset, restored).valid

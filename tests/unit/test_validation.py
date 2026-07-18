"""Unit tests for validation strategy planning and fold auditing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from autonomous_automl import AutoMLConfig
from autonomous_automl.contracts import (
    FoldAssignment,
    ValidationAudit,
    ValidationPlan,
)
from autonomous_automl.data import LoadedDataset, load_dataset
from autonomous_automl.profiling import profile_dataset
from autonomous_automl.utils.errors import ConfigurationError, DataValidationError
from autonomous_automl.validation import ValidationPlanner, audit_validation, plan_validation


def _load(
    tmp_path: Path,
    frame: pd.DataFrame,
    **config_updates: object,
) -> tuple[LoadedDataset, AutoMLConfig]:
    source = tmp_path / "train.csv"
    frame.to_csv(source, index=False)
    values: dict[str, object] = {"target": "target", "budget_seconds": 60}
    values.update(config_updates)
    config = AutoMLConfig.model_validate(values)
    return load_dataset(source, config), config


def test_binary_classification_uses_stratified_folds(tmp_path: Path) -> None:
    dataset, config = _load(
        tmp_path,
        pd.DataFrame({"feature": range(30), "target": [0, 1] * 15}),
    )

    plan = ValidationPlanner(max_splits=3).plan(dataset, profile_dataset(dataset), config)
    audit = audit_validation(dataset, plan)

    assert plan.splitter_name == "StratifiedKFold"
    assert plan.n_splits == 3
    assert audit.validation_coverage_ratio == 1.0
    assert all(set(dataset.y.iloc[fold.validation_positions]) == {0, 1} for fold in plan.folds)


def test_regression_uses_seeded_kfold(tmp_path: Path) -> None:
    dataset, config = _load(
        tmp_path,
        pd.DataFrame({"feature": range(12), "target": [float(index) / 3 for index in range(12)]}),
        task="regression",
    )

    plan = plan_validation(dataset, profile_dataset(dataset, task="regression"), config)

    assert plan.splitter_name == "KFold"
    assert plan.shuffle
    assert plan.random_seed == config.random_seed
    assert audit_validation(dataset, plan).valid


def test_grouped_classification_falls_back_when_stratification_is_impossible(
    tmp_path: Path,
) -> None:
    dataset, config = _load(
        tmp_path,
        pd.DataFrame(
            {
                "feature": range(8),
                "group": ["a", "a", "a", "a", "b", "b", "b", "b"],
                "target": [0, 0, 0, 0, 1, 1, 1, 1],
            }
        ),
        group_column="group",
    )

    plan = plan_validation(dataset, profile_dataset(dataset), config)

    assert plan.splitter_name == "GroupKFold"
    assert any("stratified grouping unavailable" in reason for reason in plan.rationale)
    assert audit_validation(dataset, plan).group_overlap_counts == [0] * plan.n_splits


def test_predefined_folds_are_materialized_and_training_only_rows_are_allowed(
    tmp_path: Path,
) -> None:
    dataset, config = _load(
        tmp_path,
        pd.DataFrame(
            {
                "feature": range(8),
                "fold": [-1, -1, 0, 0, 1, 1, 2, 2],
                "target": [0, 1, 0, 1, 0, 1, 0, 1],
            }
        ),
        predefined_fold_column="fold",
    )

    plan = plan_validation(dataset, profile_dataset(dataset), config)
    audit = audit_validation(dataset, plan)

    assert plan.splitter_name == "PredefinedSplit"
    assert plan.n_splits == 3
    assert audit.valid
    assert audit.validation_coverage_ratio == 0.75
    assert any("does not cover" in warning for warning in audit.warnings)


def test_planner_rejects_ambiguous_validation_roles(tmp_path: Path) -> None:
    dataset, config = _load(
        tmp_path,
        pd.DataFrame(
            {
                "feature": range(8),
                "group": ["a", "a", "b", "b", "c", "c", "d", "d"],
                "time": pd.date_range("2024-01-01", periods=8),
                "target": [0, 1] * 4,
            }
        ),
        group_column="group",
        time_column="time",
    )

    with pytest.raises(ConfigurationError, match="only one"):
        plan_validation(dataset, profile_dataset(dataset), config)


def test_planner_rejects_invalid_group_and_time_values(tmp_path: Path) -> None:
    grouped, group_config = _load(
        tmp_path,
        pd.DataFrame({"feature": range(6), "group": ["same"] * 6, "target": [0, 1] * 3}),
        group_column="group",
    )
    with pytest.raises(DataValidationError, match="two distinct groups"):
        plan_validation(grouped, profile_dataset(grouped), group_config)

    timed_path = tmp_path / "timed"
    timed_path.mkdir()
    timed, time_config = _load(
        timed_path,
        pd.DataFrame(
            {
                "feature": range(6),
                "event_time": ["bad", "values", "cannot", "be", "time", "ordered"],
                "target": [0, 1] * 3,
            }
        ),
        time_column="event_time",
    )
    with pytest.raises(DataValidationError, match="cannot be ordered"):
        plan_validation(timed, profile_dataset(timed), time_config)


def test_audit_reports_constructed_overlap_and_out_of_bounds(tmp_path: Path) -> None:
    dataset, _ = _load(
        tmp_path,
        pd.DataFrame({"feature": range(6), "target": [0, 1] * 3}),
    )
    unsafe_fold = FoldAssignment.model_construct(
        fold_index=0,
        train_positions=[0, 1, 2, 99],
        validation_positions=[2, 3],
    )
    unsafe_plan = ValidationPlan.model_construct(
        schema_version=1,
        splitter_name="UnsafeForAuditTest",
        n_splits=1,
        shuffle=False,
        random_seed=None,
        group_column=None,
        time_column=None,
        predefined_fold_column=None,
        rationale=["test only"],
        dataset_hash=None,
        folds=[unsafe_fold],
    )

    audit = audit_validation(dataset, unsafe_plan)

    assert not audit.valid
    assert audit.train_validation_overlap_counts == [1]
    assert audit.out_of_bounds_count == 1


def test_validation_audit_round_trips_as_json(tmp_path: Path) -> None:
    dataset, config = _load(
        tmp_path,
        pd.DataFrame({"feature": range(12), "target": [0, 1] * 6}),
    )
    audit = audit_validation(dataset, plan_validation(dataset, profile_dataset(dataset), config))

    assert ValidationAudit.from_json(audit.to_json()) == audit


@pytest.mark.parametrize("max_splits", [0, 1])
def test_planner_requires_at_least_two_splits(max_splits: int) -> None:
    with pytest.raises(ValueError, match="at least two"):
        ValidationPlanner(max_splits=max_splits)

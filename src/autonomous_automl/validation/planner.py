"""Deterministic selection and materialization of validation folds."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    PredefinedSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

from autonomous_automl.contracts import (
    AutoMLConfig,
    DatasetProfile,
    FoldAssignment,
    TaskType,
    ValidationPlan,
)
from autonomous_automl.data import LoadedDataset
from autonomous_automl.utils.errors import ConfigurationError, DataValidationError
from autonomous_automl.validation.audit import audit_validation

_CLASSIFICATION_TASKS = {
    TaskType.BINARY_CLASSIFICATION,
    TaskType.MULTICLASS_CLASSIFICATION,
}


@dataclass(frozen=True, slots=True)
class ValidationPlanner:
    """Choose one explainable V1 splitter and persist its exact row positions."""

    max_splits: int = 5

    def __post_init__(self) -> None:
        if self.max_splits < 2:
            raise ValueError("max_splits must be at least two")

    def plan(
        self,
        dataset: LoadedDataset,
        profile: DatasetProfile,
        config: AutoMLConfig,
    ) -> ValidationPlan:
        """Build and audit deterministic folds without fitting any transformer."""
        _validate_requested_roles(dataset, config)
        task = profile.inferred_task if config.task == TaskType.AUTO else config.task

        if config.predefined_fold_column is not None:
            plan = self._predefined(dataset, profile, config)
        elif config.time_column is not None:
            plan = self._time_series(dataset, profile, config)
        elif config.group_column is not None:
            plan = self._grouped(dataset, profile, config, task)
        elif task in _CLASSIFICATION_TASKS:
            plan = self._stratified(dataset, profile, config)
        else:
            plan = self._kfold(dataset, profile, config)

        audit = audit_validation(dataset, plan)
        if not audit.valid:
            raise DataValidationError(
                "materialized folds failed safety audit: "
                f"overlap={audit.train_validation_overlap_counts}, "
                f"group={audit.group_overlap_counts}, "
                f"time={audit.temporal_violation_counts}, "
                f"out_of_bounds={audit.out_of_bounds_count}"
            )
        if audit.warnings:
            plan = plan.model_copy(
                update={"rationale": [*plan.rationale, *audit.warnings]},
                deep=True,
            )
        return plan

    def _predefined(
        self,
        dataset: LoadedDataset,
        profile: DatasetProfile,
        config: AutoMLConfig,
    ) -> ValidationPlan:
        assert dataset.predefined_folds is not None
        numeric_folds = cast(
            pd.Series,
            pd.to_numeric(dataset.predefined_folds, errors="raise"),
        )
        values = np.asarray(numeric_folds, dtype=np.float64)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise DataValidationError("predefined folds must be finite integer values")
        fold_values = values.astype(int)
        if (fold_values < -1).any():
            raise DataValidationError("predefined fold values must be -1 or non-negative")
        n_splits = len(set(fold_values.tolist()).difference({-1}))
        if n_splits < 2:
            raise DataValidationError("predefined validation requires at least two folds")
        splitter = PredefinedSplit(fold_values)
        folds = _materialize(splitter.split(), expected_splits=n_splits)
        return ValidationPlan(
            splitter_name="PredefinedSplit",
            n_splits=n_splits,
            shuffle=False,
            random_seed=None,
            predefined_fold_column=config.predefined_fold_column,
            rationale=["explicit predefined fold column", "fold positions persisted"],
            dataset_hash=profile.dataset_hash,
            folds=folds,
        )

    def _time_series(
        self,
        dataset: LoadedDataset,
        profile: DatasetProfile,
        config: AutoMLConfig,
    ) -> ValidationPlan:
        assert dataset.time_values is not None
        if len(dataset.X) < 3:
            raise DataValidationError("time-series validation requires at least three rows")
        order = _stable_time_order(dataset.time_values)
        n_splits = min(self.max_splits, len(dataset.X) - 1)
        splitter = TimeSeriesSplit(n_splits=n_splits)
        folds = [
            FoldAssignment(
                fold_index=index,
                train_positions=order[train].astype(int).tolist(),
                validation_positions=order[validation].astype(int).tolist(),
            )
            for index, (train, validation) in enumerate(splitter.split(order))
        ]
        return ValidationPlan(
            splitter_name="TimeSeriesSplit",
            n_splits=n_splits,
            shuffle=False,
            random_seed=None,
            time_column=config.time_column,
            rationale=["explicit time column", "stable chronological ordering", "no shuffling"],
            dataset_hash=profile.dataset_hash,
            folds=folds,
        )

    def _grouped(
        self,
        dataset: LoadedDataset,
        profile: DatasetProfile,
        config: AutoMLConfig,
        task: TaskType,
    ) -> ValidationPlan:
        assert dataset.groups is not None
        if dataset.groups.isna().any():
            raise DataValidationError("group_column must not contain missing values")
        n_groups = int(dataset.groups.nunique(dropna=False))
        if n_groups < 2:
            raise DataValidationError("group validation requires at least two distinct groups")

        use_stratified = task in _CLASSIFICATION_TASKS
        n_splits = min(self.max_splits, n_groups)
        if use_stratified:
            minimum_class_groups = _minimum_groups_per_class(dataset.y, dataset.groups)
            n_splits = min(n_splits, minimum_class_groups)
            use_stratified = n_splits >= 2

        if use_stratified:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=config.random_seed,
            )
            raw_folds = splitter.split(dataset.X, dataset.y, dataset.groups)
            name = "StratifiedGroupKFold"
            shuffle = True
            seed: int | None = config.random_seed
            rationale = ["classification target", "group isolation", "class stratification"]
        else:
            n_splits = min(self.max_splits, n_groups)
            splitter = GroupKFold(n_splits=n_splits)
            raw_folds = splitter.split(dataset.X, dataset.y, dataset.groups)
            name = "GroupKFold"
            shuffle = False
            seed = None
            rationale = ["explicit group column", "group isolation"]
            if task in _CLASSIFICATION_TASKS:
                rationale.append("stratified grouping unavailable for class/group distribution")

        folds = _materialize(raw_folds, expected_splits=n_splits)
        return ValidationPlan(
            splitter_name=name,
            n_splits=n_splits,
            shuffle=shuffle,
            random_seed=seed,
            group_column=config.group_column,
            rationale=rationale,
            dataset_hash=profile.dataset_hash,
            folds=folds,
        )

    def _stratified(
        self,
        dataset: LoadedDataset,
        profile: DatasetProfile,
        config: AutoMLConfig,
    ) -> ValidationPlan:
        minimum_class_count = int(dataset.y.value_counts(dropna=False).min())
        n_splits = min(self.max_splits, minimum_class_count)
        if n_splits < 2:
            raise DataValidationError(
                "stratified validation requires at least two rows in every class"
            )
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=config.random_seed,
        )
        folds = _materialize(
            splitter.split(dataset.X, dataset.y),
            expected_splits=n_splits,
        )
        return ValidationPlan(
            splitter_name="StratifiedKFold",
            n_splits=n_splits,
            shuffle=True,
            random_seed=config.random_seed,
            rationale=["classification target", "class balance", "seeded shuffling"],
            dataset_hash=profile.dataset_hash,
            folds=folds,
        )

    def _kfold(
        self,
        dataset: LoadedDataset,
        profile: DatasetProfile,
        config: AutoMLConfig,
    ) -> ValidationPlan:
        n_splits = min(self.max_splits, len(dataset.X))
        if n_splits < 2:
            raise DataValidationError("KFold validation requires at least two rows")
        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=config.random_seed,
        )
        folds = _materialize(splitter.split(dataset.X), expected_splits=n_splits)
        return ValidationPlan(
            splitter_name="KFold",
            n_splits=n_splits,
            shuffle=True,
            random_seed=config.random_seed,
            rationale=["regression target", "seeded shuffling"],
            dataset_hash=profile.dataset_hash,
            folds=folds,
        )


def plan_validation(
    dataset: LoadedDataset,
    profile: DatasetProfile,
    config: AutoMLConfig,
) -> ValidationPlan:
    """Plan validation using the default maximum of five folds."""
    return ValidationPlanner().plan(dataset, profile, config)


def _validate_requested_roles(dataset: LoadedDataset, config: AutoMLConfig) -> None:
    strategies = [
        config.group_column is not None,
        config.time_column is not None,
        config.predefined_fold_column is not None,
    ]
    if sum(strategies) > 1:
        raise ConfigurationError(
            "V1 accepts only one of group_column, time_column, or predefined_fold_column"
        )
    if config.group_column is not None and dataset.groups is None:
        raise DataValidationError("configured group values are missing from LoadedDataset")
    if config.time_column is not None and dataset.time_values is None:
        raise DataValidationError("configured time values are missing from LoadedDataset")
    if config.predefined_fold_column is not None and dataset.predefined_folds is None:
        raise DataValidationError("configured predefined folds are missing from LoadedDataset")


def _materialize(
    raw_folds: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    expected_splits: int,
) -> list[FoldAssignment]:
    folds = [
        FoldAssignment(
            fold_index=index,
            train_positions=np.asarray(train, dtype=int).tolist(),
            validation_positions=np.asarray(validation, dtype=int).tolist(),
        )
        for index, (train, validation) in enumerate(raw_folds)
    ]
    if len(folds) != expected_splits:
        raise DataValidationError(
            f"splitter produced {len(folds)} folds; expected {expected_splits}"
        )
    return folds


def _stable_time_order(series: pd.Series) -> np.ndarray:
    if series.isna().any():
        raise DataValidationError("time_column must not contain missing values")
    try:
        if pd.api.types.is_numeric_dtype(series.dtype):
            numeric = cast(pd.Series, pd.to_numeric(series, errors="raise"))
            values = np.asarray(numeric, dtype=np.float64)
            if not np.isfinite(values).all():
                raise DataValidationError("time_column must contain finite values")
        else:
            parsed = cast(
                pd.Series,
                pd.to_datetime(series, errors="raise", utc=True, format="mixed"),
            )
            values = np.asarray(parsed.astype("int64"), dtype=np.int64)
    except (TypeError, ValueError) as error:
        raise DataValidationError("time_column contains values that cannot be ordered") from error
    return np.argsort(values, kind="stable")


def _minimum_groups_per_class(target: pd.Series, groups: pd.Series) -> int:
    frame = pd.DataFrame(
        {"target": target.reset_index(drop=True), "group": groups.reset_index(drop=True)}
    )
    counts = frame.groupby("target", dropna=False)["group"].nunique()
    return int(np.min(np.asarray(counts, dtype=np.int64)))


__all__ = ["ValidationPlanner", "plan_validation"]

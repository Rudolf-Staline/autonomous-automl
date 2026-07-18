"""Leakage-safe, deterministic profiling of training features and target."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype

from autonomous_automl.contracts import DatasetProfile, TaskType
from autonomous_automl.data.models import LoadedDataset
from autonomous_automl.profiling.type_inference import (
    InferredType,
    infer_dataframe_types,
)
from autonomous_automl.utils.errors import DataValidationError
from autonomous_automl.utils.json import JsonValue

_ID_NAME_PATTERN = re.compile(r"(?:^|_)(?:id|uuid|key|identifier)(?:_|$)", re.I)
_TIME_NAME_PATTERN = re.compile(r"(?:^|_)(?:date|time|timestamp|datetime)(?:_|$)", re.I)
_GROUP_NAME_PATTERN = re.compile(
    r"(?:^|_)(?:group|entity|subject|customer|user|account|patient|site|store)(?:_|$)",
    re.I,
)


@dataclass(frozen=True, slots=True)
class DatasetProfiler:
    """Compute compact metadata using training ``X`` and ``y`` only."""

    near_constant_threshold: float = 0.95
    id_unique_ratio_threshold: float = 0.98

    def __post_init__(self) -> None:
        if not 0.5 < self.near_constant_threshold <= 1.0:
            raise ValueError("near_constant_threshold must be in (0.5, 1.0]")
        if not 0.5 < self.id_unique_ratio_threshold <= 1.0:
            raise ValueError("id_unique_ratio_threshold must be in (0.5, 1.0]")

    def profile(
        self,
        dataset: LoadedDataset,
        task: TaskType = TaskType.AUTO,
    ) -> DatasetProfile:
        """Profile training data without inspecting ``test_X`` or test row IDs."""
        frame = dataset.X
        target = dataset.y
        _validate_training_data(frame, target)
        resolved_task = _resolve_task(target, task)
        inferred = infer_dataframe_types(frame)

        missing_ratios: dict[str, float] = {}
        cardinalities: dict[str, int] = {}
        unique_ratios: dict[str, float] = {}
        infinite_counts: dict[str, int] = {}
        numeric_skewness: dict[str, float] = {}
        constant_columns: list[str] = []
        near_constant_columns: list[str] = []
        id_candidates: list[str] = []
        time_candidates: list[str] = []
        group_candidates: list[str] = []

        for column in frame.columns:
            name = str(column)
            series = cast(pd.Series, frame[column])
            non_missing_count = int(cast(Any, series.notna().sum()))
            cardinality = int(cast(Any, series.nunique(dropna=True)))
            unique_ratio = float(cardinality / non_missing_count) if non_missing_count > 0 else 0.0
            missing_ratios[name] = float(cast(Any, series.isna().mean()))
            cardinalities[name] = cardinality
            unique_ratios[name] = unique_ratio
            infinite_counts[name] = _infinite_count(series)

            if cardinality <= 1:
                constant_columns.append(name)
            elif _dominant_ratio(series) >= self.near_constant_threshold:
                near_constant_columns.append(name)

            inferred_type = inferred[name]
            if inferred_type is InferredType.NUMERIC:
                numeric_skewness[name] = _finite_skewness(series)
            if _is_id_candidate(
                name,
                series,
                inferred_type=inferred_type,
                unique_ratio=unique_ratio,
                threshold=self.id_unique_ratio_threshold,
            ):
                id_candidates.append(name)
            if inferred_type is InferredType.DATETIME or _TIME_NAME_PATTERN.search(name):
                time_candidates.append(name)
            if _is_group_candidate(name, cardinality=cardinality, n_rows=len(frame)):
                group_candidates.append(name)

        numeric_columns = _columns_with_role(inferred, InferredType.NUMERIC)
        categorical_columns = _columns_with_role(inferred, InferredType.CATEGORICAL)
        categorical_columns.extend(_columns_with_role(inferred, InferredType.ALL_MISSING))
        boolean_columns = _columns_with_role(inferred, InferredType.BOOLEAN)
        datetime_columns = _columns_with_role(inferred, InferredType.DATETIME)
        text_columns = _columns_with_role(inferred, InferredType.TEXT)

        duplicate_row_count = int(frame.duplicated(keep="first").sum())
        duplicate_group_rows = int(frame.duplicated(keep=False).sum())
        estimated_memory_mb = float(
            (frame.memory_usage(index=True, deep=True).sum() + target.memory_usage(deep=True))
            / (1024**2)
        )
        landmarks = _statistical_landmarks(
            frame,
            target,
            inferred=inferred,
            missing_ratios=missing_ratios,
            unique_ratios=unique_ratios,
            duplicate_row_count=duplicate_row_count,
        )
        return DatasetProfile(
            dataset_hash=_training_dataset_hash(dataset),
            n_rows=len(frame),
            n_features=frame.shape[1],
            inferred_task=resolved_task,
            inferred_types={name: inferred_type.value for name, inferred_type in inferred.items()},
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            boolean_columns=boolean_columns,
            datetime_columns=datetime_columns,
            text_columns=text_columns,
            constant_columns=constant_columns,
            near_constant_columns=near_constant_columns,
            id_candidates=id_candidates,
            time_candidates=time_candidates,
            group_candidates=group_candidates,
            missing_ratios=missing_ratios,
            cardinalities=cardinalities,
            unique_ratios=unique_ratios,
            numeric_skewness=numeric_skewness,
            infinite_counts=infinite_counts,
            target_profile=_target_profile(target, resolved_task),
            landmarks=landmarks,
            duplicate_row_count=duplicate_row_count,
            potential_cross_fold_duplicate_count=duplicate_group_rows,
            estimated_memory_mb=estimated_memory_mb,
        )


def profile_dataset(
    dataset: LoadedDataset,
    task: TaskType = TaskType.AUTO,
) -> DatasetProfile:
    """Profile a loaded training dataset with the default deterministic policy."""
    return DatasetProfiler().profile(dataset, task)


def _validate_training_data(frame: pd.DataFrame, target: pd.Series) -> None:
    if frame.empty:
        raise DataValidationError("cannot profile an empty training feature frame")
    if frame.shape[1] == 0:
        raise DataValidationError("cannot profile a dataset without feature columns")
    if len(frame) != len(target):
        raise DataValidationError("training features and target have different row counts")
    names = [str(column) for column in frame.columns]
    if len(names) != len(set(names)):
        raise DataValidationError("feature column names must be unique before profiling")
    if target.isna().any():
        raise DataValidationError("target must not contain missing values during profiling")
    if int(target.nunique(dropna=True)) < 2:
        raise DataValidationError("target must contain at least two distinct values")


def _resolve_task(target: pd.Series, requested: TaskType) -> TaskType:
    try:
        task = TaskType(requested)
    except ValueError as error:
        raise DataValidationError(f"unsupported task for profiling: {requested}") from error

    unique_count = int(target.nunique(dropna=True))
    if task is TaskType.BINARY_CLASSIFICATION:
        if unique_count != 2:
            raise DataValidationError(
                "binary_classification requires exactly two distinct target values"
            )
        return task
    if task is TaskType.MULTICLASS_CLASSIFICATION:
        if unique_count < 3:
            raise DataValidationError(
                "multiclass_classification requires at least three distinct target values"
            )
        return task
    if task is TaskType.REGRESSION:
        _validated_numeric_target(target)
        return task

    if unique_count == 2:
        return TaskType.BINARY_CLASSIFICATION
    if not is_numeric_dtype(target.dtype):
        return TaskType.MULTICLASS_CLASSIFICATION
    if _looks_discrete_numeric_target(target, unique_count=unique_count):
        return TaskType.MULTICLASS_CLASSIFICATION
    _validated_numeric_target(target)
    return TaskType.REGRESSION


def _looks_discrete_numeric_target(target: pd.Series, *, unique_count: int) -> bool:
    if not is_integer_dtype(target.dtype):
        numeric = _validated_numeric_target(target)
        if not np.allclose(numeric, np.rint(numeric), rtol=0.0, atol=1e-12):
            return False
    threshold = max(20, int(math.sqrt(len(target))))
    return unique_count <= threshold


def _validated_numeric_target(target: pd.Series) -> np.ndarray[Any, np.dtype[np.float64]]:
    try:
        converted = cast(pd.Series, pd.to_numeric(target, errors="raise"))
        numeric = np.asarray(converted, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise DataValidationError("regression target must be numeric") from error
    if not bool(np.isfinite(numeric).all()):
        raise DataValidationError("regression target must contain only finite values")
    return numeric


def _columns_with_role(
    inferred: dict[str, InferredType],
    role: InferredType,
) -> list[str]:
    return [name for name, inferred_type in inferred.items() if inferred_type is role]


def _infinite_count(series: pd.Series) -> int:
    if not is_numeric_dtype(series.dtype):
        return 0
    converted = cast(pd.Series, pd.to_numeric(series, errors="coerce"))
    numeric = np.asarray(converted, dtype=np.float64)
    return int(np.isinf(numeric).sum())


def _finite_skewness(series: pd.Series) -> float:
    converted = cast(pd.Series, pd.to_numeric(series, errors="coerce"))
    numeric = np.asarray(converted, dtype=np.float64)
    finite = numeric[np.isfinite(numeric)]
    if len(finite) < 3 or len(np.unique(finite)) <= 1:
        return 0.0
    skewness = float(cast(Any, pd.Series(finite).skew()))
    return skewness if math.isfinite(skewness) else 0.0


def _dominant_ratio(series: pd.Series) -> float:
    counts = series.value_counts(dropna=False, normalize=True)
    return 0.0 if counts.empty else float(counts.iloc[0])


def _is_id_candidate(
    name: str,
    series: pd.Series,
    *,
    inferred_type: InferredType,
    unique_ratio: float,
    threshold: float,
) -> bool:
    if unique_ratio < threshold:
        return False
    if _ID_NAME_PATTERN.search(name):
        return True
    if inferred_type in {InferredType.CATEGORICAL, InferredType.TEXT}:
        return True
    if is_integer_dtype(series.dtype):
        non_missing = series.dropna()
        return bool(non_missing.is_monotonic_increasing or non_missing.is_monotonic_decreasing)
    return False


def _is_group_candidate(name: str, *, cardinality: int, n_rows: int) -> bool:
    if not _GROUP_NAME_PATTERN.search(name):
        return False
    upper_bound = max(2, min(100, n_rows // 2))
    return 2 <= cardinality <= upper_bound


def _target_profile(target: pd.Series, task: TaskType) -> dict[str, JsonValue]:
    profile: dict[str, JsonValue] = {
        "task": task.value,
        "n_unique": int(target.nunique(dropna=True)),
        "missing_count": int(target.isna().sum()),
        "missing_ratio": float(target.isna().mean()),
    }
    if task in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}:
        counts = target.value_counts(dropna=False)
        labelled_counts = sorted(
            ((str(label), int(count)) for label, count in counts.items()),
            key=lambda item: item[0],
        )
        class_counts = dict(labelled_counts)
        count_values = list(class_counts.values())
        minimum = min(count_values)
        maximum = max(count_values)
        profile.update(
            {
                "class_counts": cast(JsonValue, class_counts),
                "minority_class_fraction": float(minimum / len(target)),
                "majority_class_fraction": float(maximum / len(target)),
                "imbalance_ratio": float(maximum / minimum),
            }
        )
        return profile

    values = _validated_numeric_target(target)
    target_series = pd.Series(values)
    skewness = float(cast(Any, target_series.skew())) if len(values) >= 3 else 0.0
    profile.update(
        {
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=0)),
            "minimum": float(np.min(values)),
            "median": float(np.median(values)),
            "maximum": float(np.max(values)),
            "skewness": skewness if math.isfinite(skewness) else 0.0,
        }
    )
    return profile


def _statistical_landmarks(
    frame: pd.DataFrame,
    target: pd.Series,
    *,
    inferred: dict[str, InferredType],
    missing_ratios: dict[str, float],
    unique_ratios: dict[str, float],
    duplicate_row_count: int,
) -> dict[str, float]:
    n_rows = len(frame)
    n_features = frame.shape[1]
    denominator = float(n_features)
    roles = [inferred_type.primary_role for inferred_type in inferred.values()]
    return {
        "numeric_fraction": float(roles.count(InferredType.NUMERIC) / denominator),
        "categorical_fraction": float(roles.count(InferredType.CATEGORICAL) / denominator),
        "boolean_fraction": float(roles.count(InferredType.BOOLEAN) / denominator),
        "datetime_fraction": float(roles.count(InferredType.DATETIME) / denominator),
        "text_fraction": float(roles.count(InferredType.TEXT) / denominator),
        "missing_fraction": float(sum(missing_ratios.values()) / denominator),
        "mean_unique_ratio": float(sum(unique_ratios.values()) / denominator),
        "duplicate_ratio": float(duplicate_row_count / n_rows),
        "feature_to_row_ratio": float(n_features / n_rows),
        "target_unique_ratio": float(target.nunique(dropna=True) / n_rows),
    }


def _training_dataset_hash(dataset: LoadedDataset) -> str:
    """Aggregate train hashes only; test changes must not influence search."""
    digest = hashlib.sha256()
    for path in dataset.bundle.train_paths:
        source_hash = _source_hash_for_path(path, dataset.bundle.source_hashes)
        digest.update(source_hash.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _source_hash_for_path(path: Path, source_hashes: dict[str, str]) -> str:
    direct = source_hashes.get(str(path))
    if direct is not None:
        return direct
    resolved = path.expanduser().resolve()
    for source, source_hash in source_hashes.items():
        if Path(source).expanduser().resolve() == resolved:
            return source_hash
    raise DataValidationError(f"training source hash is missing for profiling: {path}")


__all__ = ["DatasetProfiler", "profile_dataset"]

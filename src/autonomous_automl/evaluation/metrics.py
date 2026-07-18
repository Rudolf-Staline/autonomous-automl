"""Task-aware metrics with a uniform greater-is-better convention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)

from autonomous_automl.contracts import DatasetProfile, MetricName, TaskType
from autonomous_automl.utils.errors import MetricError

_BINARY_METRICS = {
    MetricName.ROC_AUC,
    MetricName.AVERAGE_PRECISION,
    MetricName.LOG_LOSS,
    MetricName.ACCURACY,
    MetricName.BALANCED_ACCURACY,
    MetricName.F1,
    MetricName.MACRO_F1,
}
_MULTICLASS_METRICS = {
    MetricName.LOG_LOSS,
    MetricName.ACCURACY,
    MetricName.BALANCED_ACCURACY,
    MetricName.MACRO_F1,
}
_REGRESSION_METRICS = {MetricName.RMSE, MetricName.MAE, MetricName.R2}
_LOSS_METRICS = {MetricName.LOG_LOSS, MetricName.RMSE, MetricName.MAE}
_PROBABILITY_METRICS = {
    MetricName.ROC_AUC,
    MetricName.AVERAGE_PRECISION,
    MetricName.LOG_LOSS,
}


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """Resolved scoring policy for one task."""

    name: MetricName
    task: TaskType
    requires_probabilities: bool
    is_loss: bool


@dataclass(frozen=True, slots=True)
class MetricValue:
    """Both optimizer-oriented and user-facing values of one score."""

    internal_value: float
    user_value: float


def resolve_metric(
    metric: MetricName | str,
    task: TaskType,
    profile: DatasetProfile | None = None,
) -> MetricDefinition:
    """Resolve ``auto`` and reject metrics which cannot score the task."""
    if task is TaskType.AUTO:
        raise MetricError("a metric requires a resolved task")
    try:
        requested = MetricName(metric)
    except ValueError as error:
        raise MetricError(f"unknown metric: {metric}") from error

    resolved = _automatic_metric(task, profile) if requested is MetricName.AUTO else requested
    allowed = _allowed_metrics(task)
    if resolved not in allowed:
        raise MetricError(f"metric {resolved.value!r} is incompatible with task {task.value!r}")
    return MetricDefinition(
        name=resolved,
        task=task,
        requires_probabilities=resolved in _PROBABILITY_METRICS,
        is_loss=resolved in _LOSS_METRICS,
    )


def score_metric(
    definition: MetricDefinition,
    y_true: pd.Series | np.ndarray[Any, Any] | list[object],
    *,
    predictions: np.ndarray[Any, Any] | pd.Series | list[object] | None = None,
    probabilities: np.ndarray[Any, Any] | None = None,
    classes: np.ndarray[Any, Any] | list[object] | None = None,
) -> MetricValue:
    """Compute a finite metric and orient losses for maximization."""
    truth = np.asarray(y_true)
    if truth.ndim != 1 or len(truth) == 0:
        raise MetricError("metric target must be a non-empty one-dimensional array")

    try:
        user_value = _score_user_value(
            definition,
            truth,
            predictions=predictions,
            probabilities=probabilities,
            classes=classes,
        )
    except MetricError:
        raise
    except (TypeError, ValueError) as error:
        raise MetricError(
            f"could not compute {definition.name.value!r} for this validation fold"
        ) from error
    if not np.isfinite(user_value):
        raise MetricError(f"metric {definition.name.value!r} produced a non-finite value")
    internal_value = -user_value if definition.is_loss else user_value
    return MetricValue(internal_value=float(internal_value), user_value=float(user_value))


def ordered_classes(y: pd.Series | np.ndarray[Any, Any] | list[object]) -> np.ndarray[Any, Any]:
    """Return deterministic class order even for heterogeneous scalar labels."""
    values = list(pd.unique(np.asarray(y)))
    values.sort(key=lambda value: (type(value).__name__, str(value)))
    return np.asarray(values, dtype=object)


def align_probabilities(
    probabilities: np.ndarray[Any, Any],
    model_classes: np.ndarray[Any, Any] | list[object],
    global_classes: np.ndarray[Any, Any] | list[object],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Align a classifier's probability columns to the run-wide class order."""
    raw = np.asarray(probabilities, dtype=np.float64)
    local = list(model_classes)
    global_order = list(global_classes)
    if raw.ndim != 2 or raw.shape[1] != len(local):
        raise MetricError("classifier probabilities do not match its classes")
    aligned = np.zeros((raw.shape[0], len(global_order)), dtype=np.float64)
    for local_index, label in enumerate(local):
        matches = [index for index, candidate in enumerate(global_order) if candidate == label]
        if not matches:
            raise MetricError("classifier returned a class absent from the training target")
        aligned[:, matches[0]] = raw[:, local_index]
    if not bool(np.isfinite(aligned).all()) or bool((aligned < 0).any()):
        raise MetricError("classifier returned invalid probabilities")
    totals = aligned.sum(axis=1)
    if bool((totals <= 0).any()):
        raise MetricError("classifier returned zero probability mass")
    if not bool(np.allclose(totals, 1.0, rtol=1e-6, atol=1e-8)):
        raise MetricError("classifier probabilities must sum to one on every row")
    return aligned


def _automatic_metric(task: TaskType, profile: DatasetProfile | None) -> MetricName:
    if task is TaskType.BINARY_CLASSIFICATION:
        imbalance_ratio = 1.0
        if profile is not None:
            raw_ratio = profile.target_profile.get("imbalance_ratio", 1.0)
            if isinstance(raw_ratio, int | float):
                imbalance_ratio = float(raw_ratio)
        return MetricName.AVERAGE_PRECISION if imbalance_ratio >= 4.0 else MetricName.ROC_AUC
    if task is TaskType.MULTICLASS_CLASSIFICATION:
        return MetricName.MACRO_F1
    if task is TaskType.REGRESSION:
        return MetricName.RMSE
    raise MetricError("a metric requires a resolved task")


def _allowed_metrics(task: TaskType) -> set[MetricName]:
    if task is TaskType.BINARY_CLASSIFICATION:
        return _BINARY_METRICS
    if task is TaskType.MULTICLASS_CLASSIFICATION:
        return _MULTICLASS_METRICS
    if task is TaskType.REGRESSION:
        return _REGRESSION_METRICS
    return set()


def _score_user_value(
    definition: MetricDefinition,
    truth: np.ndarray[Any, Any],
    *,
    predictions: np.ndarray[Any, Any] | pd.Series | list[object] | None,
    probabilities: np.ndarray[Any, Any] | None,
    classes: np.ndarray[Any, Any] | list[object] | None,
) -> float:
    metric = definition.name
    if definition.task is TaskType.REGRESSION:
        predicted = _required_predictions(predictions)
        if metric is MetricName.RMSE:
            return float(root_mean_squared_error(truth, predicted))
        if metric is MetricName.MAE:
            return float(mean_absolute_error(truth, predicted))
        return float(r2_score(truth, predicted))

    labels = list(classes) if classes is not None else list(ordered_classes(truth))
    if len(labels) < 2:
        raise MetricError("classification metrics require at least two classes")
    if metric in _PROBABILITY_METRICS:
        probability_values = _required_probabilities(probabilities, len(truth), len(labels))
        if metric is MetricName.ROC_AUC:
            positive = labels[-1]
            binary_truth = np.asarray([int(value == positive) for value in truth])
            return float(roc_auc_score(binary_truth, probability_values[:, -1]))
        if metric is MetricName.AVERAGE_PRECISION:
            positive = labels[-1]
            binary_truth = np.asarray([int(value == positive) for value in truth])
            return float(average_precision_score(binary_truth, probability_values[:, -1]))
        return float(log_loss(truth, probability_values, labels=labels))

    predicted = _required_predictions(predictions)
    if metric is MetricName.ACCURACY:
        return float(accuracy_score(truth, predicted))
    if metric is MetricName.BALANCED_ACCURACY:
        return float(balanced_accuracy_score(truth, predicted))
    if metric is MetricName.F1:
        return float(f1_score(truth, predicted, pos_label=cast(Any, labels[-1]), average="binary"))
    if metric is MetricName.MACRO_F1:
        return float(f1_score(truth, predicted, average="macro"))
    raise MetricError(f"metric {metric.value!r} has no scoring implementation")


def _required_predictions(
    values: np.ndarray[Any, Any] | pd.Series | list[object] | None,
) -> np.ndarray[Any, Any]:
    if values is None:
        raise MetricError("this metric requires predictions")
    predictions = np.asarray(values)
    if predictions.ndim != 1:
        predictions = cast(np.ndarray[Any, Any], predictions.reshape(-1))
    return predictions


def _required_probabilities(
    values: np.ndarray[Any, Any] | None,
    n_rows: int,
    n_classes: int,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    if values is None:
        raise MetricError("this metric requires class probabilities")
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.shape != (n_rows, n_classes):
        raise MetricError("class probability matrix has an unexpected shape")
    if not bool(np.isfinite(probabilities).all()) or bool((probabilities < 0).any()):
        raise MetricError("class probabilities must be finite and non-negative")
    if not bool(np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8)):
        raise MetricError("class probabilities must sum to one on every row")
    return probabilities


__all__ = [
    "MetricDefinition",
    "MetricValue",
    "align_probabilities",
    "ordered_classes",
    "resolve_metric",
    "score_metric",
]

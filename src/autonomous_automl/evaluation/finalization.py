"""Deterministic full-data training of a selected ``PipelineSpec``."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from autonomous_automl.components import ModelRegistry
from autonomous_automl.contracts import DatasetProfile, PipelineSpec, TaskType
from autonomous_automl.data import LoadedDataset
from autonomous_automl.evaluation.metrics import align_probabilities, ordered_classes
from autonomous_automl.pipelines import build_pipeline
from autonomous_automl.tracking.artifacts import ArtifactRecord, ArtifactStore
from autonomous_automl.tracking.store import ArtifactRegistration
from autonomous_automl.utils.errors import ArtifactValidationError

_CLASSIFICATION_TASKS = {
    TaskType.BINARY_CLASSIFICATION,
    TaskType.MULTICLASS_CLASSIFICATION,
}


@dataclass(frozen=True, slots=True)
class FinalTrainingResult:
    """Fitted runtime objects and optional source-ordered test predictions."""

    pipeline_spec: PipelineSpec
    pipeline: Pipeline
    fit_seconds: float
    predict_seconds: float
    n_train_rows: int
    n_test_rows: int
    predictions: np.ndarray[Any, Any] | None
    probabilities: np.ndarray[Any, np.dtype[np.float64]] | None
    classes: tuple[object, ...] | None
    probability_columns: tuple[str, ...]
    prediction_frame: pd.DataFrame | None


class FinalTrainer:
    """Rebuild and fit the selected pipeline exactly once on all training rows."""

    def __init__(self, registry: ModelRegistry | None = None, *, n_jobs: int = 1) -> None:
        if n_jobs < 1:
            raise ValueError("n_jobs must be at least one")
        self.registry = registry or ModelRegistry.default()
        self.n_jobs = n_jobs

    def fit(
        self,
        pipeline_spec: PipelineSpec,
        dataset: LoadedDataset,
        profile: DatasetProfile,
        *,
        predict_test: bool = True,
    ) -> FinalTrainingResult:
        """Fit from ``PipelineSpec`` and optionally infer on the isolated test frame."""

        _validate_training_inputs(pipeline_spec, dataset, profile)
        pipeline = build_pipeline(
            pipeline_spec,
            profile,
            n_jobs=self.n_jobs,
            registry=self.registry,
        )

        fit_started = time.monotonic()
        pipeline.fit(dataset.X, dataset.y)
        fit_seconds = time.monotonic() - fit_started

        classes = _training_classes(pipeline_spec.task, dataset.y)
        test_frame = dataset.test_X if predict_test else None
        if test_frame is None:
            return FinalTrainingResult(
                pipeline_spec=pipeline_spec,
                pipeline=pipeline,
                fit_seconds=fit_seconds,
                predict_seconds=0.0,
                n_train_rows=len(dataset.X),
                n_test_rows=0,
                predictions=None,
                probabilities=None,
                classes=classes,
                probability_columns=(),
                prediction_frame=None,
            )

        predict_started = time.monotonic()
        predictions = _validated_predictions(
            pipeline.predict(test_frame),
            expected_rows=len(test_frame),
            task=pipeline_spec.task,
            classes=classes,
        )
        probabilities = _classification_probabilities(
            pipeline,
            test_frame,
            pipeline_spec.task,
            classes,
        )
        predict_seconds = time.monotonic() - predict_started
        probability_columns = (
            tuple(f"probability_{index}" for index in range(probabilities.shape[1]))
            if probabilities is not None
            else ()
        )
        prediction_frame = _prediction_frame(
            predictions,
            probabilities,
            probability_columns,
            dataset.test_row_ids,
        )
        return FinalTrainingResult(
            pipeline_spec=pipeline_spec,
            pipeline=pipeline,
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            n_train_rows=len(dataset.X),
            n_test_rows=len(test_frame),
            predictions=predictions,
            probabilities=probabilities,
            classes=classes,
            probability_columns=probability_columns,
            prediction_frame=prediction_frame,
        )

    @staticmethod
    def save_pipeline(
        result: FinalTrainingResult,
        artifact_store: ArtifactStore,
        *,
        name: str = "best_pipeline",
        relative_path: str | Path | None = None,
        compress: int = 3,
    ) -> ArtifactRecord:
        """Delegate atomic local serialization to the confined artifact service."""

        return artifact_store.write_joblib(
            name,
            result.pipeline,
            relative_path=relative_path,
            compress=compress,
        )

    @staticmethod
    def load_pipeline(
        artifact_store: ArtifactStore,
        registration: ArtifactRegistration,
    ) -> Pipeline:
        """Load only a registry-backed, checksum-verified local artifact."""

        value = artifact_store.load_joblib(registration)
        if not isinstance(value, Pipeline):
            raise ArtifactValidationError("registered best-pipeline artifact is not a Pipeline")
        return value


def _validate_training_inputs(
    pipeline_spec: PipelineSpec,
    dataset: LoadedDataset,
    profile: DatasetProfile,
) -> None:
    if pipeline_spec.task is not profile.inferred_task:
        raise ValueError("pipeline task does not match the profiled task")
    if len(dataset.X) != profile.n_rows:
        raise ValueError("runtime dataset does not match the profile row count")
    if len(dataset.y) != profile.n_rows:
        raise ValueError("training target does not match the profile row count")
    if dataset.bundle.target in dataset.X.columns:
        raise ValueError("the target must never be included in final-training features")
    if dataset.test_row_ids is not None and (
        dataset.test_X is None or len(dataset.test_row_ids) != len(dataset.test_X)
    ):
        raise ValueError("test row IDs must match the optional test frame")


def _training_classes(task: TaskType, target: pd.Series) -> tuple[object, ...] | None:
    if task not in _CLASSIFICATION_TASKS:
        return None
    values = ordered_classes(target)
    if len(values) < 2:
        raise ValueError("classification final training requires at least two classes")
    return tuple(cast(list[object], values.tolist()))


def _validated_predictions(
    raw_predictions: object,
    *,
    expected_rows: int,
    task: TaskType,
    classes: tuple[object, ...] | None,
) -> np.ndarray[Any, Any]:
    predictions = np.asarray(raw_predictions).reshape(-1)
    if len(predictions) != expected_rows:
        raise ValueError("final predictions do not match the test row count")
    if task is TaskType.REGRESSION:
        numeric = np.asarray(predictions, dtype=np.float64)
        if not bool(np.isfinite(numeric).all()):
            raise ValueError("final regression predictions must be finite")
        return numeric
    assert classes is not None
    for prediction in predictions:
        if not any(prediction == known_class for known_class in classes):
            raise ValueError("final classifier returned a class absent from training")
    return predictions


def _classification_probabilities(
    pipeline: Pipeline,
    test_frame: pd.DataFrame,
    task: TaskType,
    classes: tuple[object, ...] | None,
) -> np.ndarray[Any, np.dtype[np.float64]] | None:
    if task not in _CLASSIFICATION_TASKS:
        return None
    assert classes is not None
    predict_proba = getattr(pipeline, "predict_proba", None)
    if not callable(predict_proba):
        raise TypeError("selected classification pipeline does not expose predict_proba")
    model_classes = getattr(pipeline, "classes_", None)
    if model_classes is None:
        raise TypeError("selected classification pipeline does not expose fitted classes")
    raw_probabilities = np.asarray(predict_proba(test_frame), dtype=np.float64)
    return align_probabilities(raw_probabilities, model_classes, list(classes))


def _prediction_frame(
    predictions: np.ndarray[Any, Any],
    probabilities: np.ndarray[Any, np.dtype[np.float64]] | None,
    probability_columns: tuple[str, ...],
    row_ids: pd.Series | None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "row_position": np.arange(len(predictions), dtype=np.int64),
            "prediction": predictions,
        }
    )
    if row_ids is not None:
        frame.insert(1, "row_id", row_ids.reset_index(drop=True).to_numpy(copy=True))
    if probabilities is not None:
        for index, column in enumerate(probability_columns):
            frame[column] = probabilities[:, index]
    return frame


__all__ = ["FinalTrainer", "FinalTrainingResult"]

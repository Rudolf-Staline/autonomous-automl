"""Safe inspection, validation, loading, and inference for completed runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError
from sklearn.pipeline import Pipeline

from autonomous_automl.contracts import RunManifest, RunStatus
from autonomous_automl.tracking import (
    ArtifactRecord,
    ArtifactStore,
    ExperimentStore,
    verify_manifest_source_hashes,
)
from autonomous_automl.utils.errors import ArtifactValidationError, DataValidationError

REGISTRY_FILENAME = "registry.sqlite3"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class ArtifactValidationSummary:
    """Evidence returned after a complete local artifact validation."""

    run_id: str
    artifact_count: int
    sqlite_integrity: bool
    manifest_match: bool
    source_hashes_match: bool
    pipeline_loadable: bool
    predictions_match: bool | None


def open_run(run_directory: str | Path) -> tuple[Path, ExperimentStore, RunManifest]:
    """Open a run and require its on-disk manifest to match the SQLite authority."""

    directory = Path(run_directory).expanduser().resolve()
    if not directory.is_dir():
        raise ArtifactValidationError(f"run directory does not exist: {directory}")
    manifest_path = directory / MANIFEST_FILENAME
    registry_path = directory / REGISTRY_FILENAME
    if not manifest_path.is_file() or not registry_path.is_file():
        raise ArtifactValidationError(
            "run directory must contain manifest.json and registry.sqlite3"
        )
    disk_manifest = RunManifest.read_json(manifest_path)
    store = ExperimentStore.open(registry_path)
    database_manifest = store.get_manifest(disk_manifest.run_id)
    if disk_manifest != database_manifest:
        raise ArtifactValidationError("manifest.json does not match the SQLite run manifest")
    return directory, store, database_manifest


def load_best_pipeline(run_directory: str | Path) -> Pipeline:
    """Checksum-verify and load the locally generated final pipeline."""

    directory, store, manifest = open_run(run_directory)
    if manifest.status is not RunStatus.COMPLETED:
        raise ArtifactValidationError("only a completed run has a final pipeline")
    registration = store.get_artifact(manifest.run_id, "best_pipeline")
    value = ArtifactStore(directory, registry=store, run_id=manifest.run_id).load_joblib(
        registration
    )
    if not isinstance(value, Pipeline):
        raise ArtifactValidationError("best_pipeline is not a scikit-learn Pipeline")
    return value


def validate_run_artifacts(run_directory: str | Path) -> ArtifactValidationSummary:
    """Validate SQLite, source hashes, every artifact, and persisted predictions."""

    directory, store, manifest = open_run(run_directory)
    store.integrity_check()
    verify_manifest_source_hashes(manifest)
    registrations = store.list_artifacts(manifest.run_id)
    artifact_store = ArtifactStore(directory, registry=store, run_id=manifest.run_id)
    for registration in registrations:
        artifact_store.validate(
            ArtifactRecord(
                name=registration.name,
                kind=registration.kind,
                relative_path=registration.relative_path,
                sha256=registration.sha256,
                size_bytes=registration.size_bytes,
                media_type=registration.media_type,
            )
        )

    pipeline_loadable = False
    predictions_match: bool | None = None
    if manifest.status is RunStatus.COMPLETED:
        pipeline = load_best_pipeline(directory)
        pipeline_loadable = True
        if manifest.dataset.test_path is not None and "predictions" in manifest.artifacts:
            test_frame, _ = _read_prediction_frame(
                manifest.dataset.test_path,
                manifest,
            )
            reproduced = np.asarray(pipeline.predict(test_frame)).reshape(-1)
            expected_frame = pd.read_csv(directory / manifest.artifacts["predictions"])
            if "prediction" not in expected_frame:
                raise ArtifactValidationError("predictions artifact has no prediction column")
            expected = expected_frame["prediction"].to_numpy()
            predictions_match = _predictions_equal(reproduced, expected)
            if not predictions_match:
                raise ArtifactValidationError(
                    "predictions do not match the checksum-verified serialized pipeline"
                )

    return ArtifactValidationSummary(
        run_id=manifest.run_id,
        artifact_count=len(registrations),
        sqlite_integrity=True,
        manifest_match=True,
        source_hashes_match=True,
        pipeline_loadable=pipeline_loadable,
        predictions_match=predictions_match,
    )


def predict_csv(
    run_directory: str | Path,
    csv_path: str | Path,
    *,
    output_path: str | Path,
) -> Path:
    """Predict a schema-compatible CSV with a verified completed-run pipeline."""

    directory, _, manifest = open_run(run_directory)
    pipeline = load_best_pipeline(directory)
    features, row_ids = _read_prediction_frame(csv_path, manifest)
    predictions = np.asarray(pipeline.predict(features)).reshape(-1)
    result = pd.DataFrame(
        {
            "row_position": np.arange(len(predictions), dtype=np.int64),
            "prediction": predictions,
        }
    )
    if row_ids is not None:
        result.insert(1, "row_id", row_ids.to_numpy(copy=True))
    predict_proba = getattr(pipeline, "predict_proba", None)
    if callable(predict_proba):
        probabilities = np.asarray(predict_proba(features), dtype=np.float64)
        if probabilities.ndim == 2:
            for index in range(probabilities.shape[1]):
                result[f"probability_{index}"] = probabilities[:, index]

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ArtifactValidationError(f"prediction output already exists: {destination}")
    result.to_csv(destination, index=False, lineterminator="\n")
    return destination


def _read_prediction_frame(
    csv_path: str | Path,
    manifest: RunManifest,
) -> tuple[pd.DataFrame, pd.Series | None]:
    path = Path(csv_path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".csv":
        raise DataValidationError(f"prediction input must be an existing CSV file: {path}")
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, on_bad_lines="error")
    except (UnicodeDecodeError, EmptyDataError, ParserError, OSError) as error:
        raise DataValidationError(f"prediction CSV could not be read: {path}") from error
    if frame.empty:
        raise DataValidationError("prediction CSV contains no rows")
    if manifest.dataset.target in frame:
        raise DataValidationError("prediction CSV must not contain the target column")
    expected = list(manifest.dataset.feature_columns)
    id_column = manifest.configuration.id_column
    allowed_extras = {
        column
        for column in (
            id_column,
            manifest.configuration.group_column,
            manifest.configuration.predefined_fold_column,
        )
        if column is not None
    }
    missing = sorted(set(expected).difference(frame.columns))
    extra = sorted(set(frame.columns).difference(expected).difference(allowed_extras))
    if missing or extra:
        raise DataValidationError(
            f"prediction CSV schema mismatch; missing={missing}, extra={extra}"
        )
    row_ids: pd.Series | None = None
    if id_column is not None:
        row_id_value = frame[id_column]
        if isinstance(row_id_value, pd.DataFrame):
            raise DataValidationError("prediction CSV contains duplicate ID columns")
        row_ids = row_id_value.copy()
    features = cast(pd.DataFrame, frame.loc[:, expected].copy())
    return features, row_ids


def _predictions_equal(left: np.ndarray[Any, Any], right: np.ndarray[Any, Any]) -> bool:
    if left.shape != right.shape:
        return False
    try:
        return bool(
            np.allclose(
                np.asarray(left, dtype=np.float64),
                np.asarray(right, dtype=np.float64),
                rtol=1e-10,
                atol=1e-12,
            )
        )
    except (TypeError, ValueError):
        return bool(np.array_equal(left, right))


__all__ = [
    "ArtifactValidationSummary",
    "load_best_pipeline",
    "open_run",
    "predict_csv",
    "validate_run_artifacts",
]

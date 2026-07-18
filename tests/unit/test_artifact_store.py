"""Atomic artifact persistence and validation tests."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import Mock

import joblib
import pandas as pd
import pytest

from autonomous_automl.contracts import AutoMLConfig
from autonomous_automl.tracking.artifacts import ArtifactRecord, ArtifactStore
from autonomous_automl.tracking.store import ArtifactRegistration
from autonomous_automl.utils.errors import ArtifactValidationError


def test_contract_dataframe_and_joblib_round_trip(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run")
    config = AutoMLConfig(target="target", budget_seconds=60)
    frame = pd.DataFrame({"row_id": [1, 2], "prediction": [0.25, 0.75]})

    config_record = store.write_contract("configuration", config)
    frame_record = store.write_dataframe("predictions", frame)
    model_record = store.write_joblib("best_pipeline", {"model": "locally-created"})
    registration = _registration(model_record)
    registry = Mock()
    registry.get_artifact.return_value = registration
    loading_store = ArtifactStore(store.run_directory, registry=registry, run_id="run-1")

    assert AutoMLConfig.read_json(store.validate(config_record)) == config
    pd.testing.assert_frame_equal(pd.read_csv(store.validate(frame_record)), frame)
    assert loading_store.load_joblib(registration) == {"model": "locally-created"}
    assert all(len(record.sha256) == 64 for record in (config_record, frame_record, model_record))


@pytest.mark.parametrize("path", ["../escape.json", "/tmp/escape.json", "nested/../../x"])
def test_artifact_paths_cannot_escape_run_directory(tmp_path: Path, path: str) -> None:
    store = ArtifactStore(tmp_path / "run")

    with pytest.raises(ArtifactValidationError, match=r"inside|escape"):
        store.write_text("unsafe", "payload", relative_path=path)


def test_symlinked_parent_cannot_escape_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    outside = tmp_path / "outside"
    run_dir.mkdir()
    outside.mkdir()
    try:
        (run_dir / "linked").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    store = ArtifactStore(run_dir)

    with pytest.raises(ArtifactValidationError, match="escape"):
        store.write_text("unsafe", "payload", relative_path="linked/result.txt")

    assert not (outside / "result.txt").exists()


def test_joblib_checksum_is_verified_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore(tmp_path / "run")
    record = store.write_joblib("pipeline", {"safe": True})
    registration = _registration(record)
    registry = Mock()
    registry.get_artifact.return_value = registration
    loading_store = ArtifactStore(store.run_directory, registry=registry, run_id="run-1")
    artifact_path = store.run_directory / record.relative_path
    artifact_path.write_bytes(artifact_path.read_bytes() + b"tampered")
    called = False

    def forbidden_load(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("joblib.load must not run before checksum validation")

    monkeypatch.setattr(joblib, "load", forbidden_load)

    with pytest.raises(ArtifactValidationError, match=r"size|checksum"):
        loading_store.load_joblib(registration)
    assert not called


def test_failed_joblib_write_preserves_existing_artifact_and_removes_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore(tmp_path / "run")
    original = store.write_joblib("pipeline", {"version": 1})
    destination = store.run_directory / original.relative_path
    original_bytes = destination.read_bytes()

    def failing_dump(value: object, filename: str | Path, **kwargs: object) -> None:
        Path(filename).write_bytes(b"partial")
        raise RuntimeError("simulated serialization failure")

    monkeypatch.setattr(joblib, "dump", failing_dump)

    with pytest.raises(RuntimeError, match="serialization failure"):
        store.write_joblib("pipeline", {"version": 2})

    assert destination.read_bytes() == original_bytes
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))


def test_artifact_path_is_immutable_after_first_successful_write(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "run")
    original = store.write_text("report", "version one", relative_path="report.html")

    retry = store.write_text("report", "version one", relative_path="report.html")
    with pytest.raises(ArtifactValidationError, match="immutable"):
        store.write_text("report", "version two", relative_path="report.html")

    assert retry == original
    assert (store.run_directory / "report.html").read_text(encoding="utf-8") == "version one"


def test_registered_joblib_is_hashed_and_loaded_from_same_open_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore(tmp_path / "run")
    record = store.write_joblib("pipeline", {"trusted": True})
    registration = _registration(record)
    registry = Mock()
    registry.get_artifact.return_value = registration
    loading_store = ArtifactStore(store.run_directory, registry=registry, run_id="run-1")
    path = store.run_directory / record.relative_path

    def replace_then_load(handle: object) -> object:
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(b"untrusted replacement")
        os.replace(replacement, path)
        return joblib.numpy_pickle.load(handle)

    monkeypatch.setattr(joblib, "load", replace_then_load)

    assert loading_store.load_joblib(registration) == {"trusted": True}


def _registration(record: ArtifactRecord) -> ArtifactRegistration:
    return ArtifactRegistration(
        artifact_id=1,
        run_id="run-1",
        name=record.name,
        kind=record.kind,
        relative_path=record.relative_path,
        sha256=record.sha256,
        size_bytes=record.size_bytes,
        media_type=record.media_type,
    )

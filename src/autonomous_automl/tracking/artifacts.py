"""Atomic, confined, checksum-verified local artifact storage."""

from __future__ import annotations

import hashlib
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, BinaryIO

import joblib
import pandas as pd

from autonomous_automl.contracts import ContractModel
from autonomous_automl.tracking.store import ArtifactRegistration
from autonomous_automl.utils.atomic import atomic_write_text
from autonomous_automl.utils.errors import ArtifactValidationError, TrackingError
from autonomous_automl.utils.hashing import sha256_file

if TYPE_CHECKING:
    from autonomous_automl.tracking.store import ExperimentStore


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Integrity metadata returned only after a durable local write."""

    name: str
    kind: str
    relative_path: str
    sha256: str
    size_bytes: int
    media_type: str | None = None


class ArtifactStore:
    """Write artifacts below one run directory and validate before loading."""

    def __init__(
        self,
        run_directory: str | Path,
        *,
        registry: ExperimentStore | None = None,
        run_id: str | None = None,
    ) -> None:
        self.run_directory = Path(run_directory).expanduser().resolve()
        self.run_directory.mkdir(parents=True, exist_ok=True)
        if not self.run_directory.is_dir():
            raise ArtifactValidationError("artifact root is not a directory")
        if (registry is None) != (run_id is None):
            raise ValueError("registry and run_id must be configured together")
        self.registry = registry
        self.run_id = run_id

    def write_contract(
        self,
        name: str,
        value: ContractModel,
        *,
        relative_path: str | Path | None = None,
    ) -> ArtifactRecord:
        """Atomically persist a validated contract as canonical JSON."""
        _validate_label(name, "artifact name")
        destination = self._destination(relative_path or f"{name}.json")
        self._write_text_once(destination, value.to_json())
        return self._record(name, "contract", destination, "application/json")

    def write_dataframe(
        self,
        name: str,
        frame: pd.DataFrame,
        *,
        relative_path: str | Path | None = None,
        index: bool = False,
    ) -> ArtifactRecord:
        """Persist a portable CSV atomically; Parquet remains an optional export."""
        _validate_label(name, "artifact name")
        destination = self._destination(relative_path or f"{name}.csv")
        self._write_text_once(destination, frame.to_csv(index=index, lineterminator="\n"))
        return self._record(name, "dataframe", destination, "text/csv")

    def write_text(
        self,
        name: str,
        text: str,
        *,
        relative_path: str | Path,
        kind: str = "text",
        media_type: str = "text/plain",
    ) -> ArtifactRecord:
        """Write deterministic UTF-8 text for reports, logs, and environments."""
        _validate_label(name, "artifact name")
        _validate_label(kind, "artifact kind")
        destination = self._destination(relative_path)
        self._write_text_once(destination, text)
        return self._record(name, kind, destination, media_type)

    def write_joblib(
        self,
        name: str,
        value: object,
        *,
        relative_path: str | Path | None = None,
        compress: int = 3,
    ) -> ArtifactRecord:
        """Atomically serialize a locally created Python object with Joblib."""
        _validate_label(name, "artifact name")
        destination = self._destination(relative_path or f"{name}.joblib")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
            )
            os.close(descriptor)
            temporary_path = Path(temporary_name)
            joblib.dump(value, temporary_path, compress=compress)
            with temporary_path.open("rb") as handle:
                os.fsync(handle.fileno())
            if destination.exists():
                if (
                    destination.is_file()
                    and destination.stat().st_size == temporary_path.stat().st_size
                    and sha256_file(destination) == sha256_file(temporary_path)
                ):
                    temporary_path.unlink()
                    temporary_path = None
                else:
                    raise ArtifactValidationError(
                        "artifact paths are immutable once written with different bytes"
                    )
            else:
                os.replace(temporary_path, destination)
                temporary_path = None
            _fsync_directory(destination.parent)
        except BaseException:
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink()
            raise
        return self._record(
            name,
            "joblib",
            destination,
            "application/x-python-joblib",
        )

    def validate(self, record: ArtifactRecord) -> Path:
        """Return the confined path only after size and SHA-256 match."""
        destination = self._destination(record.relative_path)
        if not destination.is_file():
            raise ArtifactValidationError(f"artifact is missing: {record.name}")
        size = destination.stat().st_size
        if size != record.size_bytes:
            raise ArtifactValidationError(f"artifact size mismatch: {record.name}")
        if sha256_file(destination) != record.sha256:
            raise ArtifactValidationError(f"artifact checksum mismatch: {record.name}")
        return destination

    def load_joblib(self, registration: ArtifactRegistration) -> Any:
        """Load a DB-registered Joblib from the same descriptor used for hashing."""
        if registration.kind != "joblib":
            raise ArtifactValidationError("only joblib artifact records can be loaded")
        if self.registry is None or self.run_id is None:
            raise ArtifactValidationError("loading Joblib requires a registry-backed ArtifactStore")
        if registration.run_id != self.run_id:
            raise ArtifactValidationError("artifact registration belongs to another run")
        try:
            stored = self.registry.get_artifact(self.run_id, registration.name)
        except TrackingError as error:
            raise ArtifactValidationError("artifact is not registered for this run") from error
        if stored != registration:
            raise ArtifactValidationError("artifact registration does not match SQLite")
        handle = self._open_confined(registration.relative_path)
        with handle:
            stat = os.fstat(handle.fileno())
            if stat.st_size != registration.size_bytes:
                raise ArtifactValidationError(f"artifact size mismatch: {registration.name}")
            digest = hashlib.sha256()
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
            if digest.hexdigest() != registration.sha256:
                raise ArtifactValidationError(f"artifact checksum mismatch: {registration.name}")
            handle.seek(0)
            return joblib.load(handle)

    def _write_text_once(self, destination: Path, text: str) -> None:
        if destination.exists():
            if destination.is_file() and destination.read_text(encoding="utf-8") == text:
                return
            raise ArtifactValidationError(
                "artifact paths are immutable once written with different bytes"
            )
        atomic_write_text(destination, text)

    def _open_confined(self, relative_path: str) -> BinaryIO:
        raw = Path(relative_path)
        self._destination(raw)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        directory_flag = getattr(os, "O_DIRECTORY", 0)
        if os.name != "posix":
            descriptor = os.open(self.run_directory / raw, flags | no_follow)
            return os.fdopen(descriptor, "rb")

        root_descriptor = os.open(
            self.run_directory,
            os.O_RDONLY | directory_flag | no_follow,
        )
        current_descriptor = root_descriptor
        try:
            for part in raw.parts[:-1]:
                next_descriptor = os.open(
                    part,
                    os.O_RDONLY | directory_flag | no_follow,
                    dir_fd=current_descriptor,
                )
                if current_descriptor != root_descriptor:
                    os.close(current_descriptor)
                current_descriptor = next_descriptor
            file_descriptor = os.open(
                raw.parts[-1],
                flags | no_follow,
                dir_fd=current_descriptor,
            )
            return os.fdopen(file_descriptor, "rb")
        except OSError as error:
            raise ArtifactValidationError("artifact path cannot be opened safely") from error
        finally:
            if current_descriptor != root_descriptor:
                os.close(current_descriptor)
            os.close(root_descriptor)

    def _destination(self, relative_path: str | Path) -> Path:
        raw = Path(relative_path)
        if raw.is_absolute() or any(part == ".." for part in PurePath(raw).parts):
            raise ArtifactValidationError("artifact path must stay inside the run directory")
        if str(raw) in {"", "."}:
            raise ArtifactValidationError("artifact path must name a file")
        candidate = self.run_directory / raw
        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(self.run_directory):
            raise ArtifactValidationError("artifact path escapes the run directory")
        if resolved == self.run_directory:
            raise ArtifactValidationError("artifact path must name a file")
        return candidate

    def _record(
        self,
        name: str,
        kind: str,
        destination: Path,
        media_type: str | None,
    ) -> ArtifactRecord:
        _validate_label(name, "artifact name")
        _validate_label(kind, "artifact kind")
        if not destination.is_file():
            raise ArtifactValidationError("artifact write did not produce a regular file")
        return ArtifactRecord(
            name=name,
            kind=kind,
            relative_path=destination.relative_to(self.run_directory).as_posix(),
            sha256=sha256_file(destination),
            size_bytes=destination.stat().st_size,
            media_type=media_type,
        )


def _fsync_directory(directory: Path) -> None:
    if os.name != "posix":
        return
    with suppress(OSError):
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _validate_label(value: str, label: str) -> None:
    if not value.strip():
        raise ArtifactValidationError(f"{label} cannot be blank")


__all__ = ["ArtifactRecord", "ArtifactStore"]

"""Atomic persistence helpers for text and JSON artifacts."""

from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from pathlib import Path

from autonomous_automl.utils.json import JsonValue, canonical_json_dumps


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Atomically replace *path* with *text* after flushing it to disk."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
            newline="",
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_path, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink()
        raise

    return destination


def atomic_write_json(path: str | Path, value: JsonValue) -> Path:
    """Atomically write a JSON value using its canonical representation."""

    return atomic_write_text(path, canonical_json_dumps(value))


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory sync so a successful replace survives a crash."""

    if os.name != "posix":
        return
    with suppress(OSError):
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

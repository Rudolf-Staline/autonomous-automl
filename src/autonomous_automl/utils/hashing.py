"""SHA-256 helpers for source data and serializable objects."""

from __future__ import annotations

import hashlib
from pathlib import Path

from autonomous_automl.utils.json import JsonValue, canonical_json_dumps

DEFAULT_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: str | Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Return the SHA-256 hexadecimal digest of a file without loading it all."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: JsonValue) -> str:
    """Return the SHA-256 digest of a canonical JSON value."""

    payload = canonical_json_dumps(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

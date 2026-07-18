"""Tests for deterministic and safe persistence utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from autonomous_automl.utils import (
    atomic_write_json,
    atomic_write_text,
    canonical_json_dumps,
    read_json,
    sha256_file,
    sha256_json,
)


def test_canonical_json_is_sorted_compact_and_unicode_safe() -> None:
    value = {"z": [2, 1], "accent": "é"}

    assert canonical_json_dumps(value) == '{"accent":"é","z":[2,1]}'
    assert (
        sha256_json(value)
        == hashlib.sha256(canonical_json_dumps(value).encode("utf-8")).hexdigest()
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(ValueError, match="JSON"):
        canonical_json_dumps({"value": value})


def test_atomic_json_round_trip(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "manifest.json"
    payload = {"schema_version": 1, "items": [True, None, 3.5]}

    returned_path = atomic_write_json(destination, payload)

    assert returned_path == destination
    assert read_json(destination) == payload
    assert not list(destination.parent.glob("*.tmp"))


def test_atomic_text_and_file_hash(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.txt"

    atomic_write_text(destination, "deterministic\n")

    expected = hashlib.sha256(b"deterministic\n").hexdigest()
    assert sha256_file(destination, chunk_size=2) == expected


def test_file_hash_rejects_invalid_chunk_size(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.txt"
    destination.write_text("content", encoding="utf-8")

    with pytest.raises(ValueError, match="chunk_size"):
        sha256_file(destination, chunk_size=0)


def test_read_json_rejects_non_standard_constants(tmp_path: Path) -> None:
    destination = tmp_path / "invalid.json"
    destination.write_text('{"score": NaN}', encoding="utf-8")

    with pytest.raises(ValueError, match="Non-finite"):
        read_json(destination)

"""Stable seed derivation independent from Python's randomized hash."""

from __future__ import annotations

import hashlib

from autonomous_automl.utils.json import JsonValue, canonical_json_dumps


def derive_seed(base_seed: int, *components: str | int) -> int:
    """Derive a deterministic unsigned 32-bit seed from named components."""
    if not 0 <= base_seed <= 4_294_967_295:
        raise ValueError("base_seed must fit an unsigned 32-bit integer")
    payload: list[JsonValue] = [base_seed, *components]
    digest = hashlib.sha256(canonical_json_dumps(payload).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


__all__ = ["derive_seed"]

"""Strict, deterministic JSON helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import NoReturn

type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


def canonical_json_dumps(value: JsonValue) -> str:
    """Serialize a JSON value deterministically as compact UTF-8 text.

    Non-finite floats are rejected because they are not valid JSON and would
    make manifests and hashes non-portable.
    """

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def read_json(path: str | Path) -> JsonValue:
    """Read strict JSON from *path* and return only typed JSON values."""

    with Path(path).open(encoding="utf-8") as handle:
        value: object = json.load(handle, parse_constant=_reject_constant)
    return _validate_json_value(value)


def _reject_constant(_constant: str) -> NoReturn:
    raise ValueError("Non-finite numeric constants are not valid JSON")


def _validate_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite numbers are not valid JSON")
        return value
    if isinstance(value, list):
        return [_validate_json_value(item) for item in value]
    if isinstance(value, dict):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            result[key] = _validate_json_value(item)
        return result
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}")

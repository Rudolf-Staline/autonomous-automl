"""Shared behavior for validated, persistent contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Self, cast

from pydantic import BaseModel, ConfigDict

from autonomous_automl.utils.atomic import atomic_write_json
from autonomous_automl.utils.json import JsonValue, canonical_json_dumps


class ContractModel(BaseModel):
    """Immutable-by-convention base for strict JSON contracts.

    Persisted contracts reject unknown fields and non-finite numbers. A missing
    version field is accepted by each concrete model through its V1 default, which
    provides the minimal forward migration path for pre-versioned V1 artifacts.
    """

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    def to_json_value(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible representation suitable for persistence."""
        return cast(dict[str, JsonValue], self.model_dump(mode="json"))

    def to_json(self) -> str:
        """Serialize the contract using canonical deterministic JSON."""
        return canonical_json_dumps(self.to_json_value())

    def write_json(self, path: str | Path) -> Path:
        """Persist the contract atomically and return the destination path."""
        return atomic_write_json(path, self.to_json_value())

    @classmethod
    def from_json(cls, payload: str | bytes) -> Self:
        """Validate a contract from a JSON string or bytes payload."""
        return cls.model_validate_json(payload)

    @classmethod
    def read_json(cls, path: str | Path) -> Self:
        """Validate a contract read from a UTF-8 JSON artifact."""
        return cls.from_json(Path(path).read_bytes())

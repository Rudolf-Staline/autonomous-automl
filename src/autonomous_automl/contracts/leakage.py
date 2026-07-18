"""Contracts for explainable leakage findings and actions."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from autonomous_automl.contracts.base import ContractModel
from autonomous_automl.contracts.enums import (
    FindingSeverity,
    LeakageFindingType,
)
from autonomous_automl.utils.json import JsonValue


class LeakageFinding(ContractModel):
    """One evidence-backed risk without any raw user value."""

    finding_type: LeakageFindingType
    severity: FindingSeverity
    confidence: float = Field(ge=0.0, le=1.0)
    column: str | None = None
    evidence_summary: dict[str, JsonValue] = Field(default_factory=dict)
    action: str = Field(min_length=1)


class LeakageReport(ContractModel):
    """Versioned collection of leakage findings and explicit exclusions."""

    schema_version: Literal[1] = 1
    findings: list[LeakageFinding] = Field(default_factory=list)
    excluded_columns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_exclusions(self) -> LeakageReport:
        if len(self.excluded_columns) != len(set(self.excluded_columns)):
            raise ValueError("excluded_columns must not contain duplicates")
        finding_columns = {
            finding.column for finding in self.findings if finding.column is not None
        }
        unknown_exclusions = set(self.excluded_columns).difference(finding_columns)
        if unknown_exclusions:
            raise ValueError(
                f"every excluded column must have a recorded finding: {sorted(unknown_exclusions)}"
            )
        return self

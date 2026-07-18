"""Versioned, evidence-only contracts for the optional Trust Layer."""

from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from autonomous_automl.contracts.base import ContractModel
from autonomous_automl.contracts.enums import MetricName, OptimizationProfile, RunStatus, TaskType
from autonomous_automl.contracts.leakage import LeakageFinding
from autonomous_automl.contracts.pipeline import FidelitySpec, PipelineSpec
from autonomous_automl.contracts.validation import ValidationPlan
from autonomous_automl.utils.json import JsonValue


class TrustCertificateStatus(StrEnum):
    """Deterministic result of the recorded Trust Layer checks."""

    SELF_VERIFIED = "SELF_VERIFIED"
    SELF_VERIFIED_WITH_WARNINGS = "SELF_VERIFIED_WITH_WARNINGS"
    INCOMPLETE = "INCOMPLETE"
    FAILED = "FAILED"


class VerificationStatus(StrEnum):
    """Machine-readable status for one objective verification."""

    PASS = "PASS"
    FAIL = "FAIL"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class TrustGapStatus(StrEnum):
    """Whether a methodologically comparable diagnostic was completed."""

    COMPUTED = "COMPUTED"
    NOT_COMPUTED = "NOT_COMPUTED"


class MetricDirection(StrEnum):
    """User-facing metric direction used by the Trust Gap formula."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class SearchStopReason(StrEnum):
    """Exhaustive set of persisted search termination reasons."""

    BUDGET_EXHAUSTED = "budget_exhausted"
    SEARCH_SPACE_EXHAUSTED = "search_space_exhausted"
    USER_INTERRUPTED = "user_interrupted"
    CONTROLLER_STOPPED = "controller_stopped"
    FATAL_ERROR = "fatal_error"


class SourceHashEvidence(ContractModel):
    """A source digest without an absolute or user-specific filesystem path."""

    role: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("filename")
    @classmethod
    def _require_basename(cls, value: str) -> str:
        if value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("source hash evidence must contain a filename, not a path")
        return value


class VerificationEvidence(ContractModel):
    """Outcome and factual detail for one objective check."""

    status: VerificationStatus
    detail: str = Field(min_length=1)
    checked_artifacts: int | None = Field(default=None, ge=0)


class ResumeEvidence(ContractModel):
    """Persisted resume usage without inferring process history."""

    operation_count: int | None = Field(default=None, ge=0)
    resumed: bool | None = None
    interrupted_trials: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_known_resume_state(self) -> ResumeEvidence:
        if self.operation_count is None and self.resumed is not None:
            raise ValueError("resumed must be unavailable when operation_count is unavailable")
        if self.operation_count is not None and self.resumed != (self.operation_count > 0):
            raise ValueError("resumed must be derived from operation_count")
        return self


class RuntimeTelemetry(ContractModel):
    """Observed timing and budget evidence, separate from scheduler behavior."""

    schema_version: Literal[1] = 1
    configured_search_budget_seconds: float = Field(gt=0)
    search_started_at: datetime | None = None
    search_finished_at: datetime | None = None
    search_elapsed_seconds: float | None = Field(default=None, ge=0)
    finalization_elapsed_seconds: float | None = Field(default=None, ge=0)
    total_runtime_seconds: float | None = Field(default=None, ge=0)
    budget_remaining_at_search_stop_seconds: float | None = Field(default=None, ge=0)
    budget_overshoot_seconds: float | None = Field(default=None, ge=0)
    trials_started: int = Field(ge=0)
    trials_completed: int = Field(ge=0)
    trials_failed: int = Field(ge=0)
    stop_reason: SearchStopReason

    @model_validator(mode="after")
    def _validate_timings(self) -> RuntimeTelemetry:
        if (self.search_started_at is None) != (self.search_finished_at is None):
            raise ValueError("search timestamps must both be available or unavailable")
        if (
            self.search_started_at is not None
            and self.search_finished_at is not None
            and self.search_finished_at < self.search_started_at
        ):
            raise ValueError("search_finished_at cannot precede search_started_at")
        if self.trials_completed + self.trials_failed > self.trials_started:
            raise ValueError("completed and failed trials cannot exceed started trials")
        elapsed_values = (
            self.search_elapsed_seconds,
            self.finalization_elapsed_seconds,
            self.total_runtime_seconds,
        )
        if any(value is None for value in elapsed_values) and any(
            value is not None for value in elapsed_values
        ):
            raise ValueError("search, finalization, and total durations are one evidence group")
        if self.search_elapsed_seconds is not None:
            assert self.finalization_elapsed_seconds is not None
            assert self.total_runtime_seconds is not None
            expected_total = self.search_elapsed_seconds + self.finalization_elapsed_seconds
            if not math.isclose(self.total_runtime_seconds, expected_total, abs_tol=1e-6):
                raise ValueError("total runtime must equal search plus finalization")
            expected_overshoot = max(
                0.0,
                self.search_elapsed_seconds - self.configured_search_budget_seconds,
            )
            if self.budget_overshoot_seconds is None or not math.isclose(
                self.budget_overshoot_seconds,
                expected_overshoot,
                abs_tol=1e-6,
            ):
                raise ValueError("budget overshoot must be derived from observed search time")
        elif self.budget_overshoot_seconds is not None:
            raise ValueError("budget overshoot requires an observed search duration")
        if (
            self.budget_remaining_at_search_stop_seconds is not None
            and self.budget_remaining_at_search_stop_seconds > self.configured_search_budget_seconds
        ):
            raise ValueError("remaining budget cannot exceed configured search budget")
        return self


class SelectionContextItem(ContractModel):
    """One supporting fact and the persisted object from which it came."""

    label: str = Field(min_length=1)
    value: JsonValue
    provenance: str = Field(min_length=1)


class PipelineSelectionExplanation(ContractModel):
    """Factual explanation of the already-completed deterministic selection."""

    schema_version: Literal[1] = 1
    title: Literal["Why this pipeline won"] = "Why this pipeline won"
    selected_trial_id: str = Field(min_length=1)
    optimization_profile: OptimizationProfile
    eligible_trial_count: int = Field(gt=0)
    selection_reason: str = Field(min_length=1)
    selection_rule: str = Field(min_length=1)
    tie_breaker_used: bool
    tie_breaker_detail: str | None = None
    runner_up_trial_id: str | None = None
    supporting_context: list[SelectionContextItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_tie_breaker(self) -> PipelineSelectionExplanation:
        if self.tie_breaker_used != (self.tie_breaker_detail is not None):
            raise ValueError("tie_breaker_detail must exist exactly when a tie-breaker was used")
        if self.runner_up_trial_id == self.selected_trial_id:
            raise ValueError("runner-up must differ from the selected trial")
        return self


class TrustGapProtocol(ContractModel):
    """Fully reconstructible raw diagnostic protocol."""

    schema_version: Literal[1] = 1
    protocol_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    metric: MetricName
    metric_direction: MetricDirection
    pipeline_spec: PipelineSpec
    fidelity: FidelitySpec
    validation_plan: ValidationPlan
    feature_columns: list[str]
    excluded_columns: list[str]
    restored_columns: list[str]
    execution_seeds: list[int] = Field(min_length=1)
    timeout_seconds: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_columns(self) -> TrustGapProtocol:
        for field_name in ("feature_columns", "excluded_columns", "restored_columns"):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicates")
        if not set(self.restored_columns).issubset(self.feature_columns):
            raise ValueError("restored columns must be present in the raw feature set")
        if set(self.restored_columns).intersection(self.excluded_columns):
            raise ValueError("restored columns cannot remain excluded")
        expected_seed_count = self.fidelity.n_folds * len(self.fidelity.seeds)
        if len(self.execution_seeds) != expected_seed_count:
            raise ValueError("execution seeds must match every fidelity-seed/fold evaluation")
        if any(seed < 0 or seed > 4_294_967_295 for seed in self.execution_seeds):
            raise ValueError("execution seeds must fit an unsigned 32-bit integer")
        return self


class ObservedTrustGap(ContractModel):
    """Post-selection diagnostic comparison, never a selection objective."""

    schema_version: Literal[1] = 1
    label: Literal["Observed Trust Gap"] = "Observed Trust Gap"
    interpretation: Literal["Apparent score inflation under the raw protocol"] = (
        "Apparent score inflation under the raw protocol"
    )
    status: TrustGapStatus
    reason: str = Field(min_length=1)
    metric: MetricName
    metric_direction: MetricDirection
    raw_score: float | None = None
    verified_score: float | None = None
    observed_trust_gap: float | None = None
    raw_protocol: str = Field(min_length=1)
    verified_protocol: str = Field(min_length=1)
    differing_columns: list[str] = Field(default_factory=list)
    splitter_different: bool | None = None
    warnings: list[str] = Field(default_factory=list)
    diagnostic_elapsed_seconds: float = Field(default=0.0, ge=0)
    raw_protocol_path: str | None = None
    raw_predictions_path: str | None = None

    @field_validator("raw_protocol_path", "raw_predictions_path")
    @classmethod
    def _require_relative_artifact_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parts = value.split("/")
        if value.startswith("/") or "\\" in value or any(part in {"", ".", ".."} for part in parts):
            raise ValueError("Trust Gap artifact paths must be relative and confined")
        return value

    @model_validator(mode="after")
    def _validate_gap(self) -> ObservedTrustGap:
        score_values = (self.raw_score, self.verified_score, self.observed_trust_gap)
        if self.status is TrustGapStatus.NOT_COMPUTED:
            if any(value is not None for value in score_values):
                raise ValueError("NOT_COMPUTED Trust Gap cannot contain scores")
            return self
        if any(value is None for value in score_values):
            raise ValueError("computed Trust Gap requires raw, verified, and gap scores")
        assert self.raw_score is not None
        assert self.verified_score is not None
        assert self.observed_trust_gap is not None
        expected = (
            self.raw_score - self.verified_score
            if self.metric_direction is MetricDirection.MAXIMIZE
            else self.verified_score - self.raw_score
        )
        if not math.isclose(self.observed_trust_gap, expected, abs_tol=1e-12):
            raise ValueError("Observed Trust Gap does not match the metric-direction formula")
        if self.raw_protocol_path is None or self.raw_predictions_path is None:
            raise ValueError("computed Trust Gap requires protocol and prediction artifacts")
        return self


class TrustCertificate(ContractModel):
    """Self-verified evidence compiled solely from one persisted run."""

    schema_version: Literal[1] = 1
    certificate_type: Literal["self_verified"] = "self_verified"
    display_name: Literal["Self-verified trust certificate"] = "Self-verified trust certificate"
    status: TrustCertificateStatus
    run_id: str = Field(min_length=1)
    run_status: RunStatus
    generated_at: datetime
    package_version: str = Field(min_length=1)
    git_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    source_hashes: list[SourceHashEvidence]
    task: TaskType
    metric: MetricName | None = None
    metric_direction: MetricDirection | None = None
    verified_score: float | None = None
    validation_strategy: str = Field(min_length=1)
    n_folds: int = Field(ge=2)
    used_columns: list[str]
    excluded_columns: list[str]
    leakage_findings: list[LeakageFinding]
    unresolved_critical_findings: list[LeakageFinding]
    final_pipeline: PipelineSpec | None = None
    selected_model: str | None = None
    trials_completed: int = Field(ge=0)
    trials_failed: int = Field(ge=0)
    runtime_telemetry: RuntimeTelemetry | None = None
    artifact_validation: VerificationEvidence
    prediction_replay: VerificationEvidence
    resume: ResumeEvidence
    selection_explanation: PipelineSelectionExplanation | None = None
    observed_trust_gap: ObservedTrustGap | None = None
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    disclaimer: Literal[
        "This is a self-verified engineering artifact generated from the recorded run. "
        "It is not an external, regulatory, or security certification."
    ] = (
        "This is a self-verified engineering artifact generated from the recorded run. "
        "It is not an external, regulatory, or security certification."
    )

    @model_validator(mode="after")
    def _validate_derived_fields(self) -> TrustCertificate:
        if (self.metric is None) != (self.metric_direction is None):
            raise ValueError("metric and metric_direction must be available together")
        if (self.final_pipeline is None) != (self.selected_model is None):
            raise ValueError("selected_model and final_pipeline must be available together")
        if (
            self.final_pipeline is not None
            and self.selected_model != self.final_pipeline.model_name
        ):
            raise ValueError("selected_model must come from final_pipeline")
        expected = derive_certificate_status(
            run_status=self.run_status,
            final_pipeline_present=self.final_pipeline is not None,
            artifact_status=self.artifact_validation.status,
            replay_status=self.prediction_replay.status,
            unresolved_critical_count=len(self.unresolved_critical_findings),
            has_warnings=bool(self.warnings or self.leakage_findings),
        )
        if self.status is not expected:
            raise ValueError("certificate status is inconsistent with its recorded evidence")
        return self


def derive_certificate_status(
    *,
    run_status: RunStatus,
    final_pipeline_present: bool,
    artifact_status: VerificationStatus,
    replay_status: VerificationStatus,
    unresolved_critical_count: int,
    has_warnings: bool,
) -> TrustCertificateStatus:
    """Return the only valid certificate status for the supplied evidence."""

    if (
        run_status is RunStatus.FAILED
        or artifact_status is VerificationStatus.FAIL
        or replay_status is VerificationStatus.FAIL
    ):
        return TrustCertificateStatus.FAILED
    if (
        run_status is not RunStatus.COMPLETED
        or not final_pipeline_present
        or artifact_status is not VerificationStatus.PASS
        or replay_status is not VerificationStatus.PASS
        or unresolved_critical_count > 0
    ):
        return TrustCertificateStatus.INCOMPLETE
    if has_warnings:
        return TrustCertificateStatus.SELF_VERIFIED_WITH_WARNINGS
    return TrustCertificateStatus.SELF_VERIFIED


__all__ = [
    "MetricDirection",
    "ObservedTrustGap",
    "PipelineSelectionExplanation",
    "ResumeEvidence",
    "RuntimeTelemetry",
    "SearchStopReason",
    "SelectionContextItem",
    "SourceHashEvidence",
    "TrustCertificate",
    "TrustCertificateStatus",
    "TrustGapProtocol",
    "TrustGapStatus",
    "VerificationEvidence",
    "VerificationStatus",
    "derive_certificate_status",
]

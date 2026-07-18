"""Public creation and revalidation of self-verified trust artifacts."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from autonomous_automl.api.artifacts import open_run, validate_run_artifacts
from autonomous_automl.contracts import (
    ContractModel,
    ObservedTrustGap,
    PipelineSelectionExplanation,
    ResumeEvidence,
    RuntimeTelemetry,
    TrialStatus,
    TrustCertificate,
    VerificationEvidence,
    VerificationStatus,
)
from autonomous_automl.evaluation import explain_pipeline_selection
from autonomous_automl.evaluation.selection import build_leaderboard, select_finalist
from autonomous_automl.pipelines import pipeline_fingerprint
from autonomous_automl.reporting import compile_trust_certificate, render_trust_certificate
from autonomous_automl.runtime import runtime_from_scheduler_state
from autonomous_automl.tracking import ArtifactRecord, ArtifactStore, ExperimentStore
from autonomous_automl.utils.errors import ArtifactValidationError, TrackingError

TRUST_CERTIFICATE_JSON_PATH = "trust_certificate.json"
TRUST_CERTIFICATE_HTML_PATH = "trust_certificate.html"
TRUST_CERTIFICATE_JSON_NAME = "trust_certificate_json"
TRUST_CERTIFICATE_HTML_NAME = "trust_certificate_html"
_SELF_REFERENTIAL_ARTIFACTS = {
    TRUST_CERTIFICATE_JSON_NAME,
    TRUST_CERTIFICATE_HTML_NAME,
    "manifest",
    "report",
}


@dataclass(frozen=True, slots=True)
class TrustArtifactResult:
    """Recalculated evidence and durable certificate locations."""

    certificate: TrustCertificate
    json_path: Path
    html_path: Path
    persisted: bool


def create_or_validate_trust_artifacts(
    run_directory: str | Path,
    *,
    persist_if_missing: bool = True,
    git_commit: str | None = None,
) -> TrustArtifactResult:
    """Recalculate certificate status and atomically create missing artifacts."""

    directory, store, manifest = open_run(run_directory)
    trials = store.list_trials(manifest.run_id)
    contract_errors: list[str] = []
    try:
        existing = _load_existing_certificate(directory, store, manifest.run_id)
    except (ArtifactValidationError, OSError, UnicodeError, ValueError):
        # A damaged previously emitted certificate is itself deterministic FAILED
        # evidence. Rebuild from the other persisted artifacts so the caller receives
        # a status rather than an opaque pre-validation exception.
        existing = None
        contract_errors.append(TRUST_CERTIFICATE_JSON_NAME)
    resolved_commit = (
        existing.git_commit
        if existing is not None
        else git_commit
        if git_commit is not None
        else discover_git_commit()
    )
    generated_at = (
        existing.generated_at
        if existing is not None
        else manifest.completed_at or manifest.updated_at
    )
    artifact_evidence, replay_evidence = _validation_evidence(directory, store, manifest.run_id)
    runtime, runtime_error = _load_optional_contract_safely(
        directory,
        store,
        manifest.run_id,
        "runtime_telemetry",
        RuntimeTelemetry,
    )
    if runtime_error:
        contract_errors.append("runtime_telemetry")
    runtime = runtime or runtime_from_scheduler_state(manifest.scheduler_state)
    explanation, explanation_error = _load_optional_contract_safely(
        directory,
        store,
        manifest.run_id,
        "selection_explanation",
        PipelineSelectionExplanation,
    )
    if explanation_error:
        contract_errors.append("selection_explanation")
    if explanation is None:
        explanation = _derive_selection_explanation(store, manifest.run_id)
    trust_gap, trust_gap_error = _load_optional_contract_safely(
        directory,
        store,
        manifest.run_id,
        "trust_gap",
        ObservedTrustGap,
    )
    if trust_gap_error:
        contract_errors.append("trust_gap")
    has_json = _has_artifact(store, manifest.run_id, TRUST_CERTIFICATE_JSON_NAME)
    has_html = _has_artifact(store, manifest.run_id, TRUST_CERTIFICATE_HTML_NAME)
    if has_json != has_html:
        contract_errors.append("trust_certificate_artifact_pair")
    if contract_errors:
        artifact_evidence = VerificationEvidence(
            status=VerificationStatus.FAIL,
            detail=(
                "Registered Trust Layer contract validation failed: "
                + ", ".join(sorted(set(contract_errors)))
                + "."
            ),
            checked_artifacts=artifact_evidence.checked_artifacts,
        )
    resume_count = store.resume_operation_count(manifest.run_id)
    certificate = compile_trust_certificate(
        manifest,
        trials,
        generated_at=generated_at,
        git_commit=resolved_commit,
        artifact_validation=artifact_evidence,
        prediction_replay=replay_evidence,
        resume=ResumeEvidence(
            operation_count=resume_count,
            resumed=resume_count > 0,
            interrupted_trials=sum(trial.status is TrialStatus.INTERRUPTED for trial in trials),
        ),
        runtime_telemetry=runtime,
        selection_explanation=explanation,
        observed_trust_gap=trust_gap,
    )

    json_path = directory / TRUST_CERTIFICATE_JSON_PATH
    html_path = directory / TRUST_CERTIFICATE_HTML_PATH
    persisted = has_json and has_html
    if not persisted and not (has_json or has_html) and persist_if_missing:
        artifact_store = ArtifactStore(directory, registry=store, run_id=manifest.run_id)
        json_record = artifact_store.write_contract(
            TRUST_CERTIFICATE_JSON_NAME,
            certificate,
            relative_path=TRUST_CERTIFICATE_JSON_PATH,
        )
        _register_record(store, manifest.run_id, json_record)
        html_record = artifact_store.write_text(
            TRUST_CERTIFICATE_HTML_NAME,
            render_trust_certificate(certificate),
            relative_path=TRUST_CERTIFICATE_HTML_PATH,
            kind="trust_certificate",
            media_type="text/html",
        )
        _register_record(store, manifest.run_id, html_record)
        persisted = True
    return TrustArtifactResult(
        certificate=certificate,
        json_path=json_path,
        html_path=html_path,
        persisted=persisted,
    )


def discover_git_commit() -> str | None:
    """Return a clean source checkout's HEAD, otherwise mark it unavailable."""

    root = _git_root()
    if root is None:
        return None
    try:
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if status.stdout.strip():
            return None
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return revision if re.fullmatch(r"[0-9a-f]{40}", revision) else None


def _git_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    return None


def _validation_evidence(
    directory: Path,
    store: ExperimentStore,
    run_id: str,
) -> tuple[VerificationEvidence, VerificationEvidence]:
    stable_count = sum(
        registration.name not in _SELF_REFERENTIAL_ARTIFACTS
        for registration in store.list_artifacts(run_id)
    )
    try:
        summary = validate_run_artifacts(directory)
    except Exception as error:
        replay_status = (
            VerificationStatus.FAIL
            if "predictions do not match" in str(error)
            else VerificationStatus.NOT_AVAILABLE
        )
        replay_detail = (
            "Persisted predictions diverged from the checksum-verified final pipeline."
            if replay_status is VerificationStatus.FAIL
            else "Prediction replay could not be completed after an essential validation failure."
        )
        return (
            VerificationEvidence(
                status=VerificationStatus.FAIL,
                detail=f"Essential artifact validation failed ({type(error).__name__}).",
                checked_artifacts=stable_count,
            ),
            VerificationEvidence(status=replay_status, detail=replay_detail),
        )
    artifact = VerificationEvidence(
        status=VerificationStatus.PASS,
        detail=(
            "SQLite integrity, manifest agreement, source hashes, registered artifact hashes, "
            "and final pipeline loading passed."
        ),
        checked_artifacts=stable_count,
    )
    if summary.predictions_match is True:
        replay = VerificationEvidence(
            status=VerificationStatus.PASS,
            detail="Persisted predictions match replay from the checksum-verified final pipeline.",
        )
    else:
        replay = VerificationEvidence(
            status=VerificationStatus.NOT_AVAILABLE,
            detail="No target-free prediction artifact was recorded for a replay comparison.",
        )
    return artifact, replay


def _derive_selection_explanation(
    store: ExperimentStore,
    run_id: str,
) -> PipelineSelectionExplanation | None:
    manifest = store.get_manifest(run_id)
    if manifest.best_pipeline is None:
        return None
    trials = store.list_trials(run_id)
    leaderboard = build_leaderboard(trials)
    if not leaderboard.entries:
        return None
    try:
        selection = select_finalist(
            leaderboard,
            manifest.configuration.optimization_profile,
            require_full_fidelity=bool(leaderboard.finalists),
        )
        if pipeline_fingerprint(selection.selected.pipeline_spec) != pipeline_fingerprint(
            manifest.best_pipeline
        ):
            return None
        size = store.get_artifact(run_id, "best_pipeline").size_bytes
        return explain_pipeline_selection(
            leaderboard,
            trials,
            manifest.configuration.optimization_profile,
            selected_trial_id=selection.selected.trial_id,
            pipeline_size_bytes=size,
        )
    except (TrackingError, ValueError):
        return None


def _load_existing_certificate(
    directory: Path,
    store: ExperimentStore,
    run_id: str,
) -> TrustCertificate | None:
    return _load_optional_contract(
        directory,
        store,
        run_id,
        TRUST_CERTIFICATE_JSON_NAME,
        TrustCertificate,
    )


def _load_optional_contract[ContractT: ContractModel](
    directory: Path,
    store: ExperimentStore,
    run_id: str,
    name: str,
    contract_type: type[ContractT],
) -> ContractT | None:
    try:
        registration = store.get_artifact(run_id, name)
    except TrackingError:
        return None
    record = ArtifactRecord(
        name=registration.name,
        kind=registration.kind,
        relative_path=registration.relative_path,
        sha256=registration.sha256,
        size_bytes=registration.size_bytes,
        media_type=registration.media_type,
    )
    path = ArtifactStore(directory).validate(record)
    return contract_type.read_json(path)


def _load_optional_contract_safely[ContractT: ContractModel](
    directory: Path,
    store: ExperimentStore,
    run_id: str,
    name: str,
    contract_type: type[ContractT],
) -> tuple[ContractT | None, bool]:
    """Load optional evidence and convert corruption into deterministic FAILED input."""

    try:
        return _load_optional_contract(directory, store, run_id, name, contract_type), False
    except (ArtifactValidationError, OSError, UnicodeError, ValueError):
        return None, True


def _has_artifact(store: ExperimentStore, run_id: str, name: str) -> bool:
    try:
        store.get_artifact(run_id, name)
    except TrackingError:
        return False
    return True


def _register_record(store: ExperimentStore, run_id: str, record: ArtifactRecord) -> None:
    store.register_artifact(
        run_id,
        name=record.name,
        kind=record.kind,
        relative_path=record.relative_path,
        sha256=record.sha256,
        size_bytes=record.size_bytes,
        media_type=record.media_type,
    )


__all__ = [
    "TRUST_CERTIFICATE_HTML_NAME",
    "TRUST_CERTIFICATE_HTML_PATH",
    "TRUST_CERTIFICATE_JSON_NAME",
    "TRUST_CERTIFICATE_JSON_PATH",
    "TrustArtifactResult",
    "create_or_validate_trust_artifacts",
    "discover_git_commit",
]

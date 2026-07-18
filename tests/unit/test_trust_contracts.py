"""Deterministic Trust Layer contract and rendering tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from autonomous_automl.contracts import (
    MetricDirection,
    ObservedTrustGap,
    PipelineSpec,
    ResumeEvidence,
    RunStatus,
    RuntimeTelemetry,
    SearchStopReason,
    SourceHashEvidence,
    TrustCertificate,
    TrustCertificateStatus,
    TrustGapStatus,
    VerificationEvidence,
    VerificationStatus,
    derive_certificate_status,
)
from autonomous_automl.reporting import render_trust_certificate

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


def _pipeline() -> PipelineSpec:
    return PipelineSpec(
        family="linear",
        task="binary_classification",
        numeric_imputer="median",
        numeric_scaler="standard",
        categorical_imputer="constant",
        categorical_encoder="one_hot",
        datetime_transformer="drop",
        model_name="logistic_regression",
        model_params={"C": 1.0},
        excluded_columns=["target_copy"],
        random_seed=42,
    )


def _verification(
    status: VerificationStatus, detail: str = "recorded check"
) -> VerificationEvidence:
    return VerificationEvidence(status=status, detail=detail)


def _certificate(
    *,
    run_status: RunStatus = RunStatus.COMPLETED,
    artifact_status: VerificationStatus = VerificationStatus.PASS,
    replay_status: VerificationStatus = VerificationStatus.PASS,
    warnings: list[str] | None = None,
    final_pipeline: bool = True,
    runtime: RuntimeTelemetry | None = None,
) -> TrustCertificate:
    warning_values = [] if warnings is None else warnings
    pipeline = _pipeline() if final_pipeline else None
    status = derive_certificate_status(
        run_status=run_status,
        final_pipeline_present=final_pipeline,
        artifact_status=artifact_status,
        replay_status=replay_status,
        unresolved_critical_count=0,
        has_warnings=bool(warning_values),
    )
    return TrustCertificate(
        status=status,
        run_id="run-trust-contract",
        run_status=run_status,
        generated_at=NOW,
        package_version="0.1.0",
        source_hashes=[SourceHashEvidence(role="train_1", filename="train.csv", sha256="a" * 64)],
        task="binary_classification",
        metric="roc_auc",
        metric_direction="maximize",
        verified_score=0.81,
        validation_strategy="StratifiedKFold",
        n_folds=3,
        used_columns=["signal"],
        excluded_columns=["target_copy"],
        leakage_findings=[],
        unresolved_critical_findings=[],
        final_pipeline=pipeline,
        selected_model=None if pipeline is None else pipeline.model_name,
        trials_completed=4,
        trials_failed=1,
        runtime_telemetry=runtime,
        artifact_validation=_verification(artifact_status),
        prediction_replay=_verification(replay_status),
        resume=ResumeEvidence(operation_count=0, resumed=False),
        warnings=warning_values,
        limitations=["Recorded validation evidence is not a production guarantee."],
    )


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, TrustCertificateStatus.SELF_VERIFIED),
        ({"has_warnings": True}, TrustCertificateStatus.SELF_VERIFIED_WITH_WARNINGS),
        (
            {"replay_status": VerificationStatus.NOT_AVAILABLE},
            TrustCertificateStatus.INCOMPLETE,
        ),
        ({"final_pipeline_present": False}, TrustCertificateStatus.INCOMPLETE),
        ({"unresolved_critical_count": 1}, TrustCertificateStatus.INCOMPLETE),
        ({"artifact_status": VerificationStatus.FAIL}, TrustCertificateStatus.FAILED),
        ({"replay_status": VerificationStatus.FAIL}, TrustCertificateStatus.FAILED),
        ({"run_status": RunStatus.FAILED}, TrustCertificateStatus.FAILED),
    ],
)
def test_certificate_status_is_derived_only_from_recorded_evidence(
    values: dict[str, object],
    expected: TrustCertificateStatus,
) -> None:
    inputs: dict[str, object] = {
        "run_status": RunStatus.COMPLETED,
        "final_pipeline_present": True,
        "artifact_status": VerificationStatus.PASS,
        "replay_status": VerificationStatus.PASS,
        "unresolved_critical_count": 0,
        "has_warnings": False,
    }
    inputs.update(values)

    assert derive_certificate_status(**inputs) is expected  # type: ignore[arg-type]


def test_certificate_contract_rejects_a_freely_assigned_status() -> None:
    values = _certificate().to_json_value()
    values["status"] = TrustCertificateStatus.FAILED.value

    with pytest.raises(ValidationError, match="inconsistent"):
        TrustCertificate.model_validate(values)


def test_certificate_json_and_html_are_deterministic_and_self_describing() -> None:
    certificate = _certificate(warnings=["A recorded non-critical warning exists."])

    first_json = certificate.to_json()
    second_json = certificate.to_json()
    first_html = render_trust_certificate(certificate)
    second_html = render_trust_certificate(certificate)

    assert first_json == second_json
    assert first_html == second_html
    assert '"certificate_type":"self_verified"' in first_json
    assert "Self-verified trust certificate" in first_html
    assert "not an external, regulatory, or security certification" in first_html


@pytest.mark.parametrize("filename", ["/home/user/train.csv", "../train.csv", r"C:\\train.csv"])
def test_source_hash_evidence_cannot_contain_an_absolute_or_parent_path(filename: str) -> None:
    with pytest.raises(ValidationError, match="filename, not a path"):
        SourceHashEvidence(role="train", filename=filename, sha256="a" * 64)


@pytest.mark.parametrize("path", ["/tmp/raw.json", "../raw.json", r"C:\\raw.json"])
def test_trust_gap_artifact_paths_must_be_relative_and_confined(path: str) -> None:
    with pytest.raises(ValidationError, match="relative and confined"):
        ObservedTrustGap(
            status="COMPUTED",
            reason="diagnostic completed",
            metric="roc_auc",
            metric_direction="maximize",
            raw_score=0.9,
            verified_score=0.8,
            observed_trust_gap=0.1,
            raw_protocol="raw",
            verified_protocol="verified",
            raw_protocol_path=path,
            raw_predictions_path="trust_gap/predictions.csv",
        )


@pytest.mark.parametrize(
    ("direction", "raw", "verified", "expected"),
    [
        (MetricDirection.MAXIMIZE, 0.95, 0.80, 0.15),
        (MetricDirection.MAXIMIZE, 0.70, 0.80, -0.10),
        (MetricDirection.MINIMIZE, 0.20, 0.50, 0.30),
        (MetricDirection.MINIMIZE, 0.70, 0.50, -0.20),
    ],
)
def test_observed_trust_gap_formula_supports_both_metric_directions(
    direction: MetricDirection,
    raw: float,
    verified: float,
    expected: float,
) -> None:
    gap = ObservedTrustGap(
        status=TrustGapStatus.COMPUTED,
        reason="comparable protocol",
        metric="roc_auc" if direction is MetricDirection.MAXIMIZE else "rmse",
        metric_direction=direction,
        raw_score=raw,
        verified_score=verified,
        observed_trust_gap=expected,
        raw_protocol="raw",
        verified_protocol="verified",
        raw_protocol_path="trust_gap/raw.json",
        raw_predictions_path="trust_gap/raw.csv",
    )

    assert gap.observed_trust_gap == pytest.approx(expected)


def test_not_computed_gap_cannot_smuggle_a_diagnostic_score() -> None:
    with pytest.raises(ValidationError, match="cannot contain scores"):
        ObservedTrustGap(
            status=TrustGapStatus.NOT_COMPUTED,
            reason="not comparable",
            metric="roc_auc",
            metric_direction="maximize",
            raw_score=0.99,
            raw_protocol="not_available",
            verified_protocol="verified",
        )


def test_partial_legacy_runtime_renders_unavailable_values_instead_of_python_none() -> None:
    runtime = RuntimeTelemetry(
        configured_search_budget_seconds=30,
        trials_started=2,
        trials_completed=1,
        trials_failed=0,
        stop_reason=SearchStopReason.USER_INTERRUPTED,
    )
    certificate = _certificate(
        replay_status=VerificationStatus.NOT_AVAILABLE,
        runtime=runtime,
    )

    html = render_trust_certificate(certificate)

    assert certificate.status is TrustCertificateStatus.INCOMPLETE
    assert "Unavailable" in html
    assert ">None<" not in html

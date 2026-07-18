"""End-to-end Trust Layer persistence, CLI, corruption, and compatibility tests."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from autonomous_automl import AutoMLConfig, AutoMLRun
from autonomous_automl.api import (
    create_or_validate_trust_artifacts,
    load_best_pipeline,
    open_run,
    validate_run_artifacts,
)
from autonomous_automl.cli.app import app
from autonomous_automl.contracts import (
    ObservedTrustGap,
    PipelineSelectionExplanation,
    RuntimeTelemetry,
    SearchStopReason,
    TrustCertificate,
    TrustCertificateStatus,
    TrustGapStatus,
    VerificationStatus,
)
from autonomous_automl.evaluation.selection import build_leaderboard, select_finalist
from autonomous_automl.runtime import RUNTIME_STATE_KEY
from autonomous_automl.utils.errors import ArtifactValidationError, PlannedInterruption

_TRUST_ARTIFACT_NAMES = {
    "runtime_telemetry",
    "selection_explanation",
    "trust_gap",
    "trust_gap_raw_protocol",
    "trust_gap_raw_predictions",
    "trust_certificate_json",
    "trust_certificate_html",
    "manifest",
}


@pytest.fixture(scope="module")
def trusted_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("trusted-run")
    rng = np.random.default_rng(119)
    rows = 140
    signal = rng.normal(size=rows)
    latent = 0.45 * signal + rng.normal(size=rows)
    target = (latent > np.median(latent)).astype(int)
    train = pd.DataFrame(
        {
            "customer_id": [f"customer-{index:04d}" for index in range(rows)],
            "signal": signal,
            "noise": rng.normal(size=rows),
            "target_copy": target,
            "target": target,
        }
    )
    test = pd.DataFrame(
        {
            "customer_id": [f"new-customer-{index:04d}" for index in range(12)],
            "signal": rng.normal(size=12),
            "noise": rng.normal(size=12),
            "target_copy": np.zeros(12, dtype=int),
        }
    )
    train_path = root / "train.csv"
    test_path = root / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    output = root / "run"
    AutoMLRun(
        AutoMLConfig(
            target="target",
            task="binary_classification",
            metric="roc_auc",
            budget_seconds=4,
            random_seed=42,
            test_path=test_path,
            output_dir=output,
            trial_timeout_seconds=3,
            optimization_profile="accuracy",
            compute_trust_gap=True,
            trust_gap_timeout_seconds=5,
        )
    ).fit(train_path)
    return output


@pytest.mark.integration
def test_successful_run_persists_a_coherent_self_verified_certificate(
    trusted_run: Path,
) -> None:
    directory, store, manifest = open_run(trusted_run)
    trust = create_or_validate_trust_artifacts(directory)
    certificate = trust.certificate

    assert trust.json_path.is_file()
    assert trust.html_path.is_file()
    assert certificate == TrustCertificate.read_json(trust.json_path)
    assert certificate.status is TrustCertificateStatus.SELF_VERIFIED_WITH_WARNINGS
    assert certificate.artifact_validation.status is VerificationStatus.PASS
    assert certificate.prediction_replay.status is VerificationStatus.PASS
    assert certificate.final_pipeline == manifest.best_pipeline
    assert certificate.verified_score == manifest.metrics["best_score"]
    assert certificate.runtime_telemetry == RuntimeTelemetry.read_json(
        directory / manifest.artifacts["runtime_telemetry"]
    )
    assert certificate.selection_explanation == PipelineSelectionExplanation.read_json(
        directory / manifest.artifacts["selection_explanation"]
    )
    assert certificate.observed_trust_gap == ObservedTrustGap.read_json(
        directory / manifest.artifacts["trust_gap"]
    )
    assert certificate.resume.operation_count == store.resume_operation_count(manifest.run_id)


@pytest.mark.integration
def test_report_certificate_and_cli_show_only_persisted_trust_evidence(trusted_run: Path) -> None:
    certificate_json = (trusted_run / "trust_certificate.json").read_text(encoding="utf-8")
    certificate_html = (trusted_run / "trust_certificate.html").read_text(encoding="utf-8")
    report_html = (trusted_run / "report.html").read_text(encoding="utf-8")

    result = CliRunner().invoke(app, ["trust", str(trusted_run)])

    assert result.exit_code == 0
    assert "SELF_VERIFIED_WITH_WARNINGS" in result.output
    assert "Observed Trust Gap" in result.output
    assert "Search elapsed" in result.output
    assert "trust_certificate.json" in result.output
    assert 'href="trust_certificate.html"' in report_html
    assert "Why this pipeline won" in report_html
    assert "Budget and runtime telemetry" in report_html
    assert "not an external, regulatory, or security certification" in certificate_html
    assert str(trusted_run.parent) not in certificate_json
    assert str(trusted_run.parent) not in certificate_html


@pytest.mark.integration
def test_trust_gap_is_separate_and_does_not_change_leaderboard_or_selection(
    trusted_run: Path,
) -> None:
    _, store, manifest = open_run(trusted_run)
    trials_before = store.list_trials(manifest.run_id)
    leaderboard_before = (trusted_run / "leaderboard.csv").read_bytes()
    selected_before = select_finalist(
        build_leaderboard(trials_before),
        manifest.configuration.optimization_profile,
        require_full_fidelity=True,
    )

    certificate = create_or_validate_trust_artifacts(trusted_run).certificate

    _, reloaded_store, reloaded_manifest = open_run(trusted_run)
    trials_after = reloaded_store.list_trials(reloaded_manifest.run_id)
    selected_after = select_finalist(
        build_leaderboard(trials_after),
        reloaded_manifest.configuration.optimization_profile,
        require_full_fidelity=True,
    )
    gap = certificate.observed_trust_gap
    assert gap is not None
    assert gap.status is TrustGapStatus.COMPUTED
    assert gap.raw_score == pytest.approx(1.0)
    assert gap.observed_trust_gap is not None
    assert gap.observed_trust_gap > 0
    assert trials_after == trials_before
    assert (trusted_run / "leaderboard.csv").read_bytes() == leaderboard_before
    assert selected_after.selected.trial_id == selected_before.selected.trial_id
    assert selected_after.selected.pipeline_spec == manifest.best_pipeline
    assert certificate.selection_explanation is not None
    assert certificate.selection_explanation.selected_trial_id == selected_before.selected.trial_id


@pytest.mark.integration
def test_revalidating_the_same_run_is_deterministic(trusted_run: Path) -> None:
    first = create_or_validate_trust_artifacts(trusted_run)
    first_json = first.json_path.read_bytes()
    first_html = first.html_path.read_bytes()

    second = create_or_validate_trust_artifacts(trusted_run)

    assert second.certificate == first.certificate
    assert second.json_path.read_bytes() == first_json
    assert second.html_path.read_bytes() == first_html


@pytest.mark.integration
def test_model_corruption_yields_failed_certificate_and_refuses_loading(
    trusted_run: Path,
    tmp_path: Path,
) -> None:
    corrupted = tmp_path / "corrupted-model-run"
    shutil.copytree(trusted_run, corrupted)
    (corrupted / "best_pipeline.joblib").write_bytes(b"not a valid joblib artifact")

    trust = create_or_validate_trust_artifacts(corrupted)
    cli = CliRunner().invoke(app, ["trust", str(corrupted)])

    assert trust.certificate.status is TrustCertificateStatus.FAILED
    assert trust.certificate.artifact_validation.status is VerificationStatus.FAIL
    assert cli.exit_code == 1
    assert "FAILED" in cli.output
    with pytest.raises(ArtifactValidationError):
        load_best_pipeline(corrupted)


@pytest.mark.integration
def test_corrupted_certificate_is_detected_and_recalculated_as_failed(
    trusted_run: Path,
    tmp_path: Path,
) -> None:
    corrupted = tmp_path / "corrupted-certificate-run"
    shutil.copytree(trusted_run, corrupted)
    (corrupted / "trust_certificate.json").write_text("{}", encoding="utf-8")

    trust = create_or_validate_trust_artifacts(corrupted)

    assert trust.certificate.status is TrustCertificateStatus.FAILED
    assert trust.certificate.artifact_validation.status is VerificationStatus.FAIL


@pytest.mark.integration
def test_corrupted_runtime_contract_yields_failed_certificate_instead_of_an_exception(
    trusted_run: Path,
    tmp_path: Path,
) -> None:
    corrupted = tmp_path / "corrupted-runtime-run"
    shutil.copytree(trusted_run, corrupted)
    (corrupted / "runtime_telemetry.json").write_text("{}", encoding="utf-8")

    trust = create_or_validate_trust_artifacts(corrupted)

    assert trust.certificate.status is TrustCertificateStatus.FAILED
    assert trust.certificate.artifact_validation.status is VerificationStatus.FAIL
    assert "runtime_telemetry" in trust.certificate.artifact_validation.detail


@pytest.mark.integration
def test_legacy_shaped_run_remains_readable_and_gets_a_partial_certificate(
    trusted_run: Path,
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy-shaped-run"
    shutil.copytree(trusted_run, legacy)
    directory, store, manifest = open_run(legacy)
    registrations = {
        registration.name: registration for registration in store.list_artifacts(manifest.run_id)
    }
    legacy_artifacts = {
        name: path for name, path in manifest.artifacts.items() if name not in _TRUST_ARTIFACT_NAMES
    }
    legacy_states = dict(manifest.scheduler_state)
    legacy_states.pop(RUNTIME_STATE_KEY, None)
    legacy_manifest = manifest.model_copy(
        update={"artifacts": legacy_artifacts, "scheduler_state": legacy_states},
        deep=True,
    )
    store.update_manifest(legacy_manifest)
    legacy_manifest.write_json(directory / "manifest.json")
    with sqlite3.connect(directory / "registry.sqlite3") as connection:
        placeholders = ",".join("?" for _ in _TRUST_ARTIFACT_NAMES)
        connection.execute(
            f"DELETE FROM artifacts WHERE run_id = ? AND name IN ({placeholders})",
            (manifest.run_id, *_TRUST_ARTIFACT_NAMES),
        )
    for name in _TRUST_ARTIFACT_NAMES - {"manifest"}:
        registration = registrations.get(name)
        if registration is not None:
            path = directory / registration.relative_path
            if path.is_file():
                path.unlink()

    reloaded_directory, _, reloaded_manifest = open_run(legacy)
    trust = create_or_validate_trust_artifacts(reloaded_directory)

    assert reloaded_manifest.run_id == manifest.run_id
    assert trust.persisted
    assert trust.certificate.runtime_telemetry is None
    assert trust.certificate.observed_trust_gap is None
    assert trust.certificate.selection_explanation is not None
    assert trust.certificate.prediction_replay.status is VerificationStatus.PASS
    assert any("runtime telemetry is unavailable" in item for item in trust.certificate.warnings)
    assert validate_run_artifacts(legacy).predictions_match is True


def test_pre_trust_config_payload_loads_with_disabled_compatible_defaults() -> None:
    payload = AutoMLConfig(target="target", budget_seconds=30).to_json_value()
    payload.pop("compute_trust_gap")
    payload.pop("trust_gap_timeout_seconds")

    restored = AutoMLConfig.model_validate(payload)

    assert restored.compute_trust_gap is False
    assert restored.trust_gap_timeout_seconds == 30


@pytest.mark.integration
def test_interrupted_runtime_telemetry_accumulates_across_transactional_resume(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(515)
    rows = 100
    signal = rng.normal(size=rows)
    target = (signal + rng.normal(0, 0.7, size=rows) > 0).astype(int)
    train = pd.DataFrame({"signal": signal, "noise": rng.normal(size=rows), "target": target})
    test = pd.DataFrame({"signal": [-1.0, 0.0, 1.0], "noise": [0.1, -0.2, 0.3]})
    train_path = tmp_path / "resume-train.csv"
    test_path = tmp_path / "resume-test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    output = tmp_path / "resume-run"
    config = AutoMLConfig(
        target="target",
        budget_seconds=4,
        random_seed=42,
        test_path=test_path,
        output_dir=output,
        trial_timeout_seconds=3,
    )

    with pytest.raises(PlannedInterruption):
        AutoMLRun(config).fit(train_path, interrupt_after_trials=2)
    _, interrupted_store, interrupted_manifest = open_run(output)
    interrupted_trials = interrupted_store.list_trials(interrupted_manifest.run_id)
    partial = RuntimeTelemetry.model_validate(
        interrupted_manifest.scheduler_state[RUNTIME_STATE_KEY]
    )

    result = AutoMLRun.resume(output, additional_budget_seconds=3)
    _, resumed_store, resumed_manifest = open_run(result.output_dir)
    resumed_trials = resumed_store.list_trials(resumed_manifest.run_id)
    certificate = create_or_validate_trust_artifacts(output).certificate
    final = certificate.runtime_telemetry

    assert partial.stop_reason is SearchStopReason.USER_INTERRUPTED
    assert final is not None
    assert final.search_elapsed_seconds is not None
    assert partial.search_elapsed_seconds is not None
    assert final.search_elapsed_seconds >= partial.search_elapsed_seconds
    assert final.configured_search_budget_seconds == pytest.approx(7)
    assert certificate.resume.resumed is True
    assert certificate.resume.operation_count == 1
    assert {trial.trial_id for trial in interrupted_trials}.issubset(
        {trial.trial_id for trial in resumed_trials}
    )
    assert len({trial.trial_id for trial in resumed_trials}) == len(resumed_trials)

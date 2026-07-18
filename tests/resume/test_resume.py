"""M6 acceptance tests for transactional tracking and safe run resumption."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from autonomous_automl.contracts import (
    AutoMLConfig,
    FidelitySpec,
    FoldResult,
    MetricName,
    PipelineSpec,
    RunManifest,
    RunStatus,
    TrialResult,
    TrialStatus,
)
from autonomous_automl.data import load_dataset
from autonomous_automl.profiling import detect_leakage, profile_dataset
from autonomous_automl.tracking.artifacts import ArtifactStore
from autonomous_automl.tracking.migrations import LATEST_SCHEMA_VERSION
from autonomous_automl.tracking.resume import (
    ResumeManager,
    load_resume_manifest,
    verify_manifest_source_hashes,
)
from autonomous_automl.tracking.store import ExperimentStore
from autonomous_automl.utils.errors import ResumeError, TrackingError
from autonomous_automl.validation import ValidationPlanner, audit_validation

STARTED_AT = datetime(2026, 7, 17, 8, 0, tzinfo=UTC)
FINISHED_AT = STARTED_AT + timedelta(seconds=5)


@dataclass(frozen=True, slots=True)
class RunFixture:
    run_dir: Path
    train_path: Path
    manifest: RunManifest
    specification: PipelineSpec
    fidelity: FidelitySpec


def make_run_fixture(tmp_path: Path) -> RunFixture:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    train_path = tmp_path / "train.csv"
    frame = pd.DataFrame(
        {
            "signal": [float((index * 7) % 11) / 10 for index in range(24)],
            "category": [f"category-{index % 3}" for index in range(24)],
            "target": [index % 2 for index in range(24)],
        }
    )
    frame.to_csv(train_path, index=False)
    config = AutoMLConfig(
        target="target",
        task="binary_classification",
        metric="roc_auc",
        budget_seconds=120,
        random_seed=42,
        n_jobs=1,
        output_dir=run_dir,
    )
    dataset = load_dataset(train_path, config)
    profile = profile_dataset(dataset)
    leakage_report = detect_leakage(dataset, profile)
    validation_plan = ValidationPlanner(max_splits=2).plan(dataset, profile, config)
    validation_audit = audit_validation(dataset, validation_plan)
    specification = PipelineSpec(
        family="linear",
        task="binary_classification",
        numeric_imputer="median",
        numeric_scaler="standard",
        categorical_imputer="constant",
        categorical_encoder="one_hot",
        datetime_transformer="drop",
        feature_selector=None,
        model_name="logistic_regression",
        model_params={"C": 1.0, "class_weight": None, "max_iter": 100},
        excluded_columns=[],
        random_seed=42,
    )
    fidelity = FidelitySpec(
        level=2,
        sample_fraction=1.0,
        n_folds=2,
        max_iterations=100,
        seeds=[42],
    )
    manifest = RunManifest(
        run_id="run-resume-acceptance",
        package_version="0.1.0",
        status=RunStatus.RUNNING,
        created_at=STARTED_AT,
        updated_at=STARTED_AT,
        configuration=config,
        source_hashes=dataset.bundle.source_hashes,
        dataset=dataset.bundle,
        dataset_profile=profile,
        leakage_report=leakage_report,
        validation_plan=validation_plan,
        validation_audit=validation_audit,
        dependency_versions={"python": "3.12", "sqlite": "3"},
        random_seed=42,
        scheduler_state={"phase": "search"},
    )
    manifest.write_json(run_dir / "manifest.json")
    return RunFixture(run_dir, train_path, manifest, specification, fidelity)


def make_completed_trial(fixture: RunFixture) -> TrialResult:
    folds = [
        FoldResult(
            fold_index=index,
            seed=42 + index,
            score=score,
            user_metric_value=score,
            fit_seconds=0.5,
            predict_seconds=0.1,
            n_train_rows=12,
            n_validation_rows=12,
        )
        for index, score in enumerate((0.7, 0.8))
    ]
    return TrialResult(
        trial_id="trial-completed",
        family=fixture.specification.family,
        pipeline_spec=fixture.specification,
        fidelity=fixture.fidelity,
        status=TrialStatus.COMPLETED,
        primary_metric=MetricName.ROC_AUC,
        mean_score=0.75,
        std_score=0.05,
        fold_scores=[0.7, 0.8],
        fold_results=folds,
        fit_seconds=1.0,
        predict_seconds=0.2,
        peak_memory_mb=64.0,
        started_at=STARTED_AT,
        finished_at=FINISHED_AT,
    )


def make_failed_trial(fixture: RunFixture) -> TrialResult:
    return TrialResult(
        trial_id="trial-failed",
        family=fixture.specification.family,
        pipeline_spec=fixture.specification,
        fidelity=fixture.fidelity,
        status=TrialStatus.FAILED,
        primary_metric=MetricName.ROC_AUC,
        failure_type="ValueError",
        failure_message="synthetic candidate failure",
        started_at=STARTED_AT,
        finished_at=FINISHED_AT,
    )


def test_resume_refuses_a_source_whose_bytes_changed(tmp_path: Path) -> None:
    fixture = make_run_fixture(tmp_path)
    restored = load_resume_manifest(fixture.run_dir)
    verify_manifest_source_hashes(restored)

    fixture.train_path.write_text(
        fixture.train_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ResumeError, match="source hash changed"):
        verify_manifest_source_hashes(restored)


def test_resume_rejects_a_corrupt_manifest_explicitly(tmp_path: Path) -> None:
    fixture = make_run_fixture(tmp_path)
    (fixture.run_dir / "manifest.json").write_text("{not valid JSON", encoding="utf-8")

    with pytest.raises(ResumeError, match="manifest is invalid or unreadable"):
        load_resume_manifest(fixture.run_dir)


def test_registry_migration_is_idempotent_and_integrity_checked(tmp_path: Path) -> None:
    registry_path = tmp_path / "run" / "registry.sqlite3"
    store = ExperimentStore.open(registry_path)

    first_version = store.migrate()
    second_version = store.migrate()
    store.integrity_check()

    assert first_version == second_version == LATEST_SCHEMA_VERSION
    assert registry_path.is_file()


def test_completed_and_failed_trials_round_trip_with_folds_snapshots_and_budget(
    tmp_path: Path,
) -> None:
    fixture = make_run_fixture(tmp_path)
    store = ExperimentStore.open(fixture.run_dir / "registry.sqlite3")
    store.create_run(fixture.manifest)
    completed = make_completed_trial(fixture)
    completed_reservation = store.reserve_trial(
        fixture.manifest.run_id,
        completed.trial_id,
        completed.pipeline_spec,
        completed.fidelity,
        completed.primary_metric,
    )
    assert completed_reservation.reserved

    first_revision = store.record_trial_result(
        completed,
        attempt_token=completed_reservation.attempt_token,
        lease_epoch=completed_reservation.lease_epoch,
        component_states={
            "scheduler": {"level": 2, "promoted": [completed.trial_id]},
            "allocator": {"linear": {"attempts": 1}},
        },
        consumed_seconds=12.5,
    )
    idempotent_revision = store.record_trial_result(
        completed,
        attempt_token=completed_reservation.attempt_token,
        lease_epoch=completed_reservation.lease_epoch,
        component_states={"scheduler": {"should": "not overwrite"}},
        consumed_seconds=999.0,
    )

    failed_specification = fixture.specification.model_copy(update={"random_seed": 43})
    failed = make_failed_trial(fixture).model_copy(
        update={"pipeline_spec": failed_specification},
        deep=True,
    )
    failed_reservation = store.reserve_trial(
        fixture.manifest.run_id,
        failed.trial_id,
        failed.pipeline_spec,
        failed.fidelity,
        failed.primary_metric,
    )
    assert failed_reservation.reserved
    second_revision = store.record_trial_result(
        failed,
        attempt_token=failed_reservation.attempt_token,
        lease_epoch=failed_reservation.lease_epoch,
        component_states={"budget": {"reserved_seconds": 30.0}},
        consumed_seconds=2.0,
    )

    reopened = ExperimentStore.open(fixture.run_dir / "registry.sqlite3")
    assert reopened.get_trial(completed.trial_id) == completed
    assert reopened.get_trial(failed.trial_id) == failed
    assert reopened.list_trials(fixture.manifest.run_id) == [completed, failed]
    assert reopened.get_trial(completed.trial_id).fold_results == completed.fold_results
    assert first_revision == idempotent_revision == 1
    assert second_revision == 2
    assert reopened.load_component_states(fixture.manifest.run_id) == {
        "allocator": {"linear": {"attempts": 1}},
        "budget": {"reserved_seconds": 30.0},
        "scheduler": {"level": 2, "promoted": [completed.trial_id]},
    }
    progress = reopened.get_run_progress(fixture.manifest.run_id)
    assert progress.consumed_seconds == pytest.approx(14.5)
    assert progress.budget_seconds == 120.0
    assert progress.checkpoint_revision == 2
    extended = reopened.add_budget(fixture.manifest.run_id, 30.0)
    assert extended.budget_seconds == 150.0
    assert extended.consumed_seconds == pytest.approx(14.5)
    assert (
        ExperimentStore.open(fixture.run_dir / "registry.sqlite3").get_run_progress(
            fixture.manifest.run_id
        )
        == extended
    )


def test_candidate_deduplication_keeps_one_logical_trial(tmp_path: Path) -> None:
    fixture = make_run_fixture(tmp_path)
    store = ExperimentStore.open(fixture.run_dir / "registry.sqlite3")
    store.create_run(fixture.manifest)

    first = store.reserve_trial(
        fixture.manifest.run_id,
        "trial-original",
        fixture.specification,
        fixture.fidelity,
        MetricName.ROC_AUC,
    )
    duplicate = store.reserve_trial(
        fixture.manifest.run_id,
        "trial-duplicate-id",
        fixture.specification,
        fixture.fidelity,
        MetricName.ROC_AUC,
    )

    assert first.reserved
    assert not duplicate.reserved
    assert duplicate.trial_id == first.trial_id == "trial-original"
    assert duplicate.candidate_key == first.candidate_key
    assert duplicate.sequence_number == first.sequence_number


def test_stale_running_trial_resumes_without_losing_completed_work(tmp_path: Path) -> None:
    fixture = make_run_fixture(tmp_path)
    store = ExperimentStore.open(fixture.run_dir / "registry.sqlite3")
    store.create_run(fixture.manifest)
    completed = make_completed_trial(fixture)
    completed_reservation = store.reserve_trial(
        fixture.manifest.run_id,
        completed.trial_id,
        completed.pipeline_spec,
        completed.fidelity,
        completed.primary_metric,
    )
    store.record_trial_result(
        completed,
        attempt_token=completed_reservation.attempt_token,
        lease_epoch=completed_reservation.lease_epoch,
        consumed_seconds=5.0,
    )

    interrupted_specification = fixture.specification.model_copy(update={"random_seed": 99})
    initial = store.reserve_trial(
        fixture.manifest.run_id,
        "trial-stale",
        interrupted_specification,
        fixture.fidelity,
        MetricName.ROC_AUC,
    )
    interrupted = store.mark_stale_trials_interrupted(
        fixture.manifest.run_id,
        stale_after_seconds=0,
    )
    retryable = store.list_retryable_trials(fixture.manifest.run_id)
    resumed = ExperimentStore.open(fixture.run_dir / "registry.sqlite3").reserve_trial(
        fixture.manifest.run_id,
        "a-new-id-must-not-replace-the-logical-trial",
        interrupted_specification,
        fixture.fidelity,
        MetricName.ROC_AUC,
    )

    assert initial.reserved
    assert interrupted == ["trial-stale"]
    assert len(retryable) == 1
    assert retryable[0].trial_id == "trial-stale"
    assert retryable[0].pipeline_spec == interrupted_specification
    assert retryable[0].fidelity == fixture.fidelity
    assert retryable[0].primary_metric is MetricName.ROC_AUC
    assert retryable[0].attempt_count == 1
    assert resumed.reserved
    assert resumed.trial_id == "trial-stale"
    assert resumed.attempt_count == 2
    assert resumed.status is TrialStatus.RUNNING
    assert store.get_trial(completed.trial_id) == completed
    assert initial.candidate_key not in store.completed_candidate_keys(fixture.manifest.run_id)
    assert len(store.completed_candidate_keys(fixture.manifest.run_id)) == 1
    progress = store.get_run_progress(fixture.manifest.run_id)
    assert progress.consumed_seconds == pytest.approx(5.0)
    assert progress.checkpoint_revision == 1


def test_corrupt_registry_fails_with_an_explicit_tracking_error(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite3"
    ExperimentStore.open(registry_path).integrity_check()
    registry_path.write_bytes(b"this is not a SQLite database")

    with pytest.raises(TrackingError, match=r"corrupt|unreadable|could not be opened"):
        ExperimentStore(registry_path).integrity_check()


def test_resume_manager_restores_state_budget_and_completed_deduplication(
    tmp_path: Path,
) -> None:
    fixture = make_run_fixture(tmp_path)
    store = ExperimentStore.open(fixture.run_dir / "registry.sqlite3")
    store.create_run(fixture.manifest)
    completed = make_completed_trial(fixture)
    completed_reservation = store.reserve_trial(
        fixture.manifest.run_id,
        completed.trial_id,
        completed.pipeline_spec,
        completed.fidelity,
        completed.primary_metric,
    )
    store.record_trial_result(
        completed,
        attempt_token=completed_reservation.attempt_token,
        lease_epoch=completed_reservation.lease_epoch,
        component_states={"allocator": {"linear": {"attempts": 1}}},
        consumed_seconds=15.0,
    )
    stale_specification = fixture.specification.model_copy(update={"random_seed": 99})
    store.reserve_trial(
        fixture.manifest.run_id,
        "trial-abandoned",
        stale_specification,
        fixture.fidelity,
        MetricName.ROC_AUC,
    )

    manager = ResumeManager()
    state = manager.prepare(
        fixture.run_dir,
        additional_budget_seconds=30,
        acquire_lease=True,
    )

    assert state.interrupted_trial_ids == ["trial-abandoned"]
    assert state.terminal_trials == [completed]
    assert [trial.trial_id for trial in state.retryable_trials] == ["trial-abandoned"]
    assert state.retryable_trials[0].pipeline_spec == stale_specification
    assert completed_reservation.candidate_key in state.completed_candidate_keys
    assert state.component_states == {"allocator": {"linear": {"attempts": 1}}}
    assert state.progress.budget_seconds == pytest.approx(150.0)
    assert state.progress.consumed_seconds == pytest.approx(15.0)
    assert state.remaining_budget_seconds == pytest.approx(135.0)
    assert state.manifest.status is RunStatus.RUNNING
    assert state.manifest.scheduler_state["tracking"] == {
        "checkpoint_revision": 1,
        "consumed_seconds": 15.0,
        "effective_budget_seconds": 150.0,
    }
    manager.release(state)


def test_resume_manager_rejects_manifest_registry_identity_mismatch(tmp_path: Path) -> None:
    fixture = make_run_fixture(tmp_path)
    ExperimentStore.open(fixture.run_dir / "registry.sqlite3").create_run(fixture.manifest)
    changed_config = fixture.manifest.configuration.model_copy(update={"random_seed": 7})
    changed_manifest = fixture.manifest.model_copy(
        update={"configuration": changed_config, "random_seed": 7},
        deep=True,
    )
    changed_manifest.write_json(fixture.run_dir / "manifest.json")

    with pytest.raises(ResumeError, match="registry is invalid or inconsistent"):
        ResumeManager().prepare(fixture.run_dir, acquire_lease=False)


def test_concurrent_candidate_reservations_have_exactly_one_winner(tmp_path: Path) -> None:
    fixture = make_run_fixture(tmp_path)
    store = ExperimentStore.open(fixture.run_dir / "registry.sqlite3")
    store.create_run(fixture.manifest)

    def reserve(index: int) -> bool:
        return store.reserve_trial(
            fixture.manifest.run_id,
            f"trial-concurrent-{index}",
            fixture.specification,
            fixture.fidelity,
            MetricName.ROC_AUC,
        ).reserved

    with ThreadPoolExecutor(max_workers=2) as executor:
        winners = list(executor.map(reserve, [1, 2]))

    assert sorted(winners) == [False, True]


def test_trial_and_snapshot_transaction_rolls_back_together(tmp_path: Path) -> None:
    fixture = make_run_fixture(tmp_path)
    store = ExperimentStore.open(fixture.run_dir / "registry.sqlite3")
    store.create_run(fixture.manifest)
    completed = make_completed_trial(fixture)
    reservation = store.reserve_trial(
        fixture.manifest.run_id,
        completed.trial_id,
        completed.pipeline_spec,
        completed.fidelity,
        completed.primary_metric,
    )

    with pytest.raises(ValueError, match="Out of range float values"):
        store.record_trial_result(
            completed,
            attempt_token=reservation.attempt_token,
            lease_epoch=reservation.lease_epoch,
            component_states={"scheduler": {"invalid": float("nan")}},
            consumed_seconds=10.0,
        )

    assert store.get_trial(completed.trial_id) is None
    assert store.get_run_progress(fixture.manifest.run_id).checkpoint_revision == 0
    assert store.get_run_progress(fixture.manifest.run_id).consumed_seconds == 0
    assert store.load_component_states(fixture.manifest.run_id) == {}


def test_run_lease_is_exclusive_and_releasable(tmp_path: Path) -> None:
    fixture = make_run_fixture(tmp_path)
    store = ExperimentStore.open(fixture.run_dir / "registry.sqlite3")
    store.create_run(fixture.manifest)

    assert store.acquire_run_lease(fixture.manifest.run_id, "owner-one", ttl_seconds=60)
    assert not store.acquire_run_lease(fixture.manifest.run_id, "owner-two", ttl_seconds=60)
    store.renew_run_lease(fixture.manifest.run_id, "owner-one", ttl_seconds=120)
    store.release_run_lease(fixture.manifest.run_id, "owner-one")
    assert store.acquire_run_lease(fixture.manifest.run_id, "owner-two", ttl_seconds=60)


def test_verified_artifact_metadata_is_registered_transactionally(tmp_path: Path) -> None:
    fixture = make_run_fixture(tmp_path)
    store = ExperimentStore.open(fixture.run_dir / "registry.sqlite3")
    store.create_run(fixture.manifest)
    artifact = ArtifactStore(fixture.run_dir).write_text(
        "environment",
        '{"python":"3.12"}',
        relative_path="environment.json",
        kind="environment",
        media_type="application/json",
    )

    registration = store.register_artifact(
        fixture.manifest.run_id,
        name=artifact.name,
        kind=artifact.kind,
        relative_path=artifact.relative_path,
        sha256=artifact.sha256,
        size_bytes=artifact.size_bytes,
        media_type=artifact.media_type,
    )

    assert store.list_artifacts(fixture.manifest.run_id) == [registration]


def test_late_result_from_interrupted_attempt_is_fenced_out(tmp_path: Path) -> None:
    fixture = make_run_fixture(tmp_path)
    store = ExperimentStore.open(fixture.run_dir / "registry.sqlite3")
    store.create_run(fixture.manifest)
    first_attempt = store.reserve_trial(
        fixture.manifest.run_id,
        "trial-fenced",
        fixture.specification,
        fixture.fidelity,
        MetricName.ROC_AUC,
    )
    store.mark_stale_trials_interrupted(fixture.manifest.run_id)
    second_attempt = store.reserve_trial(
        fixture.manifest.run_id,
        "ignored-new-id",
        fixture.specification,
        fixture.fidelity,
        MetricName.ROC_AUC,
    )
    result = make_completed_trial(fixture).model_copy(
        update={"trial_id": "trial-fenced"},
        deep=True,
    )

    with pytest.raises(TrackingError, match="attempt token"):
        store.record_trial_result(
            result,
            attempt_token=first_attempt.attempt_token,
            lease_epoch=first_attempt.lease_epoch,
        )
    store.record_trial_result(
        result,
        attempt_token=second_attempt.attempt_token,
        lease_epoch=second_attempt.lease_epoch,
    )
    assert store.get_trial("trial-fenced") == result


def test_resume_budget_operation_is_idempotent_across_crash_retry(tmp_path: Path) -> None:
    fixture = make_run_fixture(tmp_path)
    store = ExperimentStore.open(fixture.run_dir / "registry.sqlite3")
    store.create_run(fixture.manifest)

    first = store.prepare_resume(
        fixture.manifest.run_id,
        operation_id="resume-operation-stable",
        additional_budget_seconds=30.0,
    )[1]
    retried = store.prepare_resume(
        fixture.manifest.run_id,
        operation_id="resume-operation-stable",
        additional_budget_seconds=30.0,
    )[1]

    assert first.budget_seconds == retried.budget_seconds == 150.0


def test_integrity_check_rejects_schema_with_missing_table(tmp_path: Path) -> None:
    import sqlite3

    registry_path = tmp_path / "registry.sqlite3"
    store = ExperimentStore.open(registry_path)
    with sqlite3.connect(registry_path) as connection:
        connection.execute("DROP TABLE scheduler_state")

    with pytest.raises(TrackingError, match="missing tables"):
        store.integrity_check()

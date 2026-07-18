"""Manifest and source-integrity checks used before restoring mutable run state."""

from __future__ import annotations

import math
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from autonomous_automl.contracts import RunManifest, RunStatus, TrialResult
from autonomous_automl.tracking.store import ExperimentStore, RetryableTrial, RunProgress
from autonomous_automl.utils.errors import ResumeError, TrackingError
from autonomous_automl.utils.hashing import sha256_file, sha256_json
from autonomous_automl.utils.json import JsonValue

MANIFEST_FILENAME = "manifest.json"
REGISTRY_FILENAME = "registry.sqlite3"


@dataclass(frozen=True, slots=True)
class ResumeState:
    """Validated state from which the search controller can continue exactly once."""

    run_directory: Path
    manifest: RunManifest
    terminal_trials: list[TrialResult]
    retryable_trials: list[RetryableTrial]
    completed_candidate_keys: set[str]
    interrupted_trial_ids: list[str]
    component_states: dict[str, JsonValue]
    progress: RunProgress
    remaining_budget_seconds: float
    lease_owner: str | None


class ResumeManager:
    """Reconcile manifest, hashes, SQLite state, stale trials, lease, and budget."""

    def prepare(
        self,
        run_directory: str | Path,
        *,
        additional_budget_seconds: float | None = None,
        stale_after_seconds: float = 0.0,
        acquire_lease: bool = True,
        lease_ttl_seconds: int = 120,
    ) -> ResumeState:
        manifest = load_resume_manifest(run_directory)
        if manifest.status is RunStatus.COMPLETED:
            raise ResumeError("a completed run cannot be resumed")
        verify_manifest_source_hashes(manifest)
        directory = Path(run_directory).expanduser().resolve()
        registry_path = directory / REGISTRY_FILENAME
        if not registry_path.is_file():
            raise ResumeError(f"run registry is missing: {registry_path}")
        if additional_budget_seconds is not None and (
            not math.isfinite(additional_budget_seconds) or additional_budget_seconds <= 0
        ):
            raise ResumeError("additional resume budget must be finite and positive")
        if lease_ttl_seconds <= 0:
            raise ResumeError("lease_ttl_seconds must be positive")

        try:
            store = ExperimentStore.open(registry_path)
            store.integrity_check()
            stored_manifest = store.get_manifest(manifest.run_id)
            _validate_manifest_consistency(manifest, stored_manifest)
            if stored_manifest.status is RunStatus.COMPLETED:
                raise ResumeError("a completed run cannot be resumed")
        except TrackingError as error:
            raise ResumeError("run registry is invalid or inconsistent") from error

        lease_owner = f"resume-{uuid.uuid4().hex}" if acquire_lease else None
        lease_acquired = False
        try:
            if lease_owner is not None:
                lease_acquired = store.acquire_run_lease(
                    manifest.run_id,
                    lease_owner,
                    ttl_seconds=lease_ttl_seconds,
                )
                if not lease_acquired:
                    raise ResumeError("another process currently owns the run lease")
            lease_epoch = store.get_run_progress(manifest.run_id).lease_epoch
            operation_id = sha256_json(
                {
                    "additional_budget_seconds": additional_budget_seconds or 0.0,
                    "manifest_updated_at": manifest.updated_at.isoformat(),
                    "run_id": manifest.run_id,
                }
            )
            interrupted, progress = store.prepare_resume(
                manifest.run_id,
                operation_id=operation_id,
                additional_budget_seconds=additional_budget_seconds or 0.0,
                stale_after_seconds=stale_after_seconds,
                lease_owner=lease_owner,
                lease_epoch=lease_epoch if lease_owner is not None else None,
            )
            trials = store.list_trials(manifest.run_id)
            retryable_trials = store.list_retryable_trials(manifest.run_id)
            states = store.load_component_states(manifest.run_id)
            completed_keys = store.completed_candidate_keys(manifest.run_id)
            remaining = max(0.0, progress.budget_seconds - progress.consumed_seconds)
            checkpoint_state: dict[str, JsonValue] = {
                **manifest.scheduler_state,
                "tracking": {
                    "checkpoint_revision": progress.checkpoint_revision,
                    "consumed_seconds": progress.consumed_seconds,
                    "effective_budget_seconds": progress.budget_seconds,
                },
            }
            manifest_value = manifest.to_json_value()
            manifest_value.update(
                {
                    "status": RunStatus.RUNNING.value,
                    "updated_at": max(
                        datetime.now(UTC), manifest.created_at, manifest.updated_at
                    ).isoformat(),
                    "completed_at": None,
                    "scheduler_state": checkpoint_state,
                }
            )
            resumed_manifest = RunManifest.model_validate(manifest_value)
            store.update_manifest(resumed_manifest)
            resumed_manifest.write_json(directory / MANIFEST_FILENAME)
            return ResumeState(
                run_directory=directory,
                manifest=resumed_manifest,
                terminal_trials=trials,
                retryable_trials=retryable_trials,
                completed_candidate_keys=completed_keys,
                interrupted_trial_ids=interrupted,
                component_states=states,
                progress=progress,
                remaining_budget_seconds=remaining,
                lease_owner=lease_owner,
            )
        except Exception as error:
            if lease_acquired and lease_owner is not None:
                with suppress(TrackingError):
                    store.release_run_lease(manifest.run_id, lease_owner)
            if isinstance(error, TrackingError | OSError):
                raise ResumeError("run state could not be reconciled safely") from error
            raise

    def release(self, state: ResumeState) -> None:
        """Release the run lease after the resumed controller stops or completes."""
        if state.lease_owner is None:
            return
        try:
            store = ExperimentStore.open(state.run_directory / REGISTRY_FILENAME)
            store.release_run_lease(state.manifest.run_id, state.lease_owner)
        except TrackingError as error:
            raise ResumeError("run lease could not be released") from error


def _validate_manifest_consistency(
    file_manifest: RunManifest,
    stored_manifest: RunManifest,
) -> None:
    immutable_pairs = (
        (file_manifest.run_id, stored_manifest.run_id),
        (file_manifest.configuration, stored_manifest.configuration),
        (file_manifest.source_hashes, stored_manifest.source_hashes),
        (file_manifest.dataset, stored_manifest.dataset),
        (file_manifest.dataset_profile.dataset_hash, stored_manifest.dataset_profile.dataset_hash),
        (file_manifest.validation_plan, stored_manifest.validation_plan),
        (file_manifest.random_seed, stored_manifest.random_seed),
    )
    if any(file_value != stored_value for file_value, stored_value in immutable_pairs):
        raise TrackingError("manifest and registry disagree on immutable run state")


def load_resume_manifest(run_directory: str | Path) -> RunManifest:
    """Read a strictly validated V1 manifest from an existing run directory."""
    directory = Path(run_directory).expanduser().resolve()
    if not directory.is_dir():
        raise ResumeError(f"run directory does not exist or is not a directory: {directory}")
    path = directory / MANIFEST_FILENAME
    if not path.is_file():
        raise ResumeError(f"run manifest is missing: {path}")
    try:
        return RunManifest.read_json(path)
    except (OSError, UnicodeError, ValueError, ValidationError) as error:
        raise ResumeError(f"run manifest is invalid or unreadable: {path}") from error


def verify_manifest_source_hashes(manifest: RunManifest) -> None:
    """Refuse resume if any explicitly recorded train or test source changed."""
    for source, expected_hash in manifest.source_hashes.items():
        path = Path(source).expanduser()
        if not path.is_file():
            raise ResumeError(f"recorded data source is missing: {path}")
        try:
            actual_hash = sha256_file(path)
        except OSError as error:
            raise ResumeError(f"recorded data source cannot be read: {path}") from error
        if actual_hash != expected_hash:
            raise ResumeError(f"recorded data source hash changed: {path}")


__all__ = [
    "MANIFEST_FILENAME",
    "REGISTRY_FILENAME",
    "ResumeManager",
    "ResumeState",
    "load_resume_manifest",
    "verify_manifest_source_hashes",
]

"""Transactional SQLite registry for runs, trials, folds, and checkpoints."""

from __future__ import annotations

import json
import math
import re
import secrets
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePath
from typing import cast

from autonomous_automl.contracts import (
    AutoMLConfig,
    DatasetBundle,
    DatasetProfile,
    FidelitySpec,
    FoldResult,
    LeakageReport,
    MetricName,
    PipelineSpec,
    RunManifest,
    RunStatus,
    TrialResult,
    TrialStatus,
    ValidationAudit,
    ValidationPlan,
)
from autonomous_automl.pipelines import pipeline_fingerprint
from autonomous_automl.tracking.migrations import (
    EXPECTED_INDEXES,
    EXPECTED_SCHEMA,
    LATEST_SCHEMA_VERSION,
    apply_migrations,
    current_schema_version,
)
from autonomous_automl.utils.errors import TrackingError
from autonomous_automl.utils.hashing import sha256_json
from autonomous_automl.utils.json import JsonValue, canonical_json_dumps

DEFAULT_REGISTRY_FILENAME = "registry.sqlite3"
_TERMINAL_TRIAL_STATUSES = {
    TrialStatus.COMPLETED,
    TrialStatus.FAILED,
    TrialStatus.PRUNED,
    TrialStatus.INTERRUPTED,
}


@dataclass(frozen=True, slots=True)
class TrialReservation:
    """Outcome of atomically attempting to claim one logical candidate."""

    trial_id: str
    candidate_key: str
    sequence_number: int
    reserved: bool
    status: TrialStatus
    attempt_count: int
    attempt_token: str
    lease_epoch: int


@dataclass(frozen=True, slots=True)
class RetryableTrial:
    """Interrupted reservation reconstructed without deserializing an estimator."""

    trial_id: str
    sequence_number: int
    pipeline_spec: PipelineSpec
    fidelity: FidelitySpec
    primary_metric: MetricName
    attempt_count: int


@dataclass(frozen=True, slots=True)
class RunProgress:
    """Small mutable-state projection used by budget and resume services."""

    run_id: str
    status: RunStatus
    budget_seconds: float
    consumed_seconds: float
    checkpoint_revision: int
    lease_epoch: int


@dataclass(frozen=True, slots=True)
class ArtifactRegistration:
    """Database representation of a locally persisted artifact."""

    artifact_id: int
    run_id: str
    name: str
    kind: str
    relative_path: str
    sha256: str
    size_bytes: int
    media_type: str | None


class ExperimentStore:
    """Short-lived-connection SQLite store safe for sequential process workers."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def open(cls, path: str | Path) -> ExperimentStore:
        """Create/open a registry and apply every known migration."""
        store = cls(path)
        store.migrate()
        return store

    def migrate(self) -> int:
        try:
            with closing(self._connect()) as connection:
                return apply_migrations(connection)
        except sqlite3.DatabaseError as error:
            raise TrackingError("registry database is corrupt or cannot be migrated") from error

    def integrity_check(self) -> None:
        """Fail explicitly on SQLite corruption, broken FKs, or unknown schema."""
        try:
            with closing(self._connect()) as connection:
                version = current_schema_version(connection)
                if version != LATEST_SCHEMA_VERSION:
                    raise TrackingError(
                        f"registry schema version {version} is not supported; "
                        f"expected {LATEST_SCHEMA_VERSION}"
                    )
                rows = connection.execute("PRAGMA integrity_check").fetchall()
                messages = [str(row[0]) for row in rows]
                if messages != ["ok"]:
                    raise TrackingError(f"registry integrity check failed: {messages}")
                foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
                if foreign_keys:
                    raise TrackingError("registry contains broken foreign-key references")
                tables = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }
                missing_tables = set(EXPECTED_SCHEMA).difference(tables)
                if missing_tables:
                    raise TrackingError(
                        f"registry schema is missing tables: {sorted(missing_tables)}"
                    )
                for table, expected_columns in EXPECTED_SCHEMA.items():
                    actual_columns = {
                        str(row[1])
                        for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
                    }
                    if actual_columns != set(expected_columns):
                        raise TrackingError(f"registry table schema is invalid: {table}")
                indexes = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'index'"
                    ).fetchall()
                }
                missing_indexes = set(EXPECTED_INDEXES).difference(indexes)
                if missing_indexes:
                    raise TrackingError(
                        f"registry schema is missing indexes: {sorted(missing_indexes)}"
                    )
                self._validate_persisted_contracts(connection)
        except sqlite3.DatabaseError as error:
            raise TrackingError("registry database is corrupt or unreadable") from error

    def create_run(self, manifest: RunManifest) -> None:
        """Persist a new run and all immutable planning contracts atomically."""
        manifest_json = manifest.to_json()
        now = _timestamp()
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT manifest_json FROM runs WHERE run_id = ?", (manifest.run_id,)
            ).fetchone()
            if existing is not None:
                if str(existing[0]) == manifest_json:
                    return
                raise TrackingError(
                    f"run_id already exists with different state: {manifest.run_id}"
                )
            connection.execute(
                """
                INSERT INTO runs(
                    run_id, status, manifest_version, package_version, manifest_json,
                    configuration_json, configuration_sha256, random_seed,
                    dataset_hash, source_hashes_json, created_at, updated_at,
                    completed_at, budget_seconds, consumed_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    manifest.run_id,
                    manifest.status.value,
                    manifest.manifest_version,
                    manifest.package_version,
                    manifest_json,
                    manifest.configuration.to_json(),
                    sha256_json(manifest.configuration.to_json_value()),
                    manifest.random_seed,
                    manifest.dataset_profile.dataset_hash,
                    canonical_json_dumps(cast(JsonValue, manifest.source_hashes)),
                    manifest.created_at.isoformat(),
                    manifest.updated_at.isoformat(),
                    None if manifest.completed_at is None else manifest.completed_at.isoformat(),
                    float(manifest.configuration.budget_seconds),
                ),
            )
            connection.execute(
                "INSERT INTO datasets VALUES (?, ?, ?, ?)",
                (
                    manifest.run_id,
                    manifest.dataset_profile.dataset_hash,
                    manifest.dataset.to_json(),
                    canonical_json_dumps(cast(JsonValue, manifest.source_hashes)),
                ),
            )
            connection.execute(
                "INSERT INTO profiles VALUES (?, ?, ?)",
                (
                    manifest.run_id,
                    manifest.dataset_profile.to_json(),
                    manifest.leakage_report.to_json(),
                ),
            )
            connection.execute(
                """
                INSERT INTO validation_plans(run_id, plan_json, audit_json)
                VALUES (?, ?, ?)
                """,
                (
                    manifest.run_id,
                    manifest.validation_plan.to_json(),
                    None
                    if manifest.validation_audit is None
                    else manifest.validation_audit.to_json(),
                ),
            )
            self._insert_event(
                connection,
                manifest.run_id,
                None,
                "info",
                "run_created",
                {"created_at": now},
            )

    def update_manifest(self, manifest: RunManifest) -> None:
        """Update the readable run projection without changing immutable sources."""
        with self._transaction() as connection:
            row = self._required_run_row(connection, manifest.run_id)
            if str(row["dataset_hash"]) != manifest.dataset_profile.dataset_hash:
                raise TrackingError("manifest dataset hash cannot change during a run")
            if _loads_json(str(row["source_hashes_json"])) != cast(
                JsonValue, manifest.source_hashes
            ):
                raise TrackingError("manifest source hashes cannot change during a run")
            connection.execute(
                """
                UPDATE runs SET status = ?, manifest_json = ?, updated_at = ?,
                    completed_at = ? WHERE run_id = ?
                """,
                (
                    manifest.status.value,
                    manifest.to_json(),
                    manifest.updated_at.isoformat(),
                    None if manifest.completed_at is None else manifest.completed_at.isoformat(),
                    manifest.run_id,
                ),
            )

    def get_manifest(self, run_id: str) -> RunManifest:
        with closing(self._connect()) as connection:
            row = self._required_run_row(connection, run_id)
            try:
                return RunManifest.from_json(str(row["manifest_json"]))
            except ValueError as error:
                raise TrackingError("stored run manifest is invalid") from error

    def reserve_trial(
        self,
        run_id: str,
        trial_id: str,
        spec: PipelineSpec,
        fidelity: FidelitySpec,
        primary_metric: MetricName,
    ) -> TrialReservation:
        """Claim a candidate once, or reactivate its interrupted logical trial."""
        if not trial_id.strip():
            raise ValueError("trial_id cannot be blank")
        spec_key = pipeline_fingerprint(spec)
        fidelity_key = sha256_json(fidelity.to_json_value())
        candidate_key = sha256_json(
            {"fidelity_fingerprint": fidelity_key, "spec_fingerprint": spec_key}
        )
        now = _timestamp()
        with self._transaction() as connection:
            run = self._required_run_row(connection, run_id)
            if RunStatus(str(run["status"])) not in {RunStatus.CREATED, RunStatus.RUNNING}:
                raise TrackingError("trials can only be reserved for an active run")
            connection.execute(
                """
                INSERT INTO pipeline_specs(
                    run_id, spec_fingerprint, family, model_name, spec_json
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id, spec_fingerprint) DO NOTHING
                """,
                (run_id, spec_key, spec.family, spec.model_name, spec.to_json()),
            )
            stored_spec = connection.execute(
                "SELECT spec_json FROM pipeline_specs WHERE run_id = ? AND spec_fingerprint = ?",
                (run_id, spec_key),
            ).fetchone()
            if stored_spec is None or str(stored_spec[0]) != spec.to_json():
                raise TrackingError("pipeline fingerprint collision or corrupted specification")

            existing = connection.execute(
                """
                SELECT trial_id, sequence_number, status, attempt_count, attempt_token
                FROM trials WHERE run_id = ? AND candidate_key = ?
                """,
                (run_id, candidate_key),
            ).fetchone()
            if existing is not None:
                status = TrialStatus(str(existing["status"]))
                existing_id = str(existing["trial_id"])
                attempts = int(existing["attempt_count"])
                if status is TrialStatus.INTERRUPTED:
                    attempts += 1
                    attempt_token = secrets.token_hex(16)
                    connection.execute(
                        """
                        UPDATE trials SET status = 'running', attempt_count = ?,
                            started_at = ?, finished_at = NULL, heartbeat_at = ?,
                            failure_type = NULL, failure_message = NULL, result_json = NULL,
                            attempt_token = ?
                        WHERE trial_id = ?
                        """,
                        (attempts, now, now, attempt_token, existing_id),
                    )
                    self._insert_event(
                        connection,
                        run_id,
                        existing_id,
                        "info",
                        "trial_resumed",
                        {"attempt_count": attempts},
                    )
                    return TrialReservation(
                        existing_id,
                        candidate_key,
                        int(existing["sequence_number"]),
                        True,
                        TrialStatus.RUNNING,
                        attempts,
                        attempt_token,
                        int(run["lease_epoch"]),
                    )
                return TrialReservation(
                    existing_id,
                    candidate_key,
                    int(existing["sequence_number"]),
                    False,
                    status,
                    attempts,
                    str(existing["attempt_token"] or ""),
                    int(run["lease_epoch"]),
                )

            collision = connection.execute(
                "SELECT candidate_key FROM trials WHERE trial_id = ?", (trial_id,)
            ).fetchone()
            if collision is not None:
                raise TrackingError(f"trial_id already belongs to another candidate: {trial_id}")
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence_number), -1) + 1 FROM trials WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            sequence_number = int(row[0])
            attempt_token = secrets.token_hex(16)
            connection.execute(
                """
                INSERT INTO trials(
                    trial_id, run_id, sequence_number, candidate_key, spec_fingerprint,
                    fidelity_fingerprint, fidelity_json, family, status, primary_metric,
                    started_at, heartbeat_at, attempt_token
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?, ?)
                """,
                (
                    trial_id,
                    run_id,
                    sequence_number,
                    candidate_key,
                    spec_key,
                    fidelity_key,
                    fidelity.to_json(),
                    spec.family,
                    primary_metric.value,
                    now,
                    now,
                    attempt_token,
                ),
            )
            self._insert_event(
                connection,
                run_id,
                trial_id,
                "info",
                "trial_reserved",
                {"candidate_key": candidate_key, "sequence_number": sequence_number},
            )
            return TrialReservation(
                trial_id,
                candidate_key,
                sequence_number,
                True,
                TrialStatus.RUNNING,
                1,
                attempt_token,
                int(run["lease_epoch"]),
            )

    def record_trial_result(
        self,
        result: TrialResult,
        *,
        attempt_token: str,
        lease_epoch: int,
        component_states: Mapping[str, JsonValue] | None = None,
        consumed_seconds: float = 0.0,
    ) -> int:
        """Commit result, folds, snapshots, budget, revision, and event together."""
        if result.status not in _TERMINAL_TRIAL_STATUSES:
            raise TrackingError("only terminal trial results can be recorded")
        if not attempt_token:
            raise TrackingError("attempt_token is required to fence stale workers")
        if not math.isfinite(consumed_seconds) or consumed_seconds < 0:
            raise ValueError("consumed_seconds must be finite and non-negative")
        snapshots = dict(component_states or {})
        now = _timestamp()
        with self._transaction() as connection:
            trial = connection.execute(
                "SELECT * FROM trials WHERE trial_id = ?", (result.trial_id,)
            ).fetchone()
            if trial is None:
                raise TrackingError(f"trial reservation does not exist: {result.trial_id}")
            run_id = str(trial["run_id"])
            run = self._required_run_row(connection, run_id)
            if str(trial["attempt_token"] or "") != attempt_token:
                raise TrackingError("trial attempt token is stale or invalid")
            if int(run["lease_epoch"]) != lease_epoch:
                raise TrackingError("run lease epoch is stale or invalid")
            existing_result = trial["result_json"]
            if existing_result is not None:
                if str(existing_result) == result.to_json():
                    progress = self._required_run_row(connection, run_id)
                    return int(progress["checkpoint_revision"])
                raise TrackingError("terminal trial result cannot be overwritten")
            if TrialStatus(str(trial["status"])) is not TrialStatus.RUNNING:
                raise TrackingError("invalid trial state transition to terminal result")
            stored_spec = connection.execute(
                """
                SELECT spec_json FROM pipeline_specs
                WHERE run_id = ? AND spec_fingerprint = ?
                """,
                (run_id, str(trial["spec_fingerprint"])),
            ).fetchone()
            if stored_spec is None or str(stored_spec[0]) != result.pipeline_spec.to_json():
                raise TrackingError("trial result PipelineSpec differs from its reservation")
            if str(trial["fidelity_json"]) != result.fidelity.to_json():
                raise TrackingError("trial result fidelity differs from its reservation")

            connection.execute(
                """
                UPDATE trials SET status = ?, primary_metric = ?, mean_score = ?,
                    std_score = ?, fold_scores_json = ?, fit_seconds = ?,
                    predict_seconds = ?, peak_memory_mb = ?, failure_type = ?,
                    failure_message = ?, finished_at = ?, heartbeat_at = ?, result_json = ?
                WHERE trial_id = ?
                """,
                (
                    result.status.value,
                    result.primary_metric.value,
                    result.mean_score,
                    result.std_score,
                    canonical_json_dumps(cast(JsonValue, result.fold_scores)),
                    result.fit_seconds,
                    result.predict_seconds,
                    result.peak_memory_mb,
                    result.failure_type,
                    result.failure_message,
                    None if result.finished_at is None else result.finished_at.isoformat(),
                    now,
                    result.to_json(),
                    result.trial_id,
                ),
            )
            connection.execute("DELETE FROM fold_results WHERE trial_id = ?", (result.trial_id,))
            for order, fold in enumerate(result.fold_results):
                connection.execute(
                    """
                    INSERT INTO fold_results(
                        trial_id, result_order, fold_index, seed, score,
                        user_metric_value, fit_seconds, predict_seconds,
                        n_train_rows, n_validation_rows, result_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.trial_id,
                        order,
                        fold.fold_index,
                        fold.seed,
                        fold.score,
                        fold.user_metric_value,
                        fold.fit_seconds,
                        fold.predict_seconds,
                        fold.n_train_rows,
                        fold.n_validation_rows,
                        fold.to_json(),
                    ),
                )

            revision = int(run["checkpoint_revision"]) + 1
            complete_state = self._checkpoint_state(
                connection,
                run_id,
                int(run["checkpoint_revision"]),
            )
            for component, state in sorted(snapshots.items()):
                if not component.strip():
                    raise TrackingError("snapshot component name cannot be blank")
                state_json = canonical_json_dumps(state)
                complete_state[component] = state
                connection.execute(
                    """
                    INSERT INTO scheduler_state(
                        run_id, component, revision, state_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, component) DO UPDATE SET
                        revision = excluded.revision,
                        state_json = excluded.state_json,
                        updated_at = excluded.updated_at
                    """,
                    (run_id, component, revision, state_json, now),
                )
            complete_state_json = canonical_json_dumps(cast(JsonValue, complete_state))
            connection.execute(
                """
                INSERT INTO checkpoint_snapshots(run_id, revision, state_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, revision, complete_state_json, now),
            )
            connection.execute(
                """
                UPDATE runs SET checkpoint_revision = ?,
                    consumed_seconds = consumed_seconds + ?, updated_at = ?
                WHERE run_id = ?
                """,
                (revision, consumed_seconds, now, run_id),
            )
            self._insert_event(
                connection,
                run_id,
                result.trial_id,
                "error" if result.status is TrialStatus.FAILED else "info",
                "trial_finished",
                {"revision": revision, "status": result.status.value},
            )
            return revision

    def get_trial(self, trial_id: str) -> TrialResult | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT result_json FROM trials WHERE trial_id = ?", (trial_id,)
            ).fetchone()
            if row is None or row[0] is None:
                return None
            try:
                return TrialResult.from_json(str(row[0]))
            except ValueError as error:
                raise TrackingError("stored trial result is invalid") from error

    def list_trials(self, run_id: str) -> list[TrialResult]:
        with closing(self._connect()) as connection:
            self._required_run_row(connection, run_id)
            rows = connection.execute(
                """
                SELECT result_json FROM trials
                WHERE run_id = ? AND result_json IS NOT NULL
                ORDER BY sequence_number
                """,
                (run_id,),
            ).fetchall()
        try:
            return [TrialResult.from_json(str(row[0])) for row in rows]
        except ValueError as error:
            raise TrackingError("stored trial result is invalid") from error

    def list_retryable_trials(self, run_id: str) -> list[RetryableTrial]:
        """Return interrupted logical trials with their reconstructible contracts."""

        with closing(self._connect()) as connection:
            self._required_run_row(connection, run_id)
            rows = connection.execute(
                """
                SELECT t.trial_id, t.sequence_number, t.fidelity_json,
                    t.primary_metric, t.attempt_count, p.spec_json
                FROM trials AS t
                JOIN pipeline_specs AS p
                  ON p.run_id = t.run_id
                 AND p.spec_fingerprint = t.spec_fingerprint
                WHERE t.run_id = ? AND t.status = 'interrupted'
                ORDER BY t.sequence_number
                """,
                (run_id,),
            ).fetchall()
        retryable: list[RetryableTrial] = []
        try:
            for row in rows:
                specification = PipelineSpec.from_json(str(row["spec_json"]))
                fidelity = FidelitySpec.from_json(str(row["fidelity_json"]))
                retryable.append(
                    RetryableTrial(
                        trial_id=str(row["trial_id"]),
                        sequence_number=int(row["sequence_number"]),
                        pipeline_spec=specification,
                        fidelity=fidelity,
                        primary_metric=MetricName(str(row["primary_metric"])),
                        attempt_count=int(row["attempt_count"]),
                    )
                )
        except (TypeError, ValueError) as error:
            raise TrackingError("interrupted trial contracts are invalid") from error
        return retryable

    def completed_candidate_keys(self, run_id: str) -> set[str]:
        with closing(self._connect()) as connection:
            self._required_run_row(connection, run_id)
            rows = connection.execute(
                "SELECT candidate_key FROM trials WHERE run_id = ? AND status = 'completed'",
                (run_id,),
            ).fetchall()
        return {str(row[0]) for row in rows}

    def load_component_states(self, run_id: str) -> dict[str, JsonValue]:
        with closing(self._connect()) as connection:
            run = self._required_run_row(connection, run_id)
            return self._checkpoint_state(
                connection,
                run_id,
                int(run["checkpoint_revision"]),
            )

    def get_run_progress(self, run_id: str) -> RunProgress:
        with closing(self._connect()) as connection:
            row = self._required_run_row(connection, run_id)
            return RunProgress(
                run_id=run_id,
                status=RunStatus(str(row["status"])),
                budget_seconds=float(row["budget_seconds"]),
                consumed_seconds=float(row["consumed_seconds"]),
                checkpoint_revision=int(row["checkpoint_revision"]),
                lease_epoch=int(row["lease_epoch"]),
            )

    def add_budget(self, run_id: str, additional_seconds: float) -> RunProgress:
        if not math.isfinite(additional_seconds) or additional_seconds <= 0:
            raise ValueError("additional_seconds must be finite and positive")
        with self._transaction() as connection:
            self._required_run_row(connection, run_id)
            connection.execute(
                "UPDATE runs SET budget_seconds = budget_seconds + ?, updated_at = ? "
                "WHERE run_id = ?",
                (additional_seconds, _timestamp(), run_id),
            )
        return self.get_run_progress(run_id)

    def synchronize_consumed_seconds(
        self,
        run_id: str,
        consumed_seconds: float,
    ) -> RunProgress:
        """Advance the global consumed budget without ever moving it backwards."""

        if not math.isfinite(consumed_seconds) or consumed_seconds < 0:
            raise ValueError("consumed_seconds must be finite and non-negative")
        with self._transaction() as connection:
            run = self._required_run_row(connection, run_id)
            bounded = min(float(run["budget_seconds"]), consumed_seconds)
            connection.execute(
                """
                UPDATE runs SET consumed_seconds = MAX(consumed_seconds, ?), updated_at = ?
                WHERE run_id = ?
                """,
                (bounded, _timestamp(), run_id),
            )
        return self.get_run_progress(run_id)

    def prepare_resume(
        self,
        run_id: str,
        *,
        operation_id: str,
        additional_budget_seconds: float = 0.0,
        stale_after_seconds: float = 0.0,
        lease_owner: str | None = None,
        lease_epoch: int | None = None,
    ) -> tuple[list[str], RunProgress]:
        """Atomically fence a resume, interrupt stale trials, and apply budget once."""
        if not operation_id.strip():
            raise ValueError("resume operation_id cannot be blank")
        if not math.isfinite(additional_budget_seconds) or additional_budget_seconds < 0:
            raise ValueError("additional resume budget must be finite and non-negative")
        if not math.isfinite(stale_after_seconds) or stale_after_seconds < 0:
            raise ValueError("stale_after_seconds must be finite and non-negative")
        cutoff = (datetime.now(UTC) - timedelta(seconds=stale_after_seconds)).isoformat()
        now = _timestamp()
        with self._transaction() as connection:
            run = self._required_run_row(connection, run_id)
            if lease_owner is not None and (
                str(run["lease_owner"] or "") != lease_owner
                or lease_epoch is None
                or int(run["lease_epoch"]) != lease_epoch
            ):
                raise TrackingError("resume operation has a stale run lease")
            operation = connection.execute(
                """
                SELECT additional_budget_seconds FROM resume_operations
                WHERE operation_id = ? AND run_id = ?
                """,
                (operation_id, run_id),
            ).fetchone()
            if operation is None:
                connection.execute(
                    """
                    INSERT INTO resume_operations(
                        operation_id, run_id, additional_budget_seconds, applied_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (operation_id, run_id, additional_budget_seconds, now),
                )
                budget_increment = additional_budget_seconds
            else:
                if float(operation[0]) != additional_budget_seconds:
                    raise TrackingError("resume operation was reused with a different budget")
                budget_increment = 0.0
            rows = connection.execute(
                """
                SELECT trial_id FROM trials
                WHERE run_id = ? AND status = 'running'
                    AND COALESCE(heartbeat_at, started_at) <= ?
                ORDER BY sequence_number
                """,
                (run_id, cutoff),
            ).fetchall()
            interrupted = [str(row[0]) for row in rows]
            for trial_id in interrupted:
                connection.execute(
                    """
                    UPDATE trials SET status = 'interrupted', finished_at = ?,
                        failure_type = 'InterruptedTrial',
                        failure_message = 'trial process stopped before completion'
                    WHERE trial_id = ?
                    """,
                    (now, trial_id),
                )
            connection.execute(
                """
                UPDATE runs SET status = 'running',
                    budget_seconds = budget_seconds + ?, updated_at = ?
                WHERE run_id = ?
                """,
                (budget_increment, now, run_id),
            )
            self._insert_event(
                connection,
                run_id,
                None,
                "info",
                "resume_prepared",
                {
                    "interrupted_trial_count": len(interrupted),
                    "operation_id": operation_id,
                },
            )
            updated = self._required_run_row(connection, run_id)
            progress = RunProgress(
                run_id=run_id,
                status=RunStatus(str(updated["status"])),
                budget_seconds=float(updated["budget_seconds"]),
                consumed_seconds=float(updated["consumed_seconds"]),
                checkpoint_revision=int(updated["checkpoint_revision"]),
                lease_epoch=int(updated["lease_epoch"]),
            )
            return interrupted, progress

    def set_run_status(self, run_id: str, status: RunStatus) -> None:
        completed_at = _timestamp() if status is RunStatus.COMPLETED else None
        with self._transaction() as connection:
            self._required_run_row(connection, run_id)
            connection.execute(
                "UPDATE runs SET status = ?, updated_at = ?, completed_at = ? WHERE run_id = ?",
                (status.value, _timestamp(), completed_at, run_id),
            )
            self._insert_event(
                connection,
                run_id,
                None,
                "info",
                "run_status_changed",
                {"status": status.value},
            )

    def mark_stale_trials_interrupted(
        self,
        run_id: str,
        *,
        stale_after_seconds: float = 0.0,
    ) -> list[str]:
        if not math.isfinite(stale_after_seconds) or stale_after_seconds < 0:
            raise ValueError("stale_after_seconds must be finite and non-negative")
        cutoff = (datetime.now(UTC) - timedelta(seconds=stale_after_seconds)).isoformat()
        now = _timestamp()
        with self._transaction() as connection:
            self._required_run_row(connection, run_id)
            rows = connection.execute(
                """
                SELECT trial_id FROM trials
                WHERE run_id = ? AND status = 'running'
                    AND COALESCE(heartbeat_at, started_at) <= ?
                ORDER BY sequence_number
                """,
                (run_id, cutoff),
            ).fetchall()
            trial_ids = [str(row[0]) for row in rows]
            for trial_id in trial_ids:
                connection.execute(
                    """
                    UPDATE trials SET status = 'interrupted', finished_at = ?,
                        failure_type = 'InterruptedTrial',
                        failure_message = 'trial process stopped before completion'
                    WHERE trial_id = ?
                    """,
                    (now, trial_id),
                )
                self._insert_event(
                    connection,
                    run_id,
                    trial_id,
                    "warning",
                    "trial_interrupted",
                    {},
                )
            return trial_ids

    def heartbeat_trial(
        self,
        trial_id: str,
        *,
        attempt_token: str,
        lease_epoch: int,
    ) -> None:
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE trials SET heartbeat_at = ?
                WHERE trial_id = ? AND status = 'running' AND attempt_token = ?
                  AND EXISTS (
                      SELECT 1 FROM runs
                      WHERE runs.run_id = trials.run_id AND lease_epoch = ?
                  )
                """,
                (_timestamp(), trial_id, attempt_token, lease_epoch),
            )
            if cursor.rowcount != 1:
                raise TrackingError("cannot heartbeat a missing or non-running trial")

    def acquire_run_lease(self, run_id: str, owner: str, ttl_seconds: int = 60) -> bool:
        if not owner.strip() or ttl_seconds <= 0:
            raise ValueError("lease owner and positive ttl_seconds are required")
        now = datetime.now(UTC)
        expires = (now + timedelta(seconds=ttl_seconds)).isoformat()
        with self._transaction() as connection:
            row = self._required_run_row(connection, run_id)
            current_owner = row["lease_owner"]
            current_expiry = row["lease_expires_at"]
            available = (
                current_owner is None
                or current_owner == owner
                or current_expiry is None
                or str(current_expiry) <= now.isoformat()
            )
            if not available:
                return False
            current_epoch = int(row["lease_epoch"])
            same_active_owner = (
                current_owner == owner
                and current_expiry is not None
                and str(current_expiry) > now.isoformat()
            )
            new_epoch = current_epoch if same_active_owner else current_epoch + 1
            connection.execute(
                """
                UPDATE runs SET lease_owner = ?, lease_expires_at = ?, heartbeat_at = ?,
                    lease_epoch = ?
                WHERE run_id = ?
                """,
                (owner, expires, now.isoformat(), new_epoch, run_id),
            )
            return True

    def renew_run_lease(self, run_id: str, owner: str, ttl_seconds: int = 60) -> None:
        if not owner.strip() or ttl_seconds <= 0:
            raise ValueError("lease owner and positive ttl_seconds are required")
        now = datetime.now(UTC)
        expires = (now + timedelta(seconds=ttl_seconds)).isoformat()
        with self._transaction() as connection:
            self._required_run_row(connection, run_id)
            cursor = connection.execute(
                """
                UPDATE runs SET lease_expires_at = ?, heartbeat_at = ?
                WHERE run_id = ? AND lease_owner = ?
                """,
                (expires, now.isoformat(), run_id, owner),
            )
            if cursor.rowcount != 1:
                raise TrackingError("run lease is not owned by this process")

    def release_run_lease(self, run_id: str, owner: str) -> None:
        with self._transaction() as connection:
            self._required_run_row(connection, run_id)
            cursor = connection.execute(
                """
                UPDATE runs SET lease_owner = NULL, lease_expires_at = NULL,
                    heartbeat_at = NULL WHERE run_id = ? AND lease_owner = ?
                """,
                (run_id, owner),
            )
            if cursor.rowcount != 1:
                raise TrackingError("run lease is not owned by this process")

    def register_artifact(
        self,
        run_id: str,
        *,
        name: str,
        kind: str,
        relative_path: str,
        sha256: str,
        size_bytes: int,
        media_type: str | None = None,
        metadata: JsonValue | None = None,
        trial_id: str | None = None,
    ) -> ArtifactRegistration:
        if not name.strip() or not kind.strip() or not relative_path.strip():
            raise ValueError("artifact name, kind, and relative_path are required")
        artifact_path = Path(relative_path)
        if artifact_path.is_absolute() or any(
            part == ".." for part in PurePath(artifact_path).parts
        ):
            raise ValueError("registered artifact path must be relative and confined")
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError("artifact sha256 must be a lowercase digest")
        if size_bytes < 0:
            raise ValueError("artifact size_bytes must be non-negative")
        with self._transaction() as connection:
            self._required_run_row(connection, run_id)
            existing = connection.execute(
                """
                SELECT artifact_id, run_id, name, kind, relative_path, sha256,
                       size_bytes, media_type
                FROM artifacts
                WHERE run_id = ? AND (name = ? OR relative_path = ?)
                """,
                (run_id, name, relative_path),
            ).fetchone()
            if existing is not None:
                registration = self._artifact_from_row(existing)
                expected = ArtifactRegistration(
                    registration.artifact_id,
                    run_id,
                    name,
                    kind,
                    relative_path,
                    sha256,
                    size_bytes,
                    media_type,
                )
                if registration == expected:
                    return registration
                raise TrackingError("artifact name or path is registered with different bytes")
            try:
                cursor = connection.execute(
                    """
                    INSERT INTO artifacts(
                        run_id, trial_id, name, kind, relative_path, sha256,
                        size_bytes, media_type, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        trial_id,
                        name,
                        kind,
                        relative_path,
                        sha256,
                        size_bytes,
                        media_type,
                        canonical_json_dumps(metadata if metadata is not None else {}),
                        _timestamp(),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise TrackingError("artifact name or path is already registered") from error
            if cursor.lastrowid is None:
                raise TrackingError("artifact registration did not return an identifier")
            artifact_id = int(cursor.lastrowid)
            return ArtifactRegistration(
                artifact_id,
                run_id,
                name,
                kind,
                relative_path,
                sha256,
                size_bytes,
                media_type,
            )

    def list_artifacts(self, run_id: str) -> list[ArtifactRegistration]:
        with closing(self._connect()) as connection:
            self._required_run_row(connection, run_id)
            rows = connection.execute(
                """
                SELECT artifact_id, run_id, name, kind, relative_path, sha256,
                       size_bytes, media_type
                FROM artifacts WHERE run_id = ? ORDER BY artifact_id
                """,
                (run_id,),
            ).fetchall()
        return [self._artifact_from_row(row) for row in rows]

    def get_artifact(self, run_id: str, name: str) -> ArtifactRegistration:
        with closing(self._connect()) as connection:
            self._required_run_row(connection, run_id)
            row = connection.execute(
                """
                SELECT artifact_id, run_id, name, kind, relative_path, sha256,
                       size_bytes, media_type
                FROM artifacts WHERE run_id = ? AND name = ?
                """,
                (run_id, name),
            ).fetchone()
        if row is None:
            raise TrackingError(f"artifact is not registered: {name}")
        return self._artifact_from_row(row)

    def checkpoint(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def _connect(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(
                self.path,
                timeout=30.0,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("PRAGMA journal_mode = WAL")
            return connection
        except sqlite3.DatabaseError as error:
            raise TrackingError("registry database could not be opened") from error

    def _transaction(self) -> _Transaction:
        return _Transaction(self._connect())

    @staticmethod
    def _required_run_row(connection: sqlite3.Connection, run_id: str) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise TrackingError(f"run is not registered: {run_id}")
        return cast(sqlite3.Row, row)

    @staticmethod
    def _artifact_from_row(row: sqlite3.Row) -> ArtifactRegistration:
        return ArtifactRegistration(
            int(row["artifact_id"]),
            str(row["run_id"]),
            str(row["name"]),
            str(row["kind"]),
            str(row["relative_path"]),
            str(row["sha256"]),
            int(row["size_bytes"]),
            None if row["media_type"] is None else str(row["media_type"]),
        )

    @staticmethod
    def _validate_persisted_contracts(connection: sqlite3.Connection) -> None:
        try:
            for row in connection.execute(
                "SELECT manifest_json, configuration_json, configuration_sha256 FROM runs"
            ):
                RunManifest.from_json(str(row[0]))
                configuration = AutoMLConfig.from_json(str(row[1]))
                if sha256_json(configuration.to_json_value()) != str(row[2]):
                    raise TrackingError("stored configuration checksum is invalid")
            for row in connection.execute("SELECT bundle_json FROM datasets"):
                DatasetBundle.from_json(str(row[0]))
            for row in connection.execute("SELECT profile_json, leakage_report_json FROM profiles"):
                DatasetProfile.from_json(str(row[0]))
                LeakageReport.from_json(str(row[1]))
            for row in connection.execute("SELECT plan_json, audit_json FROM validation_plans"):
                ValidationPlan.from_json(str(row[0]))
                if row[1] is not None:
                    ValidationAudit.from_json(str(row[1]))
            for row in connection.execute("SELECT spec_fingerprint, spec_json FROM pipeline_specs"):
                spec = PipelineSpec.from_json(str(row[1]))
                if pipeline_fingerprint(spec) != str(row[0]):
                    raise TrackingError("stored PipelineSpec fingerprint is invalid")
            for row in connection.execute("SELECT fidelity_json, result_json FROM trials"):
                FidelitySpec.from_json(str(row[0]))
                if row[1] is not None:
                    TrialResult.from_json(str(row[1]))
            for row in connection.execute("SELECT result_json FROM fold_results"):
                FoldResult.from_json(str(row[0]))
            for row in connection.execute("SELECT state_json FROM checkpoint_snapshots"):
                state = _loads_json(str(row[0]))
                if not isinstance(state, dict):
                    raise TrackingError("checkpoint snapshot must be a JSON object")
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise TrackingError("registry contains an invalid persisted contract") from error

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        run_id: str,
        trial_id: str | None,
        level: str,
        event_type: str,
        payload: JsonValue,
    ) -> None:
        connection.execute(
            """
            INSERT INTO events(
                run_id, trial_id, occurred_at, level, event_type, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                trial_id,
                _timestamp(),
                level,
                event_type,
                canonical_json_dumps(payload),
            ),
        )

    @staticmethod
    def _checkpoint_state(
        connection: sqlite3.Connection,
        run_id: str,
        revision: int,
    ) -> dict[str, JsonValue]:
        if revision == 0:
            return {}
        row = connection.execute(
            """
            SELECT state_json FROM checkpoint_snapshots
            WHERE run_id = ? AND revision = ?
            """,
            (run_id, revision),
        ).fetchone()
        if row is None:
            raise TrackingError("checkpoint snapshot is missing for the current revision")
        try:
            state = _loads_json(str(row[0]))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise TrackingError("checkpoint snapshot is invalid JSON") from error
        if not isinstance(state, dict):
            raise TrackingError("checkpoint snapshot must be a JSON object")
        return state


class _Transaction:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def __enter__(self) -> sqlite3.Connection:
        self.connection.execute("BEGIN IMMEDIATE")
        return self.connection

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> None:
        try:
            if exception_type is None:
                self.connection.commit()
            else:
                self.connection.rollback()
        finally:
            self.connection.close()


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _loads_json(payload: str) -> JsonValue:
    value = json.loads(payload, parse_constant=_reject_json_constant)
    typed = cast(JsonValue, value)
    canonical_json_dumps(typed)
    return typed


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid non-finite JSON constant: {value}")


__all__ = [
    "DEFAULT_REGISTRY_FILENAME",
    "ArtifactRegistration",
    "ExperimentStore",
    "RetryableTrial",
    "RunProgress",
    "TrialReservation",
]

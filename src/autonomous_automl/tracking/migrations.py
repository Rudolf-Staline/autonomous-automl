"""Transactional, idempotent SQLite schema migrations."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Migration:
    """One monotonically versioned database migration."""

    version: int
    sql: str


_INITIAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (
        status IN ('created', 'running', 'completed', 'failed', 'interrupted')
    ),
    manifest_version INTEGER NOT NULL,
    package_version TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    configuration_json TEXT NOT NULL,
    configuration_sha256 TEXT NOT NULL,
    random_seed INTEGER NOT NULL,
    dataset_hash TEXT NOT NULL,
    source_hashes_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    checkpoint_revision INTEGER NOT NULL DEFAULT 0 CHECK (checkpoint_revision >= 0),
    budget_seconds REAL NOT NULL CHECK (budget_seconds > 0),
    consumed_seconds REAL NOT NULL DEFAULT 0 CHECK (consumed_seconds >= 0),
    lease_owner TEXT,
    lease_expires_at TEXT,
    heartbeat_at TEXT
);

CREATE TABLE IF NOT EXISTS datasets (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    dataset_hash TEXT NOT NULL,
    bundle_json TEXT NOT NULL,
    source_hashes_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS profiles (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    profile_json TEXT NOT NULL,
    leakage_report_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS validation_plans (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    plan_json TEXT NOT NULL,
    audit_json TEXT
);

CREATE TABLE IF NOT EXISTS pipeline_specs (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    spec_fingerprint TEXT NOT NULL,
    family TEXT NOT NULL,
    model_name TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    PRIMARY KEY (run_id, spec_fingerprint)
);

CREATE TABLE IF NOT EXISTS trials (
    trial_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    sequence_number INTEGER NOT NULL CHECK (sequence_number >= 0),
    candidate_key TEXT NOT NULL,
    spec_fingerprint TEXT NOT NULL,
    fidelity_fingerprint TEXT NOT NULL,
    fidelity_json TEXT NOT NULL,
    family TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'running', 'completed', 'failed', 'pruned', 'interrupted')
    ),
    primary_metric TEXT NOT NULL,
    mean_score REAL,
    std_score REAL,
    fold_scores_json TEXT,
    fit_seconds REAL NOT NULL DEFAULT 0 CHECK (fit_seconds >= 0),
    predict_seconds REAL NOT NULL DEFAULT 0 CHECK (predict_seconds >= 0),
    peak_memory_mb REAL,
    failure_type TEXT,
    failure_message TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    heartbeat_at TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 1 CHECK (attempt_count >= 1),
    result_json TEXT,
    UNIQUE (run_id, sequence_number),
    UNIQUE (run_id, candidate_key),
    FOREIGN KEY (run_id, spec_fingerprint)
        REFERENCES pipeline_specs(run_id, spec_fingerprint)
);

CREATE TABLE IF NOT EXISTS fold_results (
    trial_id TEXT NOT NULL REFERENCES trials(trial_id) ON DELETE CASCADE,
    result_order INTEGER NOT NULL CHECK (result_order >= 0),
    fold_index INTEGER NOT NULL CHECK (fold_index >= 0),
    seed INTEGER,
    score REAL NOT NULL,
    user_metric_value REAL NOT NULL,
    fit_seconds REAL NOT NULL CHECK (fit_seconds >= 0),
    predict_seconds REAL NOT NULL CHECK (predict_seconds >= 0),
    n_train_rows INTEGER NOT NULL CHECK (n_train_rows > 0),
    n_validation_rows INTEGER NOT NULL CHECK (n_validation_rows > 0),
    result_json TEXT NOT NULL,
    PRIMARY KEY (trial_id, result_order)
);

CREATE TABLE IF NOT EXISTS scheduler_state (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    component TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision >= 0),
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, component)
);

CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    trial_id TEXT REFERENCES trials(trial_id) ON DELETE CASCADE,
    occurred_at TEXT NOT NULL,
    level TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    trial_id TEXT REFERENCES trials(trial_id) ON DELETE SET NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    media_type TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (run_id, name),
    UNIQUE (run_id, relative_path)
);

CREATE INDEX IF NOT EXISTS trials_run_status ON trials(run_id, status);
CREATE INDEX IF NOT EXISTS trials_run_family_score ON trials(run_id, family, mean_score);
CREATE INDEX IF NOT EXISTS events_run_event ON events(run_id, event_id);
CREATE INDEX IF NOT EXISTS artifacts_run_kind ON artifacts(run_id, kind);
"""

_FENCING_AND_CHECKPOINT_SCHEMA = """
ALTER TABLE runs ADD COLUMN lease_epoch INTEGER NOT NULL DEFAULT 0;
ALTER TABLE trials ADD COLUMN attempt_token TEXT;

CREATE TABLE checkpoint_snapshots (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    revision INTEGER NOT NULL CHECK (revision >= 0),
    state_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, revision)
);

CREATE TABLE resume_operations (
    operation_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    additional_budget_seconds REAL NOT NULL CHECK (additional_budget_seconds >= 0),
    applied_at TEXT NOT NULL
);

CREATE INDEX checkpoint_snapshots_run_revision
    ON checkpoint_snapshots(run_id, revision);
CREATE INDEX resume_operations_run
    ON resume_operations(run_id, applied_at);
"""

MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, _INITIAL_SCHEMA),
    Migration(2, _FENCING_AND_CHECKPOINT_SCHEMA),
)
LATEST_SCHEMA_VERSION = MIGRATIONS[-1].version

EXPECTED_SCHEMA: dict[str, frozenset[str]] = {
    "schema_migrations": frozenset({"version", "applied_at"}),
    "runs": frozenset(
        {
            "run_id",
            "status",
            "manifest_version",
            "package_version",
            "manifest_json",
            "configuration_json",
            "configuration_sha256",
            "random_seed",
            "dataset_hash",
            "source_hashes_json",
            "created_at",
            "updated_at",
            "completed_at",
            "checkpoint_revision",
            "budget_seconds",
            "consumed_seconds",
            "lease_owner",
            "lease_expires_at",
            "heartbeat_at",
            "lease_epoch",
        }
    ),
    "datasets": frozenset({"run_id", "dataset_hash", "bundle_json", "source_hashes_json"}),
    "profiles": frozenset({"run_id", "profile_json", "leakage_report_json"}),
    "validation_plans": frozenset({"run_id", "plan_json", "audit_json"}),
    "pipeline_specs": frozenset(
        {"run_id", "spec_fingerprint", "family", "model_name", "spec_json"}
    ),
    "trials": frozenset(
        {
            "trial_id",
            "run_id",
            "sequence_number",
            "candidate_key",
            "spec_fingerprint",
            "fidelity_fingerprint",
            "fidelity_json",
            "family",
            "status",
            "primary_metric",
            "mean_score",
            "std_score",
            "fold_scores_json",
            "fit_seconds",
            "predict_seconds",
            "peak_memory_mb",
            "failure_type",
            "failure_message",
            "started_at",
            "finished_at",
            "heartbeat_at",
            "attempt_count",
            "result_json",
            "attempt_token",
        }
    ),
    "fold_results": frozenset(
        {
            "trial_id",
            "result_order",
            "fold_index",
            "seed",
            "score",
            "user_metric_value",
            "fit_seconds",
            "predict_seconds",
            "n_train_rows",
            "n_validation_rows",
            "result_json",
        }
    ),
    "scheduler_state": frozenset({"run_id", "component", "revision", "state_json", "updated_at"}),
    "events": frozenset(
        {"event_id", "run_id", "trial_id", "occurred_at", "level", "event_type", "payload_json"}
    ),
    "artifacts": frozenset(
        {
            "artifact_id",
            "run_id",
            "trial_id",
            "name",
            "kind",
            "relative_path",
            "sha256",
            "size_bytes",
            "media_type",
            "metadata_json",
            "created_at",
        }
    ),
    "checkpoint_snapshots": frozenset({"run_id", "revision", "state_json", "created_at"}),
    "resume_operations": frozenset(
        {"operation_id", "run_id", "additional_budget_seconds", "applied_at"}
    ),
}

EXPECTED_INDEXES = frozenset(
    {
        "trials_run_status",
        "trials_run_family_score",
        "events_run_event",
        "artifacts_run_kind",
        "checkpoint_snapshots_run_revision",
        "resume_operations_run",
    }
)


def apply_migrations(connection: sqlite3.Connection) -> int:
    """Apply every missing migration atomically and return the schema version."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
            """
        )
        applied = {
            int(row[0])
            for row in connection.execute("SELECT version FROM schema_migrations").fetchall()
        }
        unknown = [version for version in applied if version > LATEST_SCHEMA_VERSION]
        if unknown:
            raise RuntimeError(
                f"database schema version {max(unknown)} is newer than supported "
                f"version {LATEST_SCHEMA_VERSION}"
            )
        for migration in MIGRATIONS:
            if migration.version in applied:
                continue
            for statement in _sql_statements(migration.sql):
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) "
                "VALUES (?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))",
                (migration.version,),
            )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    return current_schema_version(connection)


def current_schema_version(connection: sqlite3.Connection) -> int:
    """Return zero for an uninitialized database, otherwise its latest migration."""
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'"
    ).fetchone()
    if exists is None:
        return 0
    row = connection.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()
    return int(row[0])


def _sql_statements(script: str) -> list[str]:
    statements: list[str] = []
    buffer = ""
    for line in script.splitlines():
        buffer += f"{line}\n"
        if sqlite3.complete_statement(buffer):
            statement = buffer.strip()
            if statement:
                statements.append(statement)
            buffer = ""
    if buffer.strip():
        raise ValueError("migration contains an incomplete SQL statement")
    return statements


__all__ = [
    "EXPECTED_INDEXES",
    "EXPECTED_SCHEMA",
    "LATEST_SCHEMA_VERSION",
    "MIGRATIONS",
    "Migration",
    "apply_migrations",
    "current_schema_version",
]

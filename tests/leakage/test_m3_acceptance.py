"""M3 acceptance tests for explainable leakage detection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from autonomous_automl.contracts import (
    AutoMLConfig,
    FindingSeverity,
    FoldAssignment,
    LeakageFinding,
    LeakageFindingType,
    ValidationPlan,
)
from autonomous_automl.data import LoadedDataset, load_dataset
from autonomous_automl.profiling import detect_leakage, profile_dataset
from autonomous_automl.validation import audit_validation


def load_frame(tmp_path: Path, frame: pd.DataFrame) -> LoadedDataset:
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    config = AutoMLConfig(
        target="target",
        task="binary_classification",
        metric="roc_auc",
        budget_seconds=30,
        random_seed=42,
        output_dir=tmp_path / "run",
    )
    return load_dataset(path, config)


def findings_of_type(
    dataset: LoadedDataset,
    finding_type: LeakageFindingType,
) -> list[LeakageFinding]:
    profile = profile_dataset(dataset)
    report = detect_leakage(dataset, profile)
    return [finding for finding in report.findings if finding.finding_type is finding_type]


def test_e1_exact_target_copy_is_critical_and_excluded_before_validation(
    tmp_path: Path,
) -> None:
    target = [0, 1] * 6
    dataset = load_frame(
        tmp_path,
        pd.DataFrame(
            {
                "signal": range(12),
                "leaked_target": target,
                "target": target,
            }
        ),
    )

    report = detect_leakage(dataset, profile_dataset(dataset))
    copies = [
        finding
        for finding in report.findings
        if finding.finding_type is LeakageFindingType.TARGET_COPY
    ]

    assert "target" not in dataset.X.columns
    assert len(copies) == 1
    assert copies[0].column == "leaked_target"
    assert copies[0].severity is FindingSeverity.CRITICAL
    assert copies[0].confidence == 1.0
    assert copies[0].action == "exclude_before_validation"
    assert copies[0].evidence_summary["matching_ratio"] == 1.0
    assert "leaked_target" in report.excluded_columns


def test_e2_nearly_unique_identifier_is_reported_and_neutralized(tmp_path: Path) -> None:
    dataset = load_frame(
        tmp_path,
        pd.DataFrame(
            {
                "customer_id": [f"customer-{index:03d}" for index in range(20)],
                "signal": [index % 3 for index in range(20)],
                "target": [index % 2 for index in range(20)],
            }
        ),
    )

    report = detect_leakage(dataset, profile_dataset(dataset))
    identifiers = [
        finding
        for finding in report.findings
        if finding.finding_type is LeakageFindingType.IDENTIFIER
    ]

    assert any(finding.column == "customer_id" for finding in identifiers)
    identifier = next(finding for finding in identifiers if finding.column == "customer_id")
    assert identifier.severity is FindingSeverity.WARNING
    assert identifier.confidence >= 0.98
    assert identifier.action == "exclude_identifier_from_features"
    assert "customer_id" in report.excluded_columns


def test_e3_exact_duplicate_rows_are_reported_for_cross_fold_audit(tmp_path: Path) -> None:
    dataset = load_frame(
        tmp_path,
        pd.DataFrame(
            {
                "signal": [1, 1, 2, 2, 3, 3, 4, 4],
                "category": ["a", "a", "b", "b", "c", "c", "d", "d"],
                "target": [0, 0, 1, 1, 0, 0, 1, 1],
            }
        ),
    )

    profile = profile_dataset(dataset)
    duplicates = findings_of_type(dataset, LeakageFindingType.DUPLICATE_ROWS)

    assert profile.duplicate_row_count == 4
    assert profile.potential_cross_fold_duplicate_count == 8
    assert len(duplicates) == 1
    assert duplicates[0].severity is FindingSeverity.WARNING
    assert duplicates[0].column is None
    assert duplicates[0].action == "audit_duplicate_groups_after_fold_materialization"
    assert duplicates[0].evidence_summary["potential_cross_fold_rows"] == 8

    deliberately_crossing_plan = ValidationPlan(
        splitter_name="AcceptanceDuplicateSplit",
        n_splits=2,
        shuffle=False,
        rationale=["force duplicate pairs across train and validation"],
        dataset_hash=profile.dataset_hash,
        folds=[
            FoldAssignment(
                fold_index=0,
                train_positions=[0, 2, 4, 6],
                validation_positions=[1, 3, 5, 7],
            ),
            FoldAssignment(
                fold_index=1,
                train_positions=[1, 3, 5, 7],
                validation_positions=[0, 2, 4, 6],
            ),
        ],
    )
    audit = audit_validation(dataset, deliberately_crossing_plan)

    assert audit.valid
    assert audit.duplicate_overlap_counts == [4, 4]
    assert any("duplicates cross" in warning for warning in audit.warnings)

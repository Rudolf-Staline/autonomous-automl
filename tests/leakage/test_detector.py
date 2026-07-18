"""Detailed tests for explainable target-leakage heuristics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from autonomous_automl import AutoMLConfig
from autonomous_automl.contracts import LeakageFindingType, LeakageReport
from autonomous_automl.data import LoadedDataset, load_dataset
from autonomous_automl.profiling import LeakageDetector, detect_leakage, profile_dataset


def _load(tmp_path: Path, frame: pd.DataFrame, *, task: str = "auto") -> LoadedDataset:
    source = tmp_path / "train.csv"
    frame.to_csv(source, index=False)
    return load_dataset(
        source,
        AutoMLConfig(target="target", task=task, budget_seconds=60),
    )


def _findings(report: LeakageReport, finding_type: LeakageFindingType) -> list[str | None]:
    return [finding.column for finding in report.findings if finding.finding_type is finding_type]


def test_near_target_copy_is_critical_and_excluded(tmp_path: Path) -> None:
    target = [index % 2 for index in range(100)]
    proxy = target.copy()
    proxy[0] = 1 - proxy[0]
    dataset = _load(
        tmp_path,
        pd.DataFrame({"feature": range(100), "proxy": proxy, "target": target}),
    )

    report = detect_leakage(dataset, profile_dataset(dataset))
    finding = next(
        item
        for item in report.findings
        if item.finding_type is LeakageFindingType.NEAR_TARGET_COPY and item.column == "proxy"
    )

    assert finding.confidence == 0.99
    assert finding.evidence_summary["matching_ratio"] == 0.99
    assert "proxy" in report.excluded_columns


def test_regression_linear_target_transform_is_excluded(tmp_path: Path) -> None:
    target = [float(index) / 10 for index in range(30)]
    dataset = _load(
        tmp_path,
        pd.DataFrame(
            {
                "safe": [index % 5 for index in range(30)],
                "post_score": [value * 2.0 + 7.0 for value in target],
                "target": target,
            }
        ),
        task="regression",
    )

    report = detect_leakage(dataset, profile_dataset(dataset, task="regression"))

    assert "post_score" in _findings(report, LeakageFindingType.NEAR_TARGET_COPY)
    assert "post_score" in report.excluded_columns


def test_suspicious_target_and_post_outcome_names_are_recorded(tmp_path: Path) -> None:
    dataset = _load(
        tmp_path,
        pd.DataFrame(
            {
                "target_probability": [0.1, 0.4, 0.2, 0.8, 0.3, 0.7],
                "final_result_code": [4, 3, 2, 1, 0, 5],
                "safe": [9, 2, 7, 4, 5, 3],
                "target": [0, 1, 0, 1, 0, 1],
            }
        ),
    )

    report = detect_leakage(dataset, profile_dataset(dataset))

    assert "target_probability" in _findings(report, LeakageFindingType.SUSPICIOUS_NAME)
    assert "final_result_code" in _findings(report, LeakageFindingType.POST_OUTCOME)
    assert "safe" not in report.excluded_columns


def test_detector_records_ids_with_aggregate_evidence_only(tmp_path: Path) -> None:
    dataset = _load(
        tmp_path,
        pd.DataFrame(
            {
                "record_id": [f"secret-{index}" for index in range(20)],
                "safe": [index % 4 for index in range(20)],
                "target": [index % 2 for index in range(20)],
            }
        ),
    )

    report = detect_leakage(dataset, profile_dataset(dataset))
    finding = next(item for item in report.findings if item.column == "record_id")

    assert finding.finding_type is LeakageFindingType.IDENTIFIER
    assert finding.evidence_summary == {"unique_ratio": 1.0, "cardinality": 20}
    assert not any("secret-" in str(value) for value in finding.evidence_summary.values())


def test_detector_is_independent_of_test_values(tmp_path: Path) -> None:
    train = pd.DataFrame({"feature": range(12), "category": ["a", "b"] * 6, "target": [0, 1] * 6})
    train_path = tmp_path / "train.csv"
    first_test = tmp_path / "first.csv"
    second_test = tmp_path / "second.csv"
    train.to_csv(train_path, index=False)
    pd.DataFrame({"feature": [13], "category": ["unseen-a"]}).to_csv(first_test, index=False)
    pd.DataFrame({"feature": [999999], "category": ["unseen-b"]}).to_csv(second_test, index=False)
    first = load_dataset(
        train_path,
        AutoMLConfig(target="target", budget_seconds=60, test_path=first_test),
    )
    second = load_dataset(
        train_path,
        AutoMLConfig(target="target", budget_seconds=60, test_path=second_test),
    )

    first_report = detect_leakage(first, profile_dataset(first))
    second_report = detect_leakage(second, profile_dataset(second))

    assert first_report == second_report


def test_leakage_report_round_trips_as_json(tmp_path: Path) -> None:
    target = [0, 1] * 5
    dataset = _load(
        tmp_path,
        pd.DataFrame({"copy": target, "feature": range(10), "target": target}),
    )
    report = detect_leakage(dataset, profile_dataset(dataset))

    assert LeakageReport.from_json(report.to_json()) == report


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("near_match_threshold", 0.89),
        ("near_match_threshold", 1.0),
        ("correlation_threshold", 0.89),
        ("correlation_threshold", 1.01),
    ],
)
def test_detector_rejects_invalid_thresholds(keyword: str, value: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        LeakageDetector(**{keyword: value})

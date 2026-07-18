"""Regression tests for the Build Week CSV-to-artifacts public workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autonomous_automl import AutoMLConfig, AutoMLRun
from autonomous_automl.api import load_best_pipeline, open_run, validate_run_artifacts
from autonomous_automl.contracts import RunStatus, TaskType
from autonomous_automl.utils.errors import PlannedInterruption


def _classification_files(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(91)
    rows = 150
    numeric = rng.normal(size=rows)
    category = rng.choice(["a", "b", "c"], size=rows)
    target = (numeric + 0.7 * (category == "a") + rng.normal(0, 0.45, rows) > 0).astype(int)
    train = pd.DataFrame(
        {
            "row_id": [f"train-{index}" for index in range(rows)],
            "numeric": numeric,
            "category": category,
            "target": target,
        }
    )
    test = pd.DataFrame(
        {
            "row_id": ["test-0", "test-1", "test-2"],
            "numeric": [-1.0, 0.2, 1.1],
            "category": ["unseen", "a", "b"],
        }
    )
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, test_path


@pytest.mark.integration
def test_public_fit_persists_and_replays_predictions(tmp_path: Path) -> None:
    train_path, test_path = _classification_files(tmp_path)
    output = tmp_path / "classification-run"
    result = AutoMLRun(
        AutoMLConfig(
            target="target",
            task="auto",
            metric="auto",
            budget_seconds=7,
            random_seed=42,
            id_column="row_id",
            test_path=test_path,
            output_dir=output,
            trial_timeout_seconds=4,
        )
    ).fit(train_path)

    assert result.status is RunStatus.COMPLETED
    assert result.primary_metric.value == "roc_auc"
    assert result.best_pipeline_path.is_file()
    assert result.predictions_path is not None
    assert result.predictions_path.is_file()
    assert result.report_path.is_file()
    report = result.report_path.read_text(encoding="utf-8")
    assert "Leakage and schema diagnostics" in report
    assert "Best PipelineSpec" in report
    assert "Reproduce and inspect" in report
    assert "train.csv" in report
    assert "Scope and known limits" in report
    assert str(tmp_path) not in report
    assert ">None<" not in report
    assert ">NaN<" not in report

    summary = validate_run_artifacts(output)
    assert summary.pipeline_loadable
    assert summary.predictions_match is True
    replay_frame = pd.read_csv(test_path).drop(columns="row_id")
    assert len(load_best_pipeline(output).predict(replay_frame)) == 3


@pytest.mark.integration
def test_interrupted_public_run_resumes_without_losing_trials(tmp_path: Path) -> None:
    train_path, test_path = _classification_files(tmp_path)
    output = tmp_path / "resume-run"
    config = AutoMLConfig(
        target="target",
        budget_seconds=8,
        random_seed=42,
        id_column="row_id",
        test_path=test_path,
        output_dir=output,
        trial_timeout_seconds=4,
    )

    with pytest.raises(PlannedInterruption):
        AutoMLRun(config).fit(train_path, interrupt_after_trials=2)
    _, interrupted_store, interrupted_manifest = open_run(output)
    assert interrupted_manifest.status is RunStatus.INTERRUPTED
    interrupted_trials = interrupted_store.list_trials(interrupted_manifest.run_id)
    before = {trial.trial_id for trial in interrupted_trials}
    assert len(before) == 2

    result = AutoMLRun.resume(output, additional_budget_seconds=4)
    _, resumed_store, resumed_manifest = open_run(output)
    after = {trial.trial_id for trial in resumed_store.list_trials(resumed_manifest.run_id)}
    assert result.status is RunStatus.COMPLETED
    assert resumed_manifest.status is RunStatus.COMPLETED
    assert before.issubset(after)
    assert validate_run_artifacts(output).predictions_match is True


@pytest.mark.integration
def test_regression_and_leakage_neutralization_complete(tmp_path: Path) -> None:
    rng = np.random.default_rng(123)
    rows = 150
    feature = rng.normal(size=rows)
    category = rng.choice(["small", "large"], size=rows)
    target = 5.0 + 3.2 * feature + 1.5 * (category == "large") + rng.normal(0, 0.3, rows)
    train = pd.DataFrame(
        {
            "customer_id": [f"customer-{index:04d}" for index in range(rows)],
            "feature": feature,
            "category": category,
            "target_copy": target,
            "value": target,
        }
    )
    train_path = tmp_path / "regression-leak.csv"
    train.to_csv(train_path, index=False)
    output = tmp_path / "regression-run"
    result = AutoMLRun(
        AutoMLConfig(
            target="value",
            task=TaskType.REGRESSION,
            metric="rmse",
            budget_seconds=7,
            output_dir=output,
            trial_timeout_seconds=4,
        )
    ).fit(train_path)

    _, _, manifest = open_run(output)
    assert result.status is RunStatus.COMPLETED
    assert result.primary_metric.value == "rmse"
    assert {"customer_id", "target_copy"}.issubset(manifest.leakage_report.excluded_columns)
    assert manifest.best_pipeline is not None
    report = result.report_path.read_text(encoding="utf-8")
    assert "post-neutralization" in report
    assert "No contaminated comparison score was calculated" in report
    assert str(tmp_path) not in report

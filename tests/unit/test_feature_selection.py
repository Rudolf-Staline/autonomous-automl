"""Tests for reconstructible, fold-local feature selection candidates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_classif

from autonomous_automl.components import ModelRegistry
from autonomous_automl.contracts import AutoMLConfig, DatasetProfile, PipelineSpec, TaskType
from autonomous_automl.data import load_dataset
from autonomous_automl.pipelines import (
    PipelineGrammar,
    build_pipeline,
    check_pipeline_compatibility,
    pipeline_fingerprint,
)
from autonomous_automl.profiling import profile_dataset
from autonomous_automl.utils.errors import IncompatiblePipelineError


def _make_problem(tmp_path: Path) -> tuple[pd.DataFrame, pd.Series, DatasetProfile]:
    n_rows = 200
    frame = pd.DataFrame(
        {
            "numeric": [np.nan if index % 17 == 0 else float(index) for index in range(n_rows)],
            "signal": [float((index % 2) * 4 + index % 5) for index in range(n_rows)],
            "city": [f"city-{index % 50}" for index in range(n_rows)],
            "event_date": pd.date_range("2023-01-01", periods=n_rows, freq="D").astype(str),
            "target": [index % 2 for index in range(n_rows)],
        }
    )
    train_path = tmp_path / "train.csv"
    frame.to_csv(train_path, index=False)
    config = AutoMLConfig(
        target="target",
        task="binary_classification",
        metric="roc_auc",
        budget_seconds=30,
        random_seed=42,
        n_jobs=1,
        output_dir=tmp_path / "run",
    )
    dataset = load_dataset(train_path, config)
    profile = profile_dataset(dataset, TaskType.BINARY_CLASSIFICATION)
    return dataset.X, dataset.y, profile


def _logistic_spec(feature_selector: str | None) -> PipelineSpec:
    return PipelineSpec(
        family="linear",
        task="binary_classification",
        numeric_imputer="median_indicator",
        numeric_scaler="standard",
        categorical_imputer="constant",
        categorical_encoder="one_hot",
        datetime_transformer="calendar",
        feature_selector=feature_selector,
        model_name="logistic_regression",
        model_params={"C": 1.0, "class_weight": None, "max_iter": 300},
        excluded_columns=[],
        random_seed=42,
    )


def _hist_gradient_boosting_spec(feature_selector: str | None) -> PipelineSpec:
    return PipelineSpec(
        family="hist_gradient_boosting",
        task="binary_classification",
        numeric_imputer="none",
        numeric_scaler="none",
        categorical_imputer="constant",
        categorical_encoder="ordinal",
        datetime_transformer="calendar",
        feature_selector=feature_selector,
        model_name="hist_gradient_boosting",
        model_params={
            "learning_rate": 0.1,
            "max_iter": 20,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 5,
            "l2_regularization": 0.0,
        },
        excluded_columns=[],
        random_seed=42,
    )


@pytest.mark.parametrize(
    ("selector_name", "selector_type"),
    [
        ("variance", VarianceThreshold),
        ("univariate_25", SelectPercentile),
        ("univariate_50", SelectPercentile),
    ],
)
def test_selectors_are_reconstructible_pipeline_steps(
    tmp_path: Path,
    selector_name: str,
    selector_type: type[object],
) -> None:
    X, y, profile = _make_problem(tmp_path)
    specification = _logistic_spec(selector_name)
    pipeline = build_pipeline(specification, profile, n_jobs=1)

    assert list(pipeline.named_steps) == ["preprocessor", "selector", "model"]
    assert isinstance(pipeline.named_steps["selector"], selector_type)
    if isinstance(pipeline.named_steps["selector"], SelectPercentile):
        assert pipeline.named_steps["selector"].score_func is f_classif
        assert pipeline.named_steps["selector"].percentile == int(selector_name.rsplit("_", 1)[1])

    fitted = pipeline.fit(X, y)
    restored = PipelineSpec.from_json(specification.to_json())
    rebuilt = build_pipeline(restored, profile, n_jobs=1).fit(X, y)
    inference = X.iloc[:8].copy()

    np.testing.assert_allclose(fitted.predict_proba(inference), rebuilt.predict_proba(inference))
    assert pipeline_fingerprint(restored) == pipeline_fingerprint(specification)


def test_univariate_selector_requires_imputed_numeric_values(tmp_path: Path) -> None:
    _, _, profile = _make_problem(tmp_path)
    specification = _hist_gradient_boosting_spec("univariate_50")
    decision = check_pipeline_compatibility(
        specification,
        profile,
        ModelRegistry.default(include_optional=False),
    )

    assert not decision.compatible
    assert "univariate feature selection requires imputed numeric values" in decision.reasons
    with pytest.raises(IncompatiblePipelineError, match="requires imputed numeric values"):
        build_pipeline(specification, profile, n_jobs=1)


def test_unknown_selector_is_rejected_before_construction(tmp_path: Path) -> None:
    _, _, profile = _make_problem(tmp_path)
    specification = _logistic_spec("external_selector")
    decision = check_pipeline_compatibility(
        specification,
        profile,
        ModelRegistry.default(include_optional=False),
    )

    assert not decision.compatible
    assert "unknown feature selector: external_selector" in decision.reasons


def test_grammar_emits_dataset_aware_selector_variants(tmp_path: Path) -> None:
    _, _, profile = _make_problem(tmp_path)
    registry = ModelRegistry.default(include_optional=False)
    candidates = PipelineGrammar(registry).generate(profile, random_seed=42)
    selectors = {candidate.feature_selector for candidate in candidates}

    assert {None, "variance", "univariate_50"} <= selectors
    assert any(candidate.feature_selector == "univariate_25" for candidate in candidates)
    assert all(
        check_pipeline_compatibility(candidate, profile, registry).compatible
        for candidate in candidates
    )
    assert len({pipeline_fingerprint(candidate) for candidate in candidates}) == len(candidates)

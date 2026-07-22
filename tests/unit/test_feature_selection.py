"""Tests for reconstructible, fold-local feature selection candidates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_regression
from sklearn.metrics import roc_auc_score

from autonomous_automl.components import ModelRegistry
from autonomous_automl.contracts import AutoMLConfig, DatasetProfile, PipelineSpec, TaskType
from autonomous_automl.data import load_dataset
from autonomous_automl.pipelines import (
    PipelineGrammar,
    build_feature_selector,
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
            "city": [f"city-{index % 90}" for index in range(n_rows)],
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


def _regression_spec(feature_selector: str | None) -> PipelineSpec:
    return PipelineSpec(
        family="linear",
        task="regression",
        numeric_imputer="mean",
        numeric_scaler="standard",
        categorical_imputer="constant",
        categorical_encoder="ordinal",
        datetime_transformer="calendar",
        feature_selector=feature_selector,
        model_name="ridge",
        model_params={"alpha": 1.0},
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
        assert pipeline.named_steps["selector"].score_func.__name__ == "stable_f_classif"
        assert pipeline.named_steps["selector"].percentile == int(selector_name.rsplit("_", 1)[1])

    fitted = pipeline.fit(X, y)
    restored = PipelineSpec.from_json(specification.to_json())
    rebuilt = build_pipeline(restored, profile, n_jobs=1).fit(X, y)
    inference = X.iloc[:8].copy()

    np.testing.assert_allclose(fitted.predict_proba(inference), rebuilt.predict_proba(inference))
    assert pipeline_fingerprint(restored) == pipeline_fingerprint(specification)


def test_regression_selector_uses_f_regression() -> None:
    selector = build_feature_selector(_regression_spec("univariate_50"))

    assert isinstance(selector, SelectPercentile)
    assert selector.score_func is f_regression


def test_stable_classification_score_keeps_a_perfect_scaled_feature(tmp_path: Path) -> None:
    rng = np.random.default_rng(119)
    rows = 140
    signal = rng.normal(size=rows)
    latent = 0.45 * signal + rng.normal(size=rows)
    target = (latent > np.median(latent)).astype(int)
    frame = pd.DataFrame(
        {
            "customer_id": [f"customer-{index:04d}" for index in range(rows)],
            "signal": signal,
            "noise": rng.normal(size=rows),
            "target_copy": target,
            "target": target,
        }
    )
    train_path = tmp_path / "perfect-feature.csv"
    frame.to_csv(train_path, index=False)
    dataset = load_dataset(
        train_path,
        AutoMLConfig(
            target="target",
            task="binary_classification",
            metric="roc_auc",
            budget_seconds=30,
            random_seed=42,
            n_jobs=1,
            output_dir=tmp_path / "perfect-run",
        ),
    )
    profile = profile_dataset(dataset, TaskType.BINARY_CLASSIFICATION).model_copy(
        update={"id_candidates": []},
        deep=True,
    )
    specification = _logistic_spec("univariate_50").model_copy(
        update={
            "numeric_imputer": "mean",
            "model_params": {"C": 0.04, "class_weight": None, "max_iter": 100},
        },
        deep=True,
    )
    pipeline = build_pipeline(specification, profile, n_jobs=1).fit(dataset.X, dataset.y)
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    selected_names = names[pipeline.named_steps["selector"].get_support()]

    assert "numeric__target_copy" in selected_names
    assert roc_auc_score(dataset.y, pipeline.predict_proba(dataset.X)[:, 1]) == pytest.approx(1.0)


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

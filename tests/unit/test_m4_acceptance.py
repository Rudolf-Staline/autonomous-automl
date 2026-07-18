"""M4 acceptance tests for reconstructible, fold-safe sklearn pipelines."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import clone

from autonomous_automl.components import (
    ModelRegistry,
    build_categorical_transformer,
    build_datetime_transformer,
    build_numeric_transformer,
)
from autonomous_automl.contracts import AutoMLConfig, DatasetProfile, PipelineSpec, TaskType
from autonomous_automl.data import load_dataset
from autonomous_automl.pipelines import (
    PipelineGrammar,
    build_pipeline,
    build_preprocessor,
    check_pipeline_compatibility,
    compatible_specs,
    pipeline_fingerprint,
)
from autonomous_automl.profiling import profile_dataset
from autonomous_automl.utils.errors import IncompatiblePipelineError, ModelUnavailableError


def make_tabular_problem(tmp_path: Path) -> tuple[pd.DataFrame, pd.Series, DatasetProfile]:
    n_rows = 200
    frame = pd.DataFrame(
        {
            "numeric": [np.nan if index % 17 == 0 else float(index) for index in range(n_rows)],
            "city": [f"city-{index % 50}" for index in range(n_rows)],
            "all_missing": [np.nan] * n_rows,
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
    return dataset.X, dataset.y, profile_dataset(dataset, TaskType.BINARY_CLASSIFICATION)


def make_logistic_spec(**updates: object) -> PipelineSpec:
    values: dict[str, object] = {
        "family": "linear",
        "task": "binary_classification",
        "numeric_imputer": "median_indicator",
        "numeric_scaler": "standard",
        "categorical_imputer": "constant",
        "categorical_encoder": "one_hot",
        "datetime_transformer": "calendar",
        "feature_selector": None,
        "model_name": "logistic_regression",
        "model_params": {"C": 1.0, "class_weight": None, "max_iter": 300},
        "excluded_columns": [],
        "random_seed": 42,
    }
    values.update(updates)
    return PipelineSpec.model_validate(values)


def make_hist_gradient_boosting_spec() -> PipelineSpec:
    return PipelineSpec(
        family="hist_gradient_boosting",
        task="binary_classification",
        numeric_imputer="none",
        numeric_scaler="none",
        categorical_imputer="constant",
        categorical_encoder="one_hot",
        datetime_transformer="calendar",
        feature_selector=None,
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


def adversarial_inference_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "numeric": [np.nan, np.inf, -5.0],
            "city": ["category-never-seen", None, "city-1"],
            "all_missing": [np.nan, "new-value", None],
            "event_date": ["2030-07-14", "not-a-date", None],
        }
    )


def test_preprocessing_components_are_sklearn_cloneable_and_unfitted() -> None:
    components = [
        build_numeric_transformer("median_indicator", "standard"),
        build_categorical_transformer("constant", "one_hot", dense_output=False),
        build_datetime_transformer("calendar"),
    ]

    for component in components:
        cloned = clone(component)

        assert cloned is not component
        assert cloned.get_params(deep=True).keys() == component.get_params(deep=True).keys()
        assert [name for name, _ in cloned.steps] == [name for name, _ in component.steps]
        assert [type(step) for _, step in cloned.steps] == [
            type(step) for _, step in component.steps
        ]
        assert not any(name.endswith("_") for name in vars(cloned))


def test_unknown_categories_and_missing_values_are_safe_at_inference() -> None:
    transformer = build_categorical_transformer("constant", "one_hot", dense_output=False)
    train = pd.DataFrame({"city": ["rabat", "casa", None, "rabat"]})
    inference = pd.DataFrame({"city": ["never_seen", None]})

    transformer.fit(train)
    transformed = transformer.transform(inference)

    assert transformed.shape[0] == len(inference)
    assert np.isfinite(transformed.toarray()).all()


def test_numeric_nan_infinities_and_all_missing_columns_become_finite() -> None:
    transformer = build_numeric_transformer("median_indicator", "robust")
    train = pd.DataFrame(
        {
            "partly_missing": [1.0, np.nan, 3.0, np.inf],
            "all_missing": [np.nan, np.nan, np.nan, np.nan],
        }
    )
    inference = pd.DataFrame(
        {
            "partly_missing": [np.nan, -np.inf, 8.0],
            "all_missing": [np.nan, np.nan, np.nan],
        }
    )

    transformer.fit(train)
    transformed = transformer.transform(inference)

    assert transformed.shape[0] == len(inference)
    assert np.isfinite(transformed).all()


def test_all_missing_categorical_column_remains_in_the_pipeline() -> None:
    transformer = build_categorical_transformer("constant", "one_hot", dense_output=True)
    train = pd.DataFrame({"all_missing": [np.nan, np.nan, np.nan, np.nan]})
    inference = pd.DataFrame({"all_missing": [np.nan, "new_category"]})

    transformer.fit(train)
    transformed = transformer.transform(inference)

    assert transformed.shape[0] == len(inference)
    assert transformed.shape[1] >= 1
    assert np.isfinite(transformed).all()


def test_date_features_are_deterministic_and_invalid_dates_are_imputed() -> None:
    transformer = build_datetime_transformer("calendar")
    train = pd.DataFrame({"event_date": ["2024-01-01", "2024-02-29", None, "2025-12-31"]})
    inference = pd.DataFrame({"event_date": ["2030-07-14", "not-a-date", None]})

    first = transformer.fit_transform(train)
    transformed = transformer.transform(inference)
    rebuilt = clone(transformer).fit_transform(train)

    np.testing.assert_allclose(first, rebuilt)
    assert transformed.shape[0] == len(inference)
    assert np.isfinite(transformed).all()


def test_registry_filters_models_that_do_not_support_the_task() -> None:
    registry = ModelRegistry.default(include_optional=False)

    classification = {
        adapter.name for adapter in registry.available(TaskType.BINARY_CLASSIFICATION)
    }
    regression = {adapter.name for adapter in registry.available(TaskType.REGRESSION)}

    assert {"dummy", "logistic_regression", "random_forest"} <= classification
    assert "ridge" not in classification
    assert "elastic_net" not in classification
    assert {"dummy", "ridge", "elastic_net", "random_forest"} <= regression
    assert "logistic_regression" not in regression


def test_absent_optional_models_are_nonfatal_and_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optional_modules = {"xgboost", "lightgbm", "catboost"}
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, package: str | None = None) -> object | None:
        if name in optional_modules:
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    registry = ModelRegistry.default(include_optional=True)

    assert optional_modules == set(registry.unavailable())
    assert "dummy" in {adapter.name for adapter in registry.available()}
    with pytest.raises(ModelUnavailableError, match="requires package 'xgboost'"):
        registry.get("xgboost").build(
            {},
            TaskType.BINARY_CLASSIFICATION,
            random_seed=42,
            n_jobs=1,
        )


def test_complete_pipeline_is_cloneable_and_handles_adversarial_inference(
    tmp_path: Path,
) -> None:
    X, y, profile = make_tabular_problem(tmp_path)
    specification = make_logistic_spec()
    pipeline = build_pipeline(specification, profile, n_jobs=1)

    cloned = clone(pipeline)
    pipeline.fit(X, y)
    cloned.fit(X, y)
    inference = adversarial_inference_frame()

    np.testing.assert_array_equal(pipeline.predict(inference), cloned.predict(inference))
    probabilities = pipeline.predict_proba(inference)
    assert probabilities.shape == (len(inference), 2)
    assert np.isfinite(probabilities).all()


def test_joblib_and_spec_rebuild_produce_the_same_predictions(tmp_path: Path) -> None:
    X, y, profile = make_tabular_problem(tmp_path)
    specification = make_logistic_spec()
    fitted = build_pipeline(specification, profile, n_jobs=1).fit(X, y)
    inference = adversarial_inference_frame()
    expected = fitted.predict_proba(inference)
    artifact_path = tmp_path / "best_pipeline.joblib"

    joblib.dump(fitted, artifact_path)
    restored_artifact = joblib.load(artifact_path)
    restored_specification = PipelineSpec.from_json(specification.to_json())
    rebuilt = build_pipeline(restored_specification, profile, n_jobs=1).fit(X, y)

    np.testing.assert_allclose(restored_artifact.predict_proba(inference), expected)
    np.testing.assert_allclose(rebuilt.predict_proba(inference), expected)
    assert pipeline_fingerprint(restored_specification) == pipeline_fingerprint(specification)
    changed = specification.model_copy(
        update={"model_params": {**specification.model_params, "C": 2.0}}
    )
    assert pipeline_fingerprint(changed) != pipeline_fingerprint(specification)


def test_one_hot_is_sparse_for_sparse_models_and_dense_for_hgb(tmp_path: Path) -> None:
    X, y, profile = make_tabular_problem(tmp_path)
    sparse_specification = make_logistic_spec()
    dense_specification = make_hist_gradient_boosting_spec()

    sparse_values = build_preprocessor(sparse_specification, profile).fit_transform(X, y)
    dense_values = build_preprocessor(dense_specification, profile).fit_transform(X, y)

    assert sparse.issparse(sparse_values)
    assert not sparse.issparse(dense_values)
    dense_pipeline = build_pipeline(dense_specification, profile, n_jobs=1).fit(X, y)
    predictions = dense_pipeline.predict(adversarial_inference_frame())
    assert predictions.shape == (3,)


def test_incompatible_specs_are_explained_filtered_and_never_built(tmp_path: Path) -> None:
    _, _, profile = make_tabular_problem(tmp_path)
    registry = ModelRegistry.default(include_optional=False)
    compatible = make_logistic_spec()
    missing_without_imputation = make_logistic_spec(numeric_imputer="none")
    wrong_family = make_logistic_spec(family="random_forest")

    missing_decision = check_pipeline_compatibility(
        missing_without_imputation,
        profile,
        registry,
    )
    family_decision = check_pipeline_compatibility(wrong_family, profile, registry)

    assert not missing_decision.compatible
    assert any("missing values require imputation" in reason for reason in missing_decision.reasons)
    assert not family_decision.compatible
    assert any("family" in reason for reason in family_decision.reasons)
    assert compatible_specs(
        [missing_without_imputation, compatible, wrong_family],
        profile,
        registry,
    ) == [compatible]
    with pytest.raises(IncompatiblePipelineError, match="missing values require imputation"):
        build_pipeline(missing_without_imputation, profile, registry=registry)


def test_grammar_emits_only_compatible_candidates_and_always_a_baseline(
    tmp_path: Path,
) -> None:
    _, _, profile = make_tabular_problem(tmp_path)
    registry = ModelRegistry.default(include_optional=False)
    candidates = PipelineGrammar(registry).generate(profile, random_seed=42)

    assert candidates
    assert any(candidate.model_name == "dummy" for candidate in candidates)
    assert all(
        check_pipeline_compatibility(candidate, profile, registry).compatible
        for candidate in candidates
    )
    assert all(candidate.random_seed == 42 for candidate in candidates)
    assert len({pipeline_fingerprint(candidate) for candidate in candidates}) == len(candidates)

"""Tests for core model adapters and optional dependency isolation."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from autonomous_automl.components import ModelRegistry
from autonomous_automl.contracts import TaskType
from autonomous_automl.utils.errors import ModelUnavailableError

CORE_NAMES = {
    "dummy",
    "logistic_regression",
    "ridge",
    "elastic_net",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
}


def test_minimal_registry_contains_every_mandatory_model() -> None:
    registry = ModelRegistry.default(include_optional=False)

    assert set(registry.names()) == CORE_NAMES
    assert registry.unavailable() == {}


@pytest.mark.parametrize(
    ("name", "task"),
    [
        ("dummy", TaskType.BINARY_CLASSIFICATION),
        ("dummy", TaskType.REGRESSION),
        ("logistic_regression", TaskType.BINARY_CLASSIFICATION),
        ("ridge", TaskType.REGRESSION),
        ("elastic_net", TaskType.REGRESSION),
        ("random_forest", TaskType.BINARY_CLASSIFICATION),
        ("random_forest", TaskType.REGRESSION),
        ("extra_trees", TaskType.MULTICLASS_CLASSIFICATION),
        ("extra_trees", TaskType.REGRESSION),
        ("hist_gradient_boosting", TaskType.BINARY_CLASSIFICATION),
        ("hist_gradient_boosting", TaskType.REGRESSION),
    ],
)
def test_core_estimators_build_deterministically_and_are_cloneable(
    name: str,
    task: TaskType,
) -> None:
    adapter = ModelRegistry.default(include_optional=False).get(name)
    params = adapter.default_params(task)
    if "n_estimators" in params:
        params["n_estimators"] = 10
    if "max_iter" in params:
        params["max_iter"] = 20

    first = adapter.build(params, task, random_seed=123, n_jobs=1)
    second = adapter.build(params, task, random_seed=123, n_jobs=1)

    assert first.get_params(deep=True) == second.get_params(deep=True)
    assert clone(first).get_params(deep=True) == first.get_params(deep=True)


def test_representative_models_produce_reproducible_predictions() -> None:
    X = np.asarray([[index, index % 3] for index in range(60)], dtype=float)
    classification_y = np.asarray([index % 2 for index in range(60)])
    regression_y = np.asarray([index * 0.5 + (index % 3) for index in range(60)])
    registry = ModelRegistry.default(include_optional=False)

    classifier_adapter = registry.get("random_forest")
    classifier_params = classifier_adapter.default_params(TaskType.BINARY_CLASSIFICATION)
    classifier_params["n_estimators"] = 20
    classifiers = [
        classifier_adapter.build(
            classifier_params, TaskType.BINARY_CLASSIFICATION, random_seed=7, n_jobs=1
        )
        for _ in range(2)
    ]
    regressor_adapter = registry.get("hist_gradient_boosting")
    regressor_params = regressor_adapter.default_params(TaskType.REGRESSION)
    regressor_params["max_iter"] = 20
    regressors = [
        regressor_adapter.build(regressor_params, TaskType.REGRESSION, random_seed=7, n_jobs=1)
        for _ in range(2)
    ]

    np.testing.assert_array_equal(
        classifiers[0].fit(X, classification_y).predict(X),
        classifiers[1].fit(X, classification_y).predict(X),
    )
    np.testing.assert_allclose(
        regressors[0].fit(X, regression_y).predict(X),
        regressors[1].fit(X, regression_y).predict(X),
    )


def test_hist_gradient_boosting_capabilities_are_conservative() -> None:
    adapter = ModelRegistry.default(include_optional=False).get("hist_gradient_boosting")

    assert adapter.supports_missing_values()
    assert not adapter.supports_sparse_input()
    assert not adapter.supports_native_categoricals()


def test_adapter_rejects_an_unsupported_task() -> None:
    adapter = ModelRegistry.default(include_optional=False).get("ridge")

    with pytest.raises(ValueError, match="does not support"):
        adapter.build({}, TaskType.BINARY_CLASSIFICATION, random_seed=42, n_jobs=1)


def test_registry_rejects_empty_duplicate_and_unknown_entries() -> None:
    dummy = ModelRegistry.default(include_optional=False).get("dummy")

    with pytest.raises(ValueError, match="at least one"):
        ModelRegistry([])
    with pytest.raises(ValueError, match="duplicate"):
        ModelRegistry([dummy, dummy])
    with pytest.raises(ModelUnavailableError, match="unknown"):
        ModelRegistry.default(include_optional=False).get("not-a-model")

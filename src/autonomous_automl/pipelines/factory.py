"""Deterministic materialization of executable sklearn pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_classif, f_regression
from sklearn.pipeline import Pipeline

from autonomous_automl.components import (
    ModelRegistry,
    build_categorical_transformer,
    build_datetime_transformer,
    build_numeric_transformer,
)
from autonomous_automl.contracts import DatasetProfile, PipelineSpec, TaskType
from autonomous_automl.pipelines.compatibility import check_pipeline_compatibility
from autonomous_automl.utils.errors import IncompatiblePipelineError


def build_preprocessor(
    spec: PipelineSpec,
    profile: DatasetProfile,
    registry: ModelRegistry | None = None,
) -> ColumnTransformer:
    """Build a ColumnTransformer using only roles recorded in the profile."""
    active_registry = registry or ModelRegistry.default()
    adapter = active_registry.get(spec.model_name)
    excluded = set(spec.excluded_columns)
    numeric_columns = [
        column
        for column in [*profile.numeric_columns, *profile.boolean_columns]
        if column not in excluded
    ]
    categorical_columns = [
        column for column in profile.categorical_columns if column not in excluded
    ]
    datetime_columns = [column for column in profile.datetime_columns if column not in excluded]

    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                build_numeric_transformer(spec.numeric_imputer, spec.numeric_scaler),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                build_categorical_transformer(
                    spec.categorical_imputer,
                    spec.categorical_encoder,
                    dense_output=not adapter.supports_sparse_input(),
                ),
                categorical_columns,
            )
        )
    if datetime_columns and spec.datetime_transformer != "drop":
        transformers.append(
            (
                "datetime",
                build_datetime_transformer(spec.datetime_transformer),
                datetime_columns,
            )
        )
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0 if not adapter.supports_sparse_input() else 0.3,
        verbose_feature_names_out=True,
    )


def stable_f_classif(X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ANOVA scores with impossible floating-point results repaired.

    The theoretical F statistic is non-negative. Perfectly separating features can
    nevertheless produce a very large negative finite value after scaling because the
    within-class variance underflows below zero. Such values represent the same limiting
    case as positive infinity and must rank first, while NaNs from constant features rank
    last.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scores, pvalues = f_classif(X, y)
    maximum = np.finfo(np.float64).max
    stable_scores = np.asarray(scores, dtype=np.float64)
    stable_scores = np.nan_to_num(
        stable_scores,
        nan=0.0,
        posinf=maximum,
        neginf=maximum,
    )
    stable_scores = np.where(stable_scores < 0.0, maximum, stable_scores)
    return stable_scores, np.asarray(pvalues, dtype=np.float64)


def build_feature_selector(spec: PipelineSpec) -> BaseEstimator | None:
    """Build the fold-local selector recorded in the PipelineSpec."""
    if spec.feature_selector is None:
        return None
    if spec.feature_selector == "variance":
        return VarianceThreshold()
    if spec.feature_selector in {"univariate_25", "univariate_50"}:
        percentile = int(spec.feature_selector.rsplit("_", maxsplit=1)[1])
        score_func = f_regression if spec.task == TaskType.REGRESSION else stable_f_classif
        return SelectPercentile(score_func=score_func, percentile=percentile)
    raise IncompatiblePipelineError(f"unknown feature selector: {spec.feature_selector}")


def build_model(
    spec: PipelineSpec,
    profile: DatasetProfile,
    *,
    n_jobs: int = 1,
    registry: ModelRegistry | None = None,
) -> BaseEstimator:
    """Build the model described by a compatible PipelineSpec."""
    active_registry = registry or ModelRegistry.default()
    decision = check_pipeline_compatibility(spec, profile, active_registry)
    if not decision.compatible:
        raise IncompatiblePipelineError("; ".join(decision.reasons))
    adapter = active_registry.get(spec.model_name)
    return adapter.build(
        dict(spec.model_params),
        spec.task,
        spec.random_seed,
        n_jobs,
    )


def build_pipeline(
    spec: PipelineSpec,
    profile: DatasetProfile,
    *,
    n_jobs: int = 1,
    registry: ModelRegistry | None = None,
) -> Pipeline:
    """Reconstruct an unfitted sklearn Pipeline from its serializable source."""
    active_registry = registry or ModelRegistry.default()
    decision = check_pipeline_compatibility(spec, profile, active_registry)
    if not decision.compatible:
        raise IncompatiblePipelineError("; ".join(decision.reasons))
    steps: list[tuple[str, object]] = [
        ("preprocessor", build_preprocessor(spec, profile, active_registry))
    ]
    selector = build_feature_selector(spec)
    if selector is not None:
        steps.append(("selector", selector))
    steps.append(
        (
            "model",
            active_registry.get(spec.model_name).build(
                dict(spec.model_params),
                spec.task,
                spec.random_seed,
                n_jobs,
            ),
        )
    )
    return Pipeline(steps)


__all__ = [
    "build_feature_selector",
    "build_model",
    "build_pipeline",
    "build_preprocessor",
    "stable_f_classif",
]

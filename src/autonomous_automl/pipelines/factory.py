"""Deterministic materialization of executable sklearn pipelines."""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

from autonomous_automl.components import (
    ModelRegistry,
    build_categorical_transformer,
    build_datetime_transformer,
    build_numeric_transformer,
)
from autonomous_automl.contracts import DatasetProfile, PipelineSpec
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
    if spec.feature_selector == "variance":
        steps.append(("selector", VarianceThreshold()))
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


__all__ = ["build_model", "build_pipeline", "build_preprocessor"]

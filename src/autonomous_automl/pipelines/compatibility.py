"""Centralized, explainable compatibility rules for PipelineSpec candidates."""

from __future__ import annotations

from dataclasses import dataclass

from autonomous_automl.components import ModelRegistry
from autonomous_automl.contracts import DatasetProfile, PipelineSpec
from autonomous_automl.utils.errors import ModelUnavailableError

_NUMERIC_IMPUTERS = {"none", "mean", "median", "constant", "median_indicator", "knn"}
_NUMERIC_SCALERS = {"none", "standard", "robust"}
_CATEGORICAL_IMPUTERS = {"constant", "most_frequent"}
_CATEGORICAL_ENCODERS = {"one_hot", "ordinal", "frequency"}
_DATETIME_TRANSFORMERS = {"calendar", "drop"}
_FEATURE_SELECTORS = {None, "variance", "univariate_25", "univariate_50"}
_UNIVARIATE_SELECTORS = {"univariate_25", "univariate_50"}


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    """Compatibility decision and every actionable rejection reason."""

    compatible: bool
    reasons: tuple[str, ...]


def check_pipeline_compatibility(
    spec: PipelineSpec,
    profile: DatasetProfile,
    registry: ModelRegistry | None = None,
) -> CompatibilityResult:
    """Evaluate all V1 rules without constructing or fitting an estimator."""
    active_registry = registry or ModelRegistry.default()
    reasons: list[str] = []
    try:
        adapter = active_registry.get(spec.model_name)
    except ModelUnavailableError as error:
        return CompatibilityResult(False, (str(error),))

    if not adapter.is_available():
        reasons.append(f"model dependency for {spec.model_name!r} is unavailable")
    if spec.task not in adapter.supported_tasks():
        reasons.append(f"model {spec.model_name!r} does not support task {spec.task.value!r}")
    if profile.inferred_task != spec.task:
        reasons.append("PipelineSpec.task does not match the resolved dataset task")
    if spec.family != adapter.family:
        reasons.append("PipelineSpec.family does not match the registered model family")

    if spec.numeric_imputer not in _NUMERIC_IMPUTERS:
        reasons.append(f"unknown numeric imputer: {spec.numeric_imputer}")
    if spec.numeric_scaler not in _NUMERIC_SCALERS:
        reasons.append(f"unknown numeric scaler: {spec.numeric_scaler}")
    if spec.categorical_imputer not in _CATEGORICAL_IMPUTERS:
        reasons.append(f"unknown categorical imputer: {spec.categorical_imputer}")
    if spec.categorical_encoder not in _CATEGORICAL_ENCODERS:
        reasons.append(f"unknown categorical encoder: {spec.categorical_encoder}")
    if spec.datetime_transformer not in _DATETIME_TRANSFORMERS:
        reasons.append(f"unknown datetime transformer: {spec.datetime_transformer}")
    if spec.feature_selector not in _FEATURE_SELECTORS:
        reasons.append(f"unknown feature selector: {spec.feature_selector}")

    excluded = set(spec.excluded_columns)
    unsupported_text = sorted(set(profile.text_columns).difference(excluded))
    if unsupported_text:
        reasons.append(f"free-text columns must be explicitly excluded: {unsupported_text}")
    unexcluded_ids = sorted(set(profile.id_candidates).difference(excluded))
    if unexcluded_ids:
        reasons.append(f"identifier candidates must be explicitly excluded: {unexcluded_ids}")

    active_numeric = [
        column
        for column in [*profile.numeric_columns, *profile.boolean_columns]
        if column not in excluded
    ]
    numeric_has_missing = any(
        profile.missing_ratios.get(column, 0.0) > 0 for column in active_numeric
    )
    if (
        spec.numeric_imputer == "none"
        and numeric_has_missing
        and not adapter.supports_missing_values()
    ):
        reasons.append("numeric missing values require imputation for this model")
    if (
        spec.feature_selector in _UNIVARIATE_SELECTORS
        and spec.numeric_imputer == "none"
        and numeric_has_missing
    ):
        reasons.append("univariate feature selection requires imputed numeric values")
    if spec.numeric_imputer == "knn" and (profile.n_rows > 50_000 or len(active_numeric) > 100):
        reasons.append("KNN imputation is limited to at most 50k rows and 100 numeric columns")

    active_categorical = [
        column for column in profile.categorical_columns if column not in excluded
    ]
    if spec.categorical_encoder == "one_hot" and active_categorical:
        estimated_width = sum(
            max(1, profile.cardinalities.get(column, 1)) for column in active_categorical
        )
        if estimated_width > 100_000:
            reasons.append("estimated one-hot width exceeds the V1 hard limit of 100k")
        elif not adapter.supports_sparse_input() and estimated_width > 10_000:
            reasons.append("dense one-hot width exceeds the model safety limit of 10k")

    active_datetime = [column for column in profile.datetime_columns if column not in excluded]
    active_features = len(active_numeric) + len(active_categorical)
    if spec.datetime_transformer == "calendar":
        active_features += len(active_datetime)
    if active_features == 0 and spec.model_name != "dummy":
        reasons.append("no executable feature remains after explicit exclusions")

    return CompatibilityResult(not reasons, tuple(reasons))


def compatible_specs(
    specs: list[PipelineSpec],
    profile: DatasetProfile,
    registry: ModelRegistry | None = None,
) -> list[PipelineSpec]:
    """Filter candidates before execution using the centralized rules."""
    return [
        spec for spec in specs if check_pipeline_compatibility(spec, profile, registry).compatible
    ]


__all__ = ["CompatibilityResult", "check_pipeline_compatibility", "compatible_specs"]

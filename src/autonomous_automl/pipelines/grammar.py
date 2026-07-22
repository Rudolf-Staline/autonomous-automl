"""Compact conditional grammar for deterministic initial candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from autonomous_automl.components import ModelRegistry
from autonomous_automl.contracts import DatasetProfile, PipelineSpec, TaskType
from autonomous_automl.pipelines.compatibility import compatible_specs
from autonomous_automl.utils.json import JsonValue

_CLASSIFICATION_MODELS = (
    "dummy",
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
)
_REGRESSION_MODELS = (
    "dummy",
    "ridge",
    "elastic_net",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
)


@dataclass(slots=True)
class PipelineGrammar:
    """Generate diverse, compatible candidates without a Cartesian explosion."""

    registry: ModelRegistry = field(default_factory=ModelRegistry.default)

    def generate(
        self,
        profile: DatasetProfile,
        *,
        task: TaskType | None = None,
        random_seed: int = 42,
        excluded_columns: list[str] | None = None,
    ) -> list[PipelineSpec]:
        resolved_task = task or profile.inferred_task
        if resolved_task == TaskType.AUTO:
            raise ValueError("pipeline grammar requires a resolved task")
        automatic_exclusions = {
            *profile.text_columns,
            *profile.constant_columns,
            *profile.id_candidates,
            *(excluded_columns or []),
        }
        exclusions = sorted(automatic_exclusions)
        model_names = (
            _CLASSIFICATION_MODELS
            if resolved_task in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}
            else _REGRESSION_MODELS
        )
        candidates: list[PipelineSpec] = []
        for model_name in model_names:
            if model_name not in self.registry.names():
                continue
            adapter = self.registry.get(model_name)
            if not adapter.is_available() or resolved_task not in adapter.supported_tasks():
                continue
            for template in _templates_for_model(model_name):
                numeric_imputer, numeric_scaler, categorical_imputer, categorical_encoder = template
                selectors = _selectors_for_template(profile, model_name, template)
                for feature_selector in selectors:
                    candidates.append(
                        PipelineSpec(
                            family=adapter.family,
                            task=resolved_task,
                            numeric_imputer=numeric_imputer,
                            numeric_scaler=numeric_scaler,
                            categorical_imputer=categorical_imputer,
                            categorical_encoder=categorical_encoder,
                            datetime_transformer="calendar",
                            feature_selector=feature_selector,
                            model_name=model_name,
                            model_params=cast(
                                dict[str, JsonValue],
                                adapter.default_params(resolved_task),
                            ),
                            excluded_columns=exclusions,
                            random_seed=random_seed,
                        )
                    )
        return compatible_specs(candidates, profile, self.registry)


def generate_initial_candidates(
    profile: DatasetProfile,
    *,
    task: TaskType | None = None,
    random_seed: int = 42,
    excluded_columns: list[str] | None = None,
    registry: ModelRegistry | None = None,
) -> list[PipelineSpec]:
    """Generate initial candidates with a mandatory naive baseline."""
    return PipelineGrammar(registry or ModelRegistry.default()).generate(
        profile,
        task=task,
        random_seed=random_seed,
        excluded_columns=excluded_columns,
    )


def _templates_for_model(model_name: str) -> tuple[tuple[str, str, str, str], ...]:
    if model_name == "dummy":
        return (("median", "none", "constant", "ordinal"),)
    if model_name in {"logistic_regression", "ridge", "elastic_net"}:
        return (
            ("mean", "standard", "constant", "one_hot"),
            ("median_indicator", "robust", "most_frequent", "ordinal"),
            ("constant", "standard", "constant", "frequency"),
            ("knn", "robust", "constant", "one_hot"),
        )
    if model_name == "hist_gradient_boosting":
        return (
            ("none", "none", "constant", "ordinal"),
            ("median_indicator", "none", "most_frequent", "frequency"),
        )
    return (
        ("median", "none", "constant", "ordinal"),
        ("constant", "none", "most_frequent", "frequency"),
        ("mean", "none", "constant", "one_hot"),
    )


def _selectors_for_template(
    profile: DatasetProfile,
    model_name: str,
    template: tuple[str, str, str, str],
) -> tuple[str | None, ...]:
    """Choose a small selector set from train-only width metadata."""
    if model_name == "dummy":
        return (None,)

    numeric_imputer, _, _, categorical_encoder = template
    estimated_width = _estimated_transformed_width(profile, categorical_encoder)
    selectors: list[str | None] = [None]
    if estimated_width >= 12:
        selectors.append("variance")
    if estimated_width >= 40 and numeric_imputer != "none":
        selectors.append("univariate_50")
    if estimated_width >= 100 and numeric_imputer != "none":
        selectors.append("univariate_25")
    return tuple(selectors)


def _estimated_transformed_width(profile: DatasetProfile, categorical_encoder: str) -> int:
    numeric_width = len(profile.numeric_columns) + len(profile.boolean_columns)
    if categorical_encoder == "one_hot":
        categorical_width = sum(
            max(1, profile.cardinalities.get(column, 1)) for column in profile.categorical_columns
        )
    else:
        categorical_width = len(profile.categorical_columns)
    datetime_width = 8 * len(profile.datetime_columns)
    return numeric_width + categorical_width + datetime_width


__all__ = ["PipelineGrammar", "generate_initial_candidates"]

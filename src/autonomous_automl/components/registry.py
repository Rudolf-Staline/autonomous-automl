"""Stable model registry with non-fatal optional dependency discovery."""

from __future__ import annotations

from collections.abc import Iterable

from autonomous_automl.components.models import (
    CatBoostAdapter,
    DummyAdapter,
    ElasticNetAdapter,
    ExtraTreesAdapter,
    HistGradientBoostingAdapter,
    LightGBMAdapter,
    LogisticRegressionAdapter,
    RandomForestAdapter,
    RidgeAdapter,
    XGBoostAdapter,
)
from autonomous_automl.components.protocols import ModelAdapter
from autonomous_automl.contracts import TaskType
from autonomous_automl.utils.errors import ModelUnavailableError


class ModelRegistry:
    """Ordered collection of adapters keyed by persistent model name."""

    def __init__(self, adapters: Iterable[ModelAdapter]) -> None:
        self._adapters: dict[str, ModelAdapter] = {}
        for adapter in adapters:
            if adapter.name in self._adapters:
                raise ValueError(f"duplicate model adapter name: {adapter.name}")
            self._adapters[adapter.name] = adapter
        if not self._adapters:
            raise ValueError("model registry must contain at least one adapter")

    @classmethod
    def default(cls, *, include_optional: bool = True) -> ModelRegistry:
        """Create the mandatory core plus lazy optional adapter descriptors."""
        adapters: list[ModelAdapter] = [
            DummyAdapter(),
            LogisticRegressionAdapter(),
            RidgeAdapter(),
            ElasticNetAdapter(),
            RandomForestAdapter(),
            ExtraTreesAdapter(),
            HistGradientBoostingAdapter(),
        ]
        if include_optional:
            adapters.extend([XGBoostAdapter(), LightGBMAdapter(), CatBoostAdapter()])
        return cls(adapters)

    def get(self, name: str) -> ModelAdapter:
        """Return an adapter, distinguishing unknown names from missing packages."""
        try:
            return self._adapters[name]
        except KeyError as error:
            raise ModelUnavailableError(f"unknown model adapter: {name}") from error

    def names(self) -> list[str]:
        return list(self._adapters)

    def available(self, task: TaskType | None = None) -> list[ModelAdapter]:
        return [
            adapter
            for adapter in self._adapters.values()
            if adapter.is_available() and (task is None or task in adapter.supported_tasks())
        ]

    def unavailable(self) -> dict[str, str]:
        return {
            adapter.name: (
                f"optional dependency {getattr(adapter, 'module_name', adapter.name)!r} is absent"
            )
            for adapter in self._adapters.values()
            if not adapter.is_available()
        }


__all__ = ["ModelRegistry"]

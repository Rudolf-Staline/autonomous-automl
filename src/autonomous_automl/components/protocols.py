"""Typed model adapter interface used by planning and materialization."""

from __future__ import annotations

from typing import Protocol

import optuna
from sklearn.base import BaseEstimator

from autonomous_automl.contracts import DatasetProfile, FidelitySpec, PipelineSpec, TaskType


class ModelAdapter(Protocol):
    """Capabilities and deterministic construction for one stable model name."""

    name: str
    family: str

    def supported_tasks(self) -> set[TaskType]: ...

    def supports_missing_values(self) -> bool: ...

    def supports_native_categoricals(self) -> bool: ...

    def supports_sparse_input(self) -> bool: ...

    def estimated_cost(self) -> float: ...

    def is_available(self) -> bool: ...

    def is_compatible(self, profile: DatasetProfile, pipeline_spec: PipelineSpec) -> bool: ...

    def default_params(self, task: TaskType) -> dict[str, object]: ...

    def suggest_params(
        self,
        trial: optuna.Trial,
        profile: DatasetProfile,
        fidelity: FidelitySpec,
    ) -> dict[str, object]: ...

    def build(
        self,
        params: dict[str, object],
        task: TaskType,
        random_seed: int,
        n_jobs: int,
    ) -> BaseEstimator: ...


__all__ = ["ModelAdapter"]

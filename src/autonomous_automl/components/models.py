"""Deterministic core and lazily imported optional model adapters."""

from __future__ import annotations

import importlib
import importlib.util
from abc import ABC, abstractmethod
from typing import Any, ClassVar, cast

import optuna
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge

from autonomous_automl.contracts import DatasetProfile, FidelitySpec, PipelineSpec, TaskType
from autonomous_automl.utils.errors import ModelUnavailableError

_CLASSIFICATION_TASKS = {
    TaskType.BINARY_CLASSIFICATION,
    TaskType.MULTICLASS_CLASSIFICATION,
}
_ALL_RESOLVED_TASKS = {*_CLASSIFICATION_TASKS, TaskType.REGRESSION}


class BaseModelAdapter(ABC):
    """Common capability behavior for concrete adapters."""

    name: ClassVar[str]
    family: ClassVar[str]
    _tasks: ClassVar[set[TaskType]]
    _missing: ClassVar[bool] = False
    _native_categorical: ClassVar[bool] = False
    _sparse: ClassVar[bool] = True
    _cost: ClassVar[float] = 1.0

    def supported_tasks(self) -> set[TaskType]:
        return set(self._tasks)

    def supports_missing_values(self) -> bool:
        return self._missing

    def supports_native_categoricals(self) -> bool:
        return self._native_categorical

    def supports_sparse_input(self) -> bool:
        return self._sparse

    def estimated_cost(self) -> float:
        return self._cost

    def is_available(self) -> bool:
        return True

    def is_compatible(self, profile: DatasetProfile, pipeline_spec: PipelineSpec) -> bool:
        return (
            self.is_available()
            and pipeline_spec.model_name == self.name
            and pipeline_spec.task in self._tasks
            and profile.inferred_task == pipeline_spec.task
        )

    @abstractmethod
    def default_params(self, task: TaskType) -> dict[str, object]:
        """Return deterministic, conservative defaults."""

    @abstractmethod
    def suggest_params(
        self,
        trial: optuna.Trial,
        profile: DatasetProfile,
        fidelity: FidelitySpec,
    ) -> dict[str, object]:
        """Suggest a conditional parameter set."""

    @abstractmethod
    def build(
        self,
        params: dict[str, object],
        task: TaskType,
        random_seed: int,
        n_jobs: int,
    ) -> BaseEstimator:
        """Build an unfitted estimator."""

    def _validate_task(self, task: TaskType) -> None:
        if task not in self._tasks:
            raise ValueError(f"model {self.name!r} does not support task {task.value!r}")


class DummyAdapter(BaseModelAdapter):
    name = "dummy"
    family = "baseline"
    _tasks = _ALL_RESOLVED_TASKS
    _cost = 0.01

    def default_params(self, task: TaskType) -> dict[str, object]:
        self._validate_task(task)
        return {"strategy": "prior" if task in _CLASSIFICATION_TASKS else "mean"}

    def suggest_params(
        self,
        trial: optuna.Trial,
        profile: DatasetProfile,
        fidelity: FidelitySpec,
    ) -> dict[str, object]:
        task = profile.inferred_task
        return self.default_params(task)

    def build(
        self,
        params: dict[str, object],
        task: TaskType,
        random_seed: int,
        n_jobs: int,
    ) -> BaseEstimator:
        self._validate_task(task)
        values = self.default_params(task) | params
        if task in _CLASSIFICATION_TASKS:
            return DummyClassifier(strategy=cast(str, values["strategy"]), random_state=random_seed)
        return DummyRegressor(strategy=cast(str, values["strategy"]))


class LogisticRegressionAdapter(BaseModelAdapter):
    name = "logistic_regression"
    family = "linear"
    _tasks = _CLASSIFICATION_TASKS
    _cost = 0.5

    def default_params(self, task: TaskType) -> dict[str, object]:
        self._validate_task(task)
        return {"C": 1.0, "class_weight": None, "max_iter": 500}

    def suggest_params(
        self,
        trial: optuna.Trial,
        profile: DatasetProfile,
        fidelity: FidelitySpec,
    ) -> dict[str, object]:
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "max_iter": fidelity.max_iterations or 500,
        }

    def build(
        self,
        params: dict[str, object],
        task: TaskType,
        random_seed: int,
        n_jobs: int,
    ) -> BaseEstimator:
        self._validate_task(task)
        values = self.default_params(task) | params
        return LogisticRegression(
            C=_float_param(values, "C"),
            class_weight=cast(str | dict[object, float] | None, values["class_weight"]),
            max_iter=_int_param(values, "max_iter"),
            solver="lbfgs",
            random_state=random_seed,
        )


class RidgeAdapter(BaseModelAdapter):
    name = "ridge"
    family = "linear"
    _tasks: ClassVar[set[TaskType]] = {TaskType.REGRESSION}
    _cost = 0.3

    def default_params(self, task: TaskType) -> dict[str, object]:
        self._validate_task(task)
        return {"alpha": 1.0}

    def suggest_params(
        self,
        trial: optuna.Trial,
        profile: DatasetProfile,
        fidelity: FidelitySpec,
    ) -> dict[str, object]:
        return {"alpha": trial.suggest_float("alpha", 1e-4, 1e3, log=True)}

    def build(
        self,
        params: dict[str, object],
        task: TaskType,
        random_seed: int,
        n_jobs: int,
    ) -> BaseEstimator:
        self._validate_task(task)
        values = self.default_params(task) | params
        return Ridge(alpha=_float_param(values, "alpha"), random_state=random_seed)


class ElasticNetAdapter(BaseModelAdapter):
    name = "elastic_net"
    family = "linear"
    _tasks: ClassVar[set[TaskType]] = {TaskType.REGRESSION}
    _cost = 0.5

    def default_params(self, task: TaskType) -> dict[str, object]:
        self._validate_task(task)
        return {"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 1_000}

    def suggest_params(
        self,
        trial: optuna.Trial,
        profile: DatasetProfile,
        fidelity: FidelitySpec,
    ) -> dict[str, object]:
        return {
            "alpha": trial.suggest_float("alpha", 1e-5, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "max_iter": fidelity.max_iterations or 1_000,
        }

    def build(
        self,
        params: dict[str, object],
        task: TaskType,
        random_seed: int,
        n_jobs: int,
    ) -> BaseEstimator:
        self._validate_task(task)
        values = self.default_params(task) | params
        return ElasticNet(
            alpha=_float_param(values, "alpha"),
            l1_ratio=_float_param(values, "l1_ratio"),
            max_iter=_int_param(values, "max_iter"),
            random_state=random_seed,
            selection="cyclic",
        )


class _ForestAdapter(BaseModelAdapter):
    _cost = 2.0

    def default_params(self, task: TaskType) -> dict[str, object]:
        self._validate_task(task)
        return {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        }

    def suggest_params(
        self,
        trial: optuna.Trial,
        profile: DatasetProfile,
        fidelity: FidelitySpec,
    ) -> dict[str, object]:
        maximum_estimators = max(50, fidelity.max_iterations or 400)
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, maximum_estimators, step=25),
            "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 20, 40]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

    def _forest_kwargs(
        self,
        params: dict[str, object],
        task: TaskType,
        random_seed: int,
        n_jobs: int,
    ) -> dict[str, object]:
        values = self.default_params(task) | params
        return {
            "n_estimators": _int_param(values, "n_estimators"),
            "max_depth": values["max_depth"],
            "min_samples_leaf": _int_param(values, "min_samples_leaf"),
            "max_features": values["max_features"],
            "random_state": random_seed,
            "n_jobs": n_jobs,
        }


class RandomForestAdapter(_ForestAdapter):
    name = "random_forest"
    family = "random_forest"
    _tasks = _ALL_RESOLVED_TASKS

    def build(
        self,
        params: dict[str, object],
        task: TaskType,
        random_seed: int,
        n_jobs: int,
    ) -> BaseEstimator:
        self._validate_task(task)
        kwargs = self._forest_kwargs(params, task, random_seed, n_jobs)
        if task in _CLASSIFICATION_TASKS:
            return RandomForestClassifier(**cast(Any, kwargs))
        return RandomForestRegressor(**cast(Any, kwargs))


class ExtraTreesAdapter(_ForestAdapter):
    name = "extra_trees"
    family = "extra_trees"
    _tasks = _ALL_RESOLVED_TASKS

    def build(
        self,
        params: dict[str, object],
        task: TaskType,
        random_seed: int,
        n_jobs: int,
    ) -> BaseEstimator:
        self._validate_task(task)
        kwargs = self._forest_kwargs(params, task, random_seed, n_jobs)
        if task in _CLASSIFICATION_TASKS:
            return ExtraTreesClassifier(**cast(Any, kwargs))
        return ExtraTreesRegressor(**cast(Any, kwargs))


class HistGradientBoostingAdapter(BaseModelAdapter):
    name = "hist_gradient_boosting"
    family = "hist_gradient_boosting"
    _tasks = _ALL_RESOLVED_TASKS
    _missing = True
    _sparse = False
    _cost = 1.5

    def default_params(self, task: TaskType) -> dict[str, object]:
        self._validate_task(task)
        return {
            "learning_rate": 0.1,
            "max_iter": 150,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
        }

    def suggest_params(
        self,
        trial: optuna.Trial,
        profile: DatasetProfile,
        fidelity: FidelitySpec,
    ) -> dict[str, object]:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter": fidelity.max_iterations or trial.suggest_int("max_iter", 75, 400),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
        }

    def build(
        self,
        params: dict[str, object],
        task: TaskType,
        random_seed: int,
        n_jobs: int,
    ) -> BaseEstimator:
        self._validate_task(task)
        values = self.default_params(task) | params
        kwargs = {
            "learning_rate": _float_param(values, "learning_rate"),
            "max_iter": _int_param(values, "max_iter"),
            "max_leaf_nodes": _int_param(values, "max_leaf_nodes"),
            "min_samples_leaf": _int_param(values, "min_samples_leaf"),
            "l2_regularization": _float_param(values, "l2_regularization"),
            "random_state": random_seed,
        }
        if task in _CLASSIFICATION_TASKS:
            return HistGradientBoostingClassifier(**cast(Any, kwargs))
        return HistGradientBoostingRegressor(**cast(Any, kwargs))


class OptionalBoostingAdapter(BaseModelAdapter):
    """Adapter whose provider is imported only inside ``build``."""

    _tasks = _ALL_RESOLVED_TASKS
    _missing = True
    _cost = 2.5
    module_name: ClassVar[str]

    def is_available(self) -> bool:
        return importlib.util.find_spec(self.module_name) is not None

    def default_params(self, task: TaskType) -> dict[str, object]:
        self._validate_task(task)
        return {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
        }

    def suggest_params(
        self,
        trial: optuna.Trial,
        profile: DatasetProfile,
        fidelity: FidelitySpec,
    ) -> dict[str, object]:
        return {
            "n_estimators": fidelity.max_iterations
            or trial.suggest_int("n_estimators", 75, 500, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
        }

    def _load_module(self) -> object:
        if not self.is_available():
            raise ModelUnavailableError(
                f"optional model {self.name!r} requires package {self.module_name!r}"
            )
        return importlib.import_module(self.module_name)


class XGBoostAdapter(OptionalBoostingAdapter):
    name = "xgboost"
    family = "xgboost"
    module_name = "xgboost"

    def build(
        self,
        params: dict[str, object],
        task: TaskType,
        random_seed: int,
        n_jobs: int,
    ) -> BaseEstimator:
        self._validate_task(task)
        module = self._load_module()
        values = self.default_params(task) | params
        cls = getattr(module, "XGBClassifier" if task in _CLASSIFICATION_TASKS else "XGBRegressor")
        return cast(
            BaseEstimator,
            cls(
                **values,
                random_state=random_seed,
                n_jobs=n_jobs,
                tree_method="hist",
                verbosity=0,
            ),
        )


class LightGBMAdapter(OptionalBoostingAdapter):
    name = "lightgbm"
    family = "lightgbm"
    module_name = "lightgbm"

    def build(
        self,
        params: dict[str, object],
        task: TaskType,
        random_seed: int,
        n_jobs: int,
    ) -> BaseEstimator:
        self._validate_task(task)
        module = self._load_module()
        values = self.default_params(task) | params
        cls = getattr(
            module, "LGBMClassifier" if task in _CLASSIFICATION_TASKS else "LGBMRegressor"
        )
        return cast(
            BaseEstimator,
            cls(**values, random_state=random_seed, n_jobs=n_jobs, verbosity=-1),
        )


class CatBoostAdapter(OptionalBoostingAdapter):
    name = "catboost"
    family = "catboost"
    module_name = "catboost"
    _native_categorical = True
    _sparse = False

    def build(
        self,
        params: dict[str, object],
        task: TaskType,
        random_seed: int,
        n_jobs: int,
    ) -> BaseEstimator:
        self._validate_task(task)
        module = self._load_module()
        values = self.default_params(task) | params
        values["iterations"] = values.pop("n_estimators")
        values["depth"] = values.pop("max_depth")
        cls = getattr(
            module, "CatBoostClassifier" if task in _CLASSIFICATION_TASKS else "CatBoostRegressor"
        )
        return cast(
            BaseEstimator,
            cls(
                **values,
                random_seed=random_seed,
                thread_count=n_jobs,
                verbose=False,
                allow_writing_files=False,
            ),
        )


def _float_param(values: dict[str, object], name: str) -> float:
    value = values[name]
    if not isinstance(value, int | float | str):
        raise ValueError(f"model parameter {name!r} must be numeric")
    return float(value)


def _int_param(values: dict[str, object], name: str) -> int:
    value = values[name]
    if not isinstance(value, int | float | str):
        raise ValueError(f"model parameter {name!r} must be an integer")
    converted = int(value)
    if float(value) != converted:
        raise ValueError(f"model parameter {name!r} must be an integer")
    return converted


__all__ = [
    "BaseModelAdapter",
    "CatBoostAdapter",
    "DummyAdapter",
    "ElasticNetAdapter",
    "ExtraTreesAdapter",
    "HistGradientBoostingAdapter",
    "LightGBMAdapter",
    "LogisticRegressionAdapter",
    "OptionalBoostingAdapter",
    "RandomForestAdapter",
    "RidgeAdapter",
    "XGBoostAdapter",
]

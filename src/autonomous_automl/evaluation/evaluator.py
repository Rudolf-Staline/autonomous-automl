"""Leakage-safe cross-validated pipeline execution."""

from __future__ import annotations

import math
import multiprocessing as mp
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing.connection import Connection
from typing import Any, cast

import numpy as np
import pandas as pd
import psutil
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from autonomous_automl.components import ModelRegistry
from autonomous_automl.contracts import (
    DatasetProfile,
    FidelitySpec,
    FoldResult,
    MetricName,
    PipelineSpec,
    TaskType,
    TrialResult,
    TrialStatus,
    ValidationPlan,
)
from autonomous_automl.data import LoadedDataset
from autonomous_automl.evaluation.metrics import ordered_classes, resolve_metric, score_metric
from autonomous_automl.evaluation.oof import OOFAccumulator
from autonomous_automl.pipelines import build_pipeline, pipeline_fingerprint
from autonomous_automl.search.sampling import FidelitySampler
from autonomous_automl.utils.errors import MetricError, TrialTimeoutError
from autonomous_automl.utils.seeds import derive_seed


@dataclass(slots=True)
class EvaluationOutcome:
    """A persistent trial result plus runtime-only OOF predictions."""

    trial_result: TrialResult
    oof_predictions: pd.DataFrame | None


@dataclass(slots=True)
class _FoldExecution:
    predictions: np.ndarray[Any, Any]
    probabilities: np.ndarray[Any, Any] | None
    model_classes: np.ndarray[Any, Any] | None
    fit_seconds: float
    predict_seconds: float
    peak_memory_mb: float


class _RemoteFoldError(Exception):
    def __init__(self, failure_type: str, message: str) -> None:
        super().__init__(message)
        self.failure_type = failure_type


class PipelineEvaluator:
    """Evaluate candidates per materialized fold and contain trial failures."""

    def __init__(self, registry: ModelRegistry | None = None, *, n_jobs: int = 1) -> None:
        if n_jobs < 1:
            raise ValueError("n_jobs must be at least one")
        self.registry = registry or ModelRegistry.default()
        self.n_jobs = n_jobs

    def evaluate(
        self,
        trial_id: str,
        pipeline_spec: PipelineSpec,
        fidelity: FidelitySpec,
        dataset: LoadedDataset,
        profile: DatasetProfile,
        validation_plan: ValidationPlan,
        metric: MetricName | str = MetricName.AUTO,
        *,
        timeout_seconds: float | None = None,
        execution_seeds: Sequence[int] | None = None,
    ) -> EvaluationOutcome:
        """Run folds and contain candidate errors after validating global inputs."""
        _validate_inputs(
            trial_id,
            pipeline_spec,
            fidelity,
            dataset,
            profile,
            validation_plan,
            timeout_seconds,
        )
        definition = resolve_metric(metric, pipeline_spec.task, profile)
        resolved_metric = definition.name
        global_classes = (
            ordered_classes(dataset.y)
            if pipeline_spec.task
            in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}
            else None
        )
        accumulator = OOFAccumulator(
            len(dataset.X),
            pipeline_spec.task,
            classes=global_classes,
            row_ids=dataset.row_ids,
        )
        started_at = datetime.now(UTC)
        started_clock = time.monotonic()
        fold_results: list[FoldResult] = []
        fit_seconds = 0.0
        predict_seconds = 0.0
        peak_memory_mb = _resident_memory_mb()
        spec_fingerprint = pipeline_fingerprint(pipeline_spec)
        fidelity_fingerprint = fidelity.to_json()
        fixed_seeds = None if execution_seeds is None else list(execution_seeds)
        expected_seed_count = len(fidelity.seeds) * fidelity.n_folds
        if fixed_seeds is not None and len(fixed_seeds) != expected_seed_count:
            raise ValueError("execution_seeds must match every fidelity-seed/fold evaluation")
        execution_index = 0

        try:
            for fidelity_seed in fidelity.seeds:
                for fold in validation_plan.folds[: fidelity.n_folds]:
                    _enforce_timeout(started_clock, timeout_seconds)
                    execution_seed = (
                        fixed_seeds[execution_index]
                        if fixed_seeds is not None
                        else derive_seed(
                            pipeline_spec.random_seed,
                            spec_fingerprint,
                            fidelity_fingerprint,
                            fidelity_seed,
                            fold.fold_index,
                        )
                    )
                    execution_index += 1
                    execution_spec = _execution_spec(pipeline_spec, fidelity, execution_seed)
                    validation_positions = np.asarray(fold.validation_positions, dtype=np.int64)
                    training_positions = FidelitySampler.sample_training_positions(
                        dataset,
                        validation_plan,
                        fold,
                        fidelity,
                        seed=execution_seed,
                        task=pipeline_spec.task,
                    )
                    X_train = dataset.X.iloc[training_positions]
                    y_train = dataset.y.iloc[training_positions]
                    X_validation = dataset.X.iloc[validation_positions]
                    y_validation = dataset.y.iloc[validation_positions]
                    if timeout_seconds is None:
                        execution = _execute_fold(
                            execution_spec,
                            profile,
                            self.registry,
                            self.n_jobs,
                            X_train,
                            y_train,
                            X_validation,
                            pipeline_spec.task,
                            definition.requires_probabilities,
                        )
                    else:
                        execution = _execute_fold_isolated(
                            execution_spec,
                            profile,
                            self.registry,
                            self.n_jobs,
                            X_train,
                            y_train,
                            X_validation,
                            pipeline_spec.task,
                            definition.requires_probabilities,
                            timeout_seconds=_remaining_timeout(started_clock, timeout_seconds),
                        )
                    fit_seconds += execution.fit_seconds
                    predict_seconds += execution.predict_seconds
                    peak_memory_mb = max(peak_memory_mb, execution.peak_memory_mb)
                    aligned_probabilities: np.ndarray[Any, Any] | None = None
                    if global_classes is not None:
                        aligned_probabilities = accumulator.add_classification(
                            validation_positions,
                            execution.predictions,
                            execution.probabilities,
                            execution.model_classes,
                        )
                    else:
                        accumulator.add_regression(validation_positions, execution.predictions)

                    metric_value = score_metric(
                        definition,
                        y_validation,
                        predictions=execution.predictions,
                        probabilities=aligned_probabilities,
                        classes=global_classes,
                    )
                    fold_results.append(
                        FoldResult(
                            fold_index=fold.fold_index,
                            seed=execution_seed,
                            score=metric_value.internal_value,
                            user_metric_value=metric_value.user_value,
                            fit_seconds=execution.fit_seconds,
                            predict_seconds=execution.predict_seconds,
                            n_train_rows=len(training_positions),
                            n_validation_rows=len(validation_positions),
                        )
                    )
                    _enforce_timeout(started_clock, timeout_seconds)

            scores = [result.score for result in fold_results]
            return EvaluationOutcome(
                trial_result=TrialResult(
                    trial_id=trial_id,
                    family=pipeline_spec.family,
                    pipeline_spec=pipeline_spec,
                    fidelity=fidelity,
                    status=TrialStatus.COMPLETED,
                    primary_metric=resolved_metric,
                    mean_score=float(np.mean(scores)),
                    std_score=float(np.std(scores, ddof=0)),
                    fold_scores=scores,
                    fold_results=fold_results,
                    fit_seconds=fit_seconds,
                    predict_seconds=predict_seconds,
                    peak_memory_mb=peak_memory_mb,
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                ),
                oof_predictions=accumulator.to_frame(),
            )
        except Exception as error:
            if isinstance(error, TrialTimeoutError):
                fit_seconds = max(
                    fit_seconds,
                    min(time.monotonic() - started_clock, timeout_seconds or math.inf),
                )
            return EvaluationOutcome(
                trial_result=TrialResult(
                    trial_id=trial_id or "invalid-trial",
                    family=pipeline_spec.family,
                    pipeline_spec=pipeline_spec,
                    fidelity=fidelity,
                    status=TrialStatus.FAILED,
                    primary_metric=resolved_metric,
                    fold_scores=[result.score for result in fold_results],
                    fold_results=fold_results,
                    fit_seconds=fit_seconds,
                    predict_seconds=predict_seconds,
                    peak_memory_mb=max(peak_memory_mb, _resident_memory_mb()),
                    failure_type=_failure_type(error),
                    failure_message=_safe_failure_message(error),
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                ),
                oof_predictions=None,
            )


def _execute_fold(
    execution_spec: PipelineSpec,
    profile: DatasetProfile,
    registry: ModelRegistry,
    n_jobs: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    task: TaskType,
    requires_probabilities: bool,
) -> _FoldExecution:
    """Build and fit one fresh clone; called in-process or in a killable worker."""
    template = build_pipeline(
        execution_spec,
        profile,
        n_jobs=n_jobs,
        registry=registry,
    )
    estimator = cast(Pipeline, clone(template))
    fit_started = time.monotonic()
    estimator.fit(X_train, y_train)
    fit_seconds = time.monotonic() - fit_started

    predict_started = time.monotonic()
    predictions = np.asarray(estimator.predict(X_validation))
    probabilities: np.ndarray[Any, Any] | None = None
    model_classes: np.ndarray[Any, Any] | None = None
    if task in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}:
        model = estimator.named_steps["model"]
        predict_proba = getattr(estimator, "predict_proba", None)
        fitted_classes = getattr(model, "classes_", None)
        if callable(predict_proba) and fitted_classes is not None:
            probabilities = np.asarray(predict_proba(X_validation))
            model_classes = np.asarray(fitted_classes)
        elif requires_probabilities:
            raise MetricError(f"model {execution_spec.model_name!r} does not provide probabilities")
    predict_seconds = time.monotonic() - predict_started
    return _FoldExecution(
        predictions=predictions,
        probabilities=probabilities,
        model_classes=model_classes,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        peak_memory_mb=_resident_memory_mb(),
    )


def _execute_fold_isolated(
    execution_spec: PipelineSpec,
    profile: DatasetProfile,
    registry: ModelRegistry,
    n_jobs: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    task: TaskType,
    requires_probabilities: bool,
    *,
    timeout_seconds: float,
) -> _FoldExecution:
    """Run one fold in a child process that can be terminated at the deadline."""
    if timeout_seconds <= 0:
        raise TrialTimeoutError("trial exceeded its configured time limit")
    method = _safe_start_method()
    context = cast(Any, mp.get_context(method))
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_fold_worker,
        args=(
            sender,
            execution_spec,
            profile,
            registry,
            n_jobs,
            X_train,
            y_train,
            X_validation,
            task,
            requires_probabilities,
        ),
        name="automl-fold-worker",
    )
    started = False
    try:
        process.start()
        started = True
        sender.close()
        if not receiver.poll(timeout_seconds):
            _stop_process(process)
            raise TrialTimeoutError("trial exceeded its configured time limit")
        try:
            payload = receiver.recv()
        except EOFError as error:
            process.join(timeout=1.0)
            raise RuntimeError(
                f"fold worker exited without a result (exit code {process.exitcode})"
            ) from error
        process.join(timeout=1.0)
        if process.is_alive():
            _stop_process(process)
        if not isinstance(payload, tuple) or not payload:
            raise RuntimeError("fold worker returned an invalid response")
        if payload[0] == "ok" and len(payload) == 2 and isinstance(payload[1], _FoldExecution):
            return payload[1]
        if payload[0] == "error" and len(payload) == 3:
            raise _RemoteFoldError(str(payload[1]), str(payload[2]))
        raise RuntimeError("fold worker returned an unrecognized response")
    finally:
        receiver.close()
        sender.close()
        if started and process.is_alive():
            _stop_process(process)
        process.close()


def _fold_worker(
    sender: Connection,
    execution_spec: PipelineSpec,
    profile: DatasetProfile,
    registry: ModelRegistry,
    n_jobs: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    task: TaskType,
    requires_probabilities: bool,
) -> None:
    """Child entry point; only sanitized exceptions cross the process boundary."""
    try:
        result = _execute_fold(
            execution_spec,
            profile,
            registry,
            n_jobs,
            X_train,
            y_train,
            X_validation,
            task,
            requires_probabilities,
        )
        sender.send(("ok", result))
    except Exception as error:
        sender.send(("error", type(error).__name__, _safe_failure_message(error)))
    finally:
        sender.close()


def _stop_process(process: mp.Process) -> None:
    process.terminate()
    process.join(timeout=1.0)
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(timeout=1.0)


def _safe_start_method() -> str:
    """Prefer safe process creation, retaining support for REPL and notebook callers."""
    available = mp.get_all_start_methods()
    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)
    interactive = not isinstance(main_file, str) or main_file.startswith("<")
    notebook = isinstance(main_file, str) and "ipykernel" in main_file
    if (interactive or notebook) and "fork" in available:
        return "fork"
    if "forkserver" in available:
        # Preloading the fold worker's module keeps the safe forkserver strategy
        # while avoiding a full pandas/scikit-learn import for every short fold.
        mp.set_forkserver_preload(["autonomous_automl.evaluation.evaluator"])
        return "forkserver"
    return "spawn"


def _validate_inputs(
    trial_id: str,
    pipeline_spec: PipelineSpec,
    fidelity: FidelitySpec,
    dataset: LoadedDataset,
    profile: DatasetProfile,
    validation_plan: ValidationPlan,
    timeout_seconds: float | None,
) -> None:
    if not trial_id.strip():
        raise ValueError("trial_id cannot be blank")
    if pipeline_spec.task is not profile.inferred_task:
        raise ValueError("pipeline task does not match the profiled task")
    if (
        validation_plan.dataset_hash is not None
        and profile.dataset_hash != validation_plan.dataset_hash
    ):
        raise ValueError("validation plan does not match the profiled dataset")
    if len(dataset.X) != profile.n_rows:
        raise ValueError("runtime dataset does not match the profile row count")
    if not validation_plan.folds:
        raise ValueError("validation plan must contain materialized folds")
    if fidelity.n_folds > len(validation_plan.folds):
        raise ValueError("fidelity requests more folds than the validation plan contains")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")


def _execution_spec(
    spec: PipelineSpec,
    fidelity: FidelitySpec,
    execution_seed: int,
) -> PipelineSpec:
    parameters = dict(spec.model_params)
    if fidelity.max_iterations is not None:
        iteration_parameter = {
            "random_forest": "n_estimators",
            "extra_trees": "n_estimators",
            "hist_gradient_boosting": "max_iter",
            "logistic_regression": "max_iter",
            "elastic_net": "max_iter",
            "xgboost": "n_estimators",
            "lightgbm": "n_estimators",
            "catboost": "n_estimators",
        }.get(spec.model_name)
        if iteration_parameter is not None:
            current = parameters.get(iteration_parameter)
            if isinstance(current, int | float):
                parameters[iteration_parameter] = min(int(current), fidelity.max_iterations)
            else:
                parameters[iteration_parameter] = fidelity.max_iterations
    return spec.model_copy(
        update={"random_seed": execution_seed, "model_params": parameters},
        deep=True,
    )


def _enforce_timeout(started_clock: float, timeout_seconds: float | None) -> None:
    if timeout_seconds is not None and time.monotonic() - started_clock > timeout_seconds:
        raise TrialTimeoutError("trial exceeded its configured time limit")


def _remaining_timeout(started_clock: float, timeout_seconds: float) -> float:
    remaining = timeout_seconds - (time.monotonic() - started_clock)
    if remaining <= 0:
        raise TrialTimeoutError("trial exceeded its configured time limit")
    return remaining


def _resident_memory_mb() -> float:
    try:
        return float(psutil.Process().memory_info().rss / (1024**2))
    except (OSError, psutil.Error):
        return 0.0


def _safe_failure_message(error: Exception) -> str:
    message = re.sub(r"(['\"]).*?\1", "<value>", str(error), flags=re.DOTALL)
    message = re.sub(r"\s+", " ", message).strip()
    if not message:
        message = "trial execution failed without an error message"
    return message[:500]


def _failure_type(error: Exception) -> str:
    if isinstance(error, _RemoteFoldError):
        return error.failure_type
    return type(error).__name__


__all__ = ["EvaluationOutcome", "PipelineEvaluator"]

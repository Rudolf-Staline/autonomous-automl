"""Public end-to-end orchestration for Build Week runs."""

from __future__ import annotations

import platform
import sys
import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

import numpy as np
import optuna
import pandas as pd

from autonomous_automl.api.trust import (
    TRUST_CERTIFICATE_HTML_PATH,
    TRUST_CERTIFICATE_JSON_PATH,
    create_or_validate_trust_artifacts,
)
from autonomous_automl.components import ModelRegistry
from autonomous_automl.contracts import (
    AutoMLConfig,
    DatasetProfile,
    LeakageReport,
    MetricName,
    PipelineSpec,
    RunManifest,
    RunResult,
    RunStatus,
    RuntimeTelemetry,
    SearchStopReason,
    TrialResult,
    TrialStatus,
    ValidationAudit,
    ValidationPlan,
)
from autonomous_automl.data import LoadedDataset, load_dataset
from autonomous_automl.evaluation import (
    PipelineEvaluator,
    compute_observed_trust_gap,
    disabled_trust_gap,
    explain_pipeline_selection,
    resolve_metric,
)
from autonomous_automl.evaluation.confirmation import FinalistConfirmer
from autonomous_automl.evaluation.finalization import FinalTrainer, FinalTrainingResult
from autonomous_automl.evaluation.selection import (
    Leaderboard,
    LeaderboardEntry,
    build_leaderboard,
    select_finalist,
)
from autonomous_automl.evaluation.trust_gap import (
    RAW_PREDICTIONS_PATH,
    RAW_PROTOCOL_PATH,
    TRUST_GAP_RESULT_PATH,
)
from autonomous_automl.pipelines import generate_initial_candidates
from autonomous_automl.profiling import detect_leakage, profile_dataset
from autonomous_automl.reporting import render_run_report
from autonomous_automl.runtime import (
    RUNTIME_STATE_KEY,
    RUNTIME_TELEMETRY_PATH,
    ProgressCallback,
    ProgressReporter,
    RunStage,
    RuntimeTelemetryRecorder,
)
from autonomous_automl.search import (
    BudgetManager,
    FamilyAllocator,
    FidelityPolicy,
    FidelityScheduler,
    OptunaFamilyOptimizer,
    SearchController,
)
from autonomous_automl.tracking import (
    ArtifactRecord,
    ArtifactRegistration,
    ArtifactStore,
    ExperimentStore,
    ResumeManager,
    RetryableTrial,
)
from autonomous_automl.utils.atomic import atomic_write_text
from autonomous_automl.utils.errors import (
    ArtifactValidationError,
    ConfigurationError,
    PlannedInterruption,
    RunExecutionError,
)
from autonomous_automl.utils.hashing import sha256_file
from autonomous_automl.utils.json import JsonValue, canonical_json_dumps
from autonomous_automl.validation import ValidationPlanner, audit_validation

REGISTRY_FILENAME = "registry.sqlite3"
OPTUNA_FILENAME = "optuna.sqlite3"
MANIFEST_FILENAME = "manifest.json"
REPORT_FILENAME = "report.html"
SELECTION_EXPLANATION_PATH = "selection_explanation.json"


@dataclass(slots=True)
class _RuntimeContext:
    config: AutoMLConfig
    run_directory: Path
    run_id: str
    dataset: LoadedDataset
    profile: DatasetProfile
    leakage: LeakageReport
    validation_plan: ValidationPlan
    validation_audit: ValidationAudit
    metric: MetricName
    store: ExperimentStore
    artifact_store: ArtifactStore
    reporter: ProgressReporter
    budget: BudgetManager


@dataclass(slots=True)
class _SearchComponents:
    registry: ModelRegistry
    policy: FidelityPolicy
    scheduler: FidelityScheduler
    allocator: FamilyAllocator
    optimizers: dict[str, OptunaFamilyOptimizer]


class AutoMLRun:
    """Run the complete CSV-to-artifacts workflow through stable contracts."""

    def __init__(
        self,
        config: AutoMLConfig,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.config = _normalize_config(config)
        self.progress_callback = progress_callback

    def fit(
        self,
        train_paths: str | Path | Sequence[str | Path],
        *,
        interrupt_after_trials: int | None = None,
    ) -> RunResult:
        """Create and execute a new run, optionally stopping at a tested checkpoint."""

        if interrupt_after_trials is not None and interrupt_after_trials < 1:
            raise ValueError("interrupt_after_trials must be positive")
        run_directory = self.config.output_dir
        _prepare_new_run_directory(run_directory)
        started_clock = _monotonic()
        reporter = ProgressReporter(
            self.config.budget_seconds,
            callback=self.progress_callback,
            log_path=run_directory / "logs" / "run.jsonl",
            started_clock=started_clock,
        )
        store: ExperimentStore | None = None
        manifest: RunManifest | None = None
        try:
            reporter.emit(RunStage.LOADING, "Loading and validating training CSV files")
            dataset = load_dataset(train_paths, self.config)
            reporter.emit(RunStage.PROFILING, "Profiling training data")
            profile = profile_dataset(dataset, self.config.task)
            reporter.emit(
                RunStage.PROFILING,
                (
                    f"Profile ready: {profile.n_rows} rows, {profile.n_features} features, "
                    f"task={profile.inferred_task.value}"
                ),
            )
            reporter.emit(RunStage.LEAKAGE, "Detecting leakage and identifier risks")
            leakage = detect_leakage(dataset, profile)
            reporter.emit(
                RunStage.LEAKAGE,
                (
                    f"Diagnostics ready: {len(leakage.findings)} alerts, "
                    f"{len(leakage.excluded_columns)} columns excluded"
                ),
            )
            reporter.emit(RunStage.VALIDATION, "Materializing leakage-safe validation folds")
            validation_plan = ValidationPlanner(max_splits=3).plan(
                dataset,
                profile,
                self.config,
            )
            validation_audit = audit_validation(dataset, validation_plan)
            metric = resolve_metric(self.config.metric, profile.inferred_task, profile).name
            reporter.emit(
                RunStage.VALIDATION,
                (
                    f"Validation ready: {validation_plan.splitter_name}, "
                    f"{validation_plan.n_splits} folds, metric={metric.value}"
                ),
            )
            run_id = _new_run_id(profile)
            now = datetime.now(UTC)
            manifest = RunManifest(
                run_id=run_id,
                package_version=_package_version(),
                status=RunStatus.RUNNING,
                created_at=now,
                updated_at=now,
                configuration=self.config,
                source_hashes=dataset.bundle.source_hashes,
                dataset=dataset.bundle,
                dataset_profile=profile,
                leakage_report=leakage,
                validation_plan=validation_plan,
                validation_audit=validation_audit,
                dependency_versions=_dependency_versions(),
                random_seed=self.config.random_seed,
            )
            manifest.write_json(run_directory / MANIFEST_FILENAME)
            store = ExperimentStore.open(run_directory / REGISTRY_FILENAME)
            store.create_run(manifest)
            artifact_store = ArtifactStore(
                run_directory,
                registry=store,
                run_id=run_id,
            )
            initial_artifacts = _persist_initial_artifacts(
                artifact_store,
                store,
                run_id,
                self.config,
                profile,
                leakage,
                validation_plan,
                validation_audit,
            )
            manifest = manifest.model_copy(update={"artifacts": initial_artifacts}, deep=True)
            store.update_manifest(manifest)
            manifest.write_json(run_directory / MANIFEST_FILENAME)
            budget = BudgetManager(
                self.config.budget_seconds,
                confirmation_fraction=0.25,
                finalization_fraction=0.20,
            )
            context = _RuntimeContext(
                config=self.config,
                run_directory=run_directory,
                run_id=run_id,
                dataset=dataset,
                profile=profile,
                leakage=leakage,
                validation_plan=validation_plan,
                validation_audit=validation_audit,
                metric=metric,
                store=store,
                artifact_store=artifact_store,
                reporter=reporter,
                budget=budget,
            )
            return self._execute(
                context,
                manifest,
                interrupt_after_trials=interrupt_after_trials,
                retryable_trials=(),
            )
        except PlannedInterruption:
            raise
        except KeyboardInterrupt:
            if store is not None and manifest is not None:
                _persist_interrupted(store, run_directory, manifest, reporter)
            raise
        except Exception as error:
            if store is not None and manifest is not None:
                _persist_failed(store, run_directory, manifest, reporter, error)
            raise

    @classmethod
    def resume(
        cls,
        run_directory: str | Path,
        *,
        additional_budget_seconds: float | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> RunResult:
        """Resume an interrupted run without repeating completed candidates."""

        manager = ResumeManager()
        state = manager.prepare(
            run_directory,
            additional_budget_seconds=additional_budget_seconds,
            acquire_lease=True,
        )
        directory = state.run_directory
        reporter = ProgressReporter(
            state.progress.budget_seconds,
            callback=progress_callback,
            log_path=directory / "logs" / "run.jsonl",
        )
        store = ExperimentStore.open(directory / REGISTRY_FILENAME)
        try:
            reporter.emit(
                RunStage.LOADING,
                "Reloading verified CSV sources for resume",
                remaining_seconds=state.remaining_budget_seconds,
            )
            config = _normalize_config(
                state.manifest.configuration.model_copy(
                    update={"output_dir": directory},
                    deep=True,
                )
            )
            dataset = load_dataset(state.manifest.dataset.train_paths, config)
            context = _RuntimeContext(
                config=config,
                run_directory=directory,
                run_id=state.manifest.run_id,
                dataset=dataset,
                profile=state.manifest.dataset_profile,
                leakage=state.manifest.leakage_report,
                validation_plan=state.manifest.validation_plan,
                validation_audit=cast(ValidationAudit, state.manifest.validation_audit),
                metric=resolve_metric(
                    config.metric,
                    state.manifest.dataset_profile.inferred_task,
                    state.manifest.dataset_profile,
                ).name,
                store=store,
                artifact_store=ArtifactStore(
                    directory,
                    registry=store,
                    run_id=state.manifest.run_id,
                ),
                reporter=reporter,
                budget=BudgetManager(
                    state.progress.budget_seconds,
                    consumed_seconds=state.progress.consumed_seconds,
                    confirmation_fraction=0.25,
                    finalization_fraction=0.20,
                ),
            )
            runner = cls(config, progress_callback=progress_callback)
            search_retryables = tuple(
                trial
                for trial in state.retryable_trials
                if not trial.trial_id.startswith("finalist-l")
            )
            return runner._execute(
                context,
                state.manifest,
                interrupt_after_trials=None,
                retryable_trials=search_retryables,
                lease_epoch=state.progress.lease_epoch,
            )
        except KeyboardInterrupt:
            manifest = store.get_manifest(state.manifest.run_id)
            _persist_interrupted(store, directory, manifest, reporter)
            raise
        except Exception as error:
            manifest = store.get_manifest(state.manifest.run_id)
            _persist_failed(store, directory, manifest, reporter, error)
            raise
        finally:
            manager.release(state)

    def _execute(
        self,
        context: _RuntimeContext,
        manifest: RunManifest,
        *,
        interrupt_after_trials: int | None,
        retryable_trials: Sequence[RetryableTrial],
        lease_epoch: int | None = None,
    ) -> RunResult:
        progress = context.store.get_run_progress(context.run_id)
        telemetry = RuntimeTelemetryRecorder.from_scheduler_state(
            progress.budget_seconds,
            manifest.scheduler_state,
        )
        try:
            components = _build_search_components(context)
            evaluator = PipelineEvaluator(components.registry, n_jobs=context.config.n_jobs)
            callback = _trial_progress_callback(context)
            controller = SearchController(
                run_id=context.run_id,
                dataset=context.dataset,
                profile=context.profile,
                validation_plan=context.validation_plan,
                metric=context.metric,
                optimizers=components.optimizers,
                allocator=components.allocator,
                scheduler=components.scheduler,
                evaluator=evaluator,
                store=context.store,
                budget=context.budget,
                trial_timeout_seconds=context.config.trial_timeout_seconds,
                lease_epoch=lease_epoch,
                retryable_trials=retryable_trials,
                trial_completed_callback=callback,
            )
            context.reporter.emit(
                RunStage.SEARCH,
                f"Searching {len(components.optimizers)} compatible pipeline families",
                remaining_seconds=controller.budget.remaining_seconds,
            )
            controller.run(max_trials=interrupt_after_trials)
            context.budget = controller.budget
            if interrupt_after_trials is not None and (
                len(context.store.list_trials(context.run_id)) >= interrupt_after_trials
            ):
                telemetry.stop_search(
                    SearchStopReason.USER_INTERRUPTED,
                    budget_remaining_seconds=context.budget.remaining_seconds,
                )
                partial = telemetry.snapshot(
                    context.store.list_trials(context.run_id),
                    include_finalization=False,
                )
                interrupted = _interrupted_manifest(
                    manifest,
                    context.store,
                    context.budget,
                    partial,
                )
                context.store.update_manifest(interrupted)
                interrupted.write_json(context.run_directory / MANIFEST_FILENAME)
                context.reporter.emit(
                    RunStage.INTERRUPTED,
                    "Requested checkpoint reached; run can now be resumed",
                    remaining_seconds=context.budget.remaining_seconds,
                )
                raise PlannedInterruption(str(context.run_directory))

            search_trials = context.store.list_trials(context.run_id)
            context.reporter.emit(
                RunStage.CONFIRMATION,
                "Re-evaluating the strongest pipeline and the baseline",
                remaining_seconds=context.budget.remaining_seconds,
            )
            confirmer = FinalistConfirmer(
                run_id=context.run_id,
                evaluator=evaluator,
                dataset=context.dataset,
                profile=context.profile,
                validation_plan=context.validation_plan,
                store=context.store,
                budget=context.budget,
                metric=context.metric,
                lease_epoch=context.store.get_run_progress(context.run_id).lease_epoch,
                fidelity_policy=components.policy,
                top_n=1,
                trial_timeout_seconds=context.config.trial_timeout_seconds,
                component_states_supplier=lambda: context.store.load_component_states(
                    context.run_id
                ),
            )
            confirmer.run(search_trials)
            trials = context.store.list_trials(context.run_id)
            leaderboard = build_leaderboard(trials)
            if not leaderboard.entries:
                raise RunExecutionError("all compatible pipelines failed; inspect trials.csv")
            selection = select_finalist(
                leaderboard,
                context.config.optimization_profile,
                require_full_fidelity=bool(leaderboard.finalists),
            )
            telemetry.stop_search(
                controller.stop_reason or SearchStopReason.CONTROLLER_STOPPED,
                budget_remaining_seconds=context.budget.remaining_seconds,
            )
            telemetry.start_finalization()
            context.reporter.emit(
                RunStage.FINAL_TRAINING,
                f"Retraining selected {selection.selected.model_name} pipeline on all rows",
                remaining_seconds=context.budget.remaining_seconds,
            )
            final_result = FinalTrainer(
                components.registry,
                n_jobs=context.config.n_jobs,
            ).fit(
                selection.selected.pipeline_spec,
                context.dataset,
                context.profile,
            )
            return _persist_final_run(
                context,
                manifest,
                leaderboard,
                selection.selected,
                final_result,
                telemetry,
            )
        except PlannedInterruption:
            raise
        except KeyboardInterrupt:
            if not telemetry.search_stopped:
                telemetry.stop_search(
                    SearchStopReason.USER_INTERRUPTED,
                    budget_remaining_seconds=context.budget.remaining_seconds,
                )
            snapshot = telemetry.snapshot(
                context.store.list_trials(context.run_id),
                include_finalization=telemetry.finalization_started,
            )
            _persist_runtime_checkpoint(context, snapshot)
            raise
        except Exception:
            if not telemetry.search_stopped:
                telemetry.stop_search(
                    SearchStopReason.FATAL_ERROR,
                    budget_remaining_seconds=context.budget.remaining_seconds,
                )
            else:
                telemetry.mark_fatal_error()
            snapshot = telemetry.snapshot(
                context.store.list_trials(context.run_id),
                include_finalization=telemetry.finalization_started,
            )
            _persist_runtime_checkpoint(context, snapshot)
            raise


def _build_search_components(context: _RuntimeContext) -> _SearchComponents:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    registry = ModelRegistry.default()
    specifications = generate_initial_candidates(
        context.profile,
        task=context.profile.inferred_task,
        random_seed=context.config.random_seed,
        excluded_columns=context.leakage.excluded_columns,
        registry=registry,
    )
    grouped: defaultdict[str, list[PipelineSpec]] = defaultdict(list)
    for specification in specifications:
        grouped[specification.family].append(specification)
    if "baseline" not in grouped:
        raise ConfigurationError("no compatible naive baseline could be constructed")
    if not grouped:
        raise ConfigurationError("no compatible pipeline could be constructed")
    policy = FidelityPolicy(
        available_folds=len(context.validation_plan.folds),
        random_seed=context.config.random_seed,
        confirmation_seed_count=2,
    )
    optimizers = {
        family: OptunaFamilyOptimizer(
            family,
            templates,
            context.profile,
            context.run_directory / OPTUNA_FILENAME,
            random_seed=context.config.random_seed,
            registry=registry,
            study_prefix=context.run_id,
        )
        for family, templates in grouped.items()
    }
    return _SearchComponents(
        registry=registry,
        policy=policy,
        scheduler=FidelityScheduler(policy),
        allocator=FamilyAllocator(list(grouped)),
        optimizers=optimizers,
    )


def _trial_progress_callback(context: _RuntimeContext):
    def callback(result: TrialResult, budget: BudgetManager) -> None:
        trials = context.store.list_trials(context.run_id)
        completed = [trial for trial in trials if trial.status is TrialStatus.COMPLETED]
        best = max(
            (trial.mean_score for trial in completed if trial.mean_score is not None),
            default=None,
        )
        context.reporter.emit(
            RunStage.SEARCH,
            f"Trial {result.status.value}: {result.family}/{result.pipeline_spec.model_name}",
            remaining_seconds=budget.remaining_seconds,
            completed_trials=len(completed),
            failed_trials=sum(trial.status is TrialStatus.FAILED for trial in trials),
            interrupted_trials=sum(trial.status is TrialStatus.INTERRUPTED for trial in trials),
            best_internal_score=best,
        )

    return callback


def _persist_final_run(
    context: _RuntimeContext,
    previous_manifest: RunManifest,
    leaderboard: Leaderboard,
    selected: LeaderboardEntry,
    final_result: FinalTrainingResult,
    telemetry: RuntimeTelemetryRecorder,
) -> RunResult:
    context.reporter.emit(
        RunStage.ARTIFACTS,
        "Saving and verifying final artifacts",
        remaining_seconds=context.budget.remaining_seconds,
    )
    artifacts = dict(previous_manifest.artifacts)
    pipeline_record = FinalTrainer.save_pipeline(
        final_result,
        context.artifact_store,
        relative_path="best_pipeline.joblib",
    )
    pipeline_registration = _register_record(
        context.store,
        context.run_id,
        pipeline_record,
    )
    artifacts["best_pipeline"] = pipeline_record.relative_path

    specification_record = context.artifact_store.write_contract(
        "best_pipeline_spec",
        selected.pipeline_spec,
        relative_path="best_pipeline_spec.json",
    )
    _register_record(context.store, context.run_id, specification_record)
    artifacts["best_pipeline_spec"] = specification_record.relative_path

    trials = context.store.list_trials(context.run_id)
    selection_explanation = explain_pipeline_selection(
        leaderboard,
        trials,
        context.config.optimization_profile,
        selected_trial_id=selected.trial_id,
        pipeline_size_bytes=pipeline_record.size_bytes,
    )
    selection_record = context.artifact_store.write_contract(
        "selection_explanation",
        selection_explanation,
        relative_path=SELECTION_EXPLANATION_PATH,
    )
    _register_record(context.store, context.run_id, selection_record)
    artifacts["selection_explanation"] = selection_record.relative_path

    selected_trial = _trial_by_id(trials, selected.trial_id)
    if context.config.compute_trust_gap:
        trust_gap_evaluation = compute_observed_trust_gap(
            selected_trial=selected_trial,
            dataset=context.dataset,
            profile=context.profile,
            leakage_report=context.leakage,
            validation_plan=context.validation_plan,
            metric=context.metric,
            timeout_seconds=context.config.trust_gap_timeout_seconds,
            n_jobs=context.config.n_jobs,
        )
    else:
        trust_gap_evaluation = None
    trust_gap = (
        disabled_trust_gap(context.metric)
        if trust_gap_evaluation is None
        else trust_gap_evaluation.result
    )
    if trust_gap_evaluation is not None and trust_gap_evaluation.raw_protocol is not None:
        raw_protocol_record = context.artifact_store.write_contract(
            "trust_gap_raw_protocol",
            trust_gap_evaluation.raw_protocol,
            relative_path=RAW_PROTOCOL_PATH,
        )
        _register_record(context.store, context.run_id, raw_protocol_record)
        artifacts["trust_gap_raw_protocol"] = raw_protocol_record.relative_path
    if trust_gap_evaluation is not None and trust_gap_evaluation.raw_predictions is not None:
        raw_predictions_record = context.artifact_store.write_dataframe(
            "trust_gap_raw_predictions",
            trust_gap_evaluation.raw_predictions,
            relative_path=RAW_PREDICTIONS_PATH,
        )
        _register_record(context.store, context.run_id, raw_predictions_record)
        artifacts["trust_gap_raw_predictions"] = raw_predictions_record.relative_path
    trust_gap_record = context.artifact_store.write_contract(
        "trust_gap",
        trust_gap,
        relative_path=TRUST_GAP_RESULT_PATH,
    )
    _register_record(context.store, context.run_id, trust_gap_record)
    artifacts["trust_gap"] = trust_gap_record.relative_path
    if context.store.list_trials(context.run_id) != trials:
        raise RunExecutionError("Trust Gap diagnostic modified the main trial registry")

    leaderboard_frame = _leaderboard_frame(leaderboard, trials)
    leaderboard_record = context.artifact_store.write_dataframe(
        "leaderboard",
        leaderboard_frame,
        relative_path="leaderboard.csv",
    )
    _register_record(context.store, context.run_id, leaderboard_record)
    artifacts["leaderboard"] = leaderboard_record.relative_path

    trials_record = context.artifact_store.write_dataframe(
        "trials",
        _trials_frame(trials),
        relative_path="trials.csv",
    )
    _register_record(context.store, context.run_id, trials_record)
    artifacts["trials"] = trials_record.relative_path

    predictions_path: Path | None = None
    if final_result.prediction_frame is not None:
        prediction_record = context.artifact_store.write_dataframe(
            "predictions",
            final_result.prediction_frame,
            relative_path="predictions.csv",
        )
        _register_record(context.store, context.run_id, prediction_record)
        artifacts["predictions"] = prediction_record.relative_path
        predictions_path = context.run_directory / prediction_record.relative_path

    environment_record = context.artifact_store.write_text(
        "environment",
        canonical_json_dumps(cast(JsonValue, _environment_payload())),
        relative_path="environment.json",
        kind="environment",
        media_type="application/json",
    )
    _register_record(context.store, context.run_id, environment_record)
    artifacts["environment"] = environment_record.relative_path
    artifacts.update(
        {
            "manifest": MANIFEST_FILENAME,
            "report": REPORT_FILENAME,
            "structured_log": "logs/run.jsonl",
            "trust_certificate_json": TRUST_CERTIFICATE_JSON_PATH,
            "trust_certificate_html": TRUST_CERTIFICATE_HTML_PATH,
        }
    )

    restored_pipeline = FinalTrainer.load_pipeline(
        context.artifact_store,
        pipeline_registration,
    )
    _verify_final_predictions(restored_pipeline, context.dataset, final_result)
    user_score = _user_metric_value(selected_trial, context.metric)
    if user_score is None:
        raise RunExecutionError("selected trial has no aggregate score")
    progress = context.store.synchronize_consumed_seconds(
        context.run_id,
        context.budget.consumed_seconds,
    )
    states = context.store.load_component_states(context.run_id)
    states["tracking"] = {
        "checkpoint_revision": progress.checkpoint_revision,
        "consumed_seconds": progress.consumed_seconds,
        "effective_budget_seconds": progress.budget_seconds,
    }
    runtime_telemetry = telemetry.snapshot(trials, include_finalization=True)
    states[RUNTIME_STATE_KEY] = runtime_telemetry.to_json_value()
    runtime_record = context.artifact_store.write_contract(
        "runtime_telemetry",
        runtime_telemetry,
        relative_path=RUNTIME_TELEMETRY_PATH,
    )
    _register_record(context.store, context.run_id, runtime_record)
    artifacts["runtime_telemetry"] = runtime_record.relative_path
    now = datetime.now(UTC)
    completed_manifest = previous_manifest.model_copy(
        update={
            "status": RunStatus.COMPLETED,
            "updated_at": now,
            "completed_at": now,
            "best_pipeline": selected.pipeline_spec,
            "finalist_trial_ids": [entry.trial_id for entry in leaderboard.finalists],
            "metrics": {
                "best_score": user_score,
                "best_internal_score": cast(float, selected.mean_score),
                "best_std_score": selected.std_score,
                "total_trials": float(len(trials)),
                "failed_trials": float(sum(trial.status is TrialStatus.FAILED for trial in trials)),
            },
            "artifacts": artifacts,
            "scheduler_state": states,
        },
        deep=True,
    )
    context.store.update_manifest(completed_manifest)
    completed_manifest.write_json(context.run_directory / MANIFEST_FILENAME)
    context.reporter.emit(
        RunStage.COMPLETED,
        "Run completed and predictions verified from the serialized pipeline",
        remaining_seconds=max(0.0, progress.budget_seconds - progress.consumed_seconds),
        completed_trials=sum(trial.status is TrialStatus.COMPLETED for trial in trials),
        failed_trials=sum(trial.status is TrialStatus.FAILED for trial in trials),
        interrupted_trials=sum(trial.status is TrialStatus.INTERRUPTED for trial in trials),
        best_internal_score=selected.mean_score,
    )
    log_registration = _register_existing(
        context.store,
        context.run_id,
        context.run_directory,
        name="structured_log",
        kind="log",
        relative_path="logs/run.jsonl",
        media_type="application/x-ndjson",
    )
    del log_registration
    trust_artifacts = create_or_validate_trust_artifacts(
        context.run_directory,
        persist_if_missing=True,
    )
    if not trust_artifacts.persisted:
        raise ArtifactValidationError("trust certificate was not persisted")
    report = render_run_report(context.store, context.run_id)
    atomic_write_text(context.run_directory / REPORT_FILENAME, report)
    _register_existing(
        context.store,
        context.run_id,
        context.run_directory,
        name="report",
        kind="report",
        relative_path=REPORT_FILENAME,
        media_type="text/html",
    )
    _register_existing(
        context.store,
        context.run_id,
        context.run_directory,
        name="manifest",
        kind="manifest",
        relative_path=MANIFEST_FILENAME,
        media_type="application/json",
    )
    return RunResult(
        run_id=context.run_id,
        status=RunStatus.COMPLETED,
        output_dir=context.run_directory,
        manifest_path=context.run_directory / MANIFEST_FILENAME,
        best_pipeline_path=context.run_directory / pipeline_record.relative_path,
        best_pipeline_spec_path=context.run_directory / specification_record.relative_path,
        leaderboard_path=context.run_directory / leaderboard_record.relative_path,
        notebook_path=None,
        report_path=context.run_directory / REPORT_FILENAME,
        primary_metric=context.metric,
        best_score=user_score,
        predictions_path=predictions_path,
    )


def _persist_initial_artifacts(
    artifact_store: ArtifactStore,
    store: ExperimentStore,
    run_id: str,
    config: AutoMLConfig,
    profile: DatasetProfile,
    leakage: LeakageReport,
    validation_plan: ValidationPlan,
    validation_audit: ValidationAudit,
) -> dict[str, str]:
    values = (
        ("configuration", config, "configuration.json"),
        ("dataset_profile", profile, "dataset_profile.json"),
        ("leakage_report", leakage, "leakage_report.json"),
        ("validation_plan", validation_plan, "validation_plan.json"),
        ("validation_audit", validation_audit, "validation_audit.json"),
    )
    paths: dict[str, str] = {}
    for name, value, path in values:
        record = artifact_store.write_contract(name, value, relative_path=path)
        _register_record(store, run_id, record)
        paths[name] = record.relative_path
    return paths


def _register_record(
    store: ExperimentStore,
    run_id: str,
    record: ArtifactRecord,
) -> ArtifactRegistration:
    return store.register_artifact(
        run_id,
        name=record.name,
        kind=record.kind,
        relative_path=record.relative_path,
        sha256=record.sha256,
        size_bytes=record.size_bytes,
        media_type=record.media_type,
    )


def _register_existing(
    store: ExperimentStore,
    run_id: str,
    run_directory: Path,
    *,
    name: str,
    kind: str,
    relative_path: str,
    media_type: str,
) -> ArtifactRegistration:
    path = run_directory / relative_path
    return store.register_artifact(
        run_id,
        name=name,
        kind=kind,
        relative_path=relative_path,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        media_type=media_type,
    )


def _leaderboard_frame(
    leaderboard: Leaderboard,
    trials: list[TrialResult],
) -> pd.DataFrame:
    frame = leaderboard.to_frame()
    by_id = {trial.trial_id: trial for trial in trials}
    frame.insert(
        7,
        "metric_value",
        [
            _user_metric_value(by_id[trial_id], cast(MetricName, leaderboard.primary_metric))
            for trial_id in frame["trial_id"]
        ],
    )
    return frame


def _trials_frame(trials: list[TrialResult]) -> pd.DataFrame:
    rows = [
        {
            "trial_id": result.trial_id,
            "status": result.status.value,
            "family": result.family,
            "model_name": result.pipeline_spec.model_name,
            "fidelity_level": result.fidelity.level,
            "metric": result.primary_metric.value,
            "internal_score": result.mean_score,
            "metric_value": _user_metric_value(result, result.primary_metric),
            "std_score": result.std_score,
            "fit_seconds": result.fit_seconds,
            "predict_seconds": result.predict_seconds,
            "peak_memory_mb": result.peak_memory_mb,
            "failure_type": result.failure_type,
            "failure_message": result.failure_message,
        }
        for result in trials
    ]
    return pd.DataFrame(rows)


def _user_metric_value(result: TrialResult, metric: MetricName) -> float | None:
    if result.fold_results:
        return float(
            sum(fold.user_metric_value for fold in result.fold_results) / len(result.fold_results)
        )
    if result.mean_score is None:
        return None
    if metric in {MetricName.RMSE, MetricName.MAE, MetricName.LOG_LOSS}:
        return -result.mean_score
    return result.mean_score


def _verify_final_predictions(
    restored_pipeline: object,
    dataset: LoadedDataset,
    result: FinalTrainingResult,
) -> None:
    if dataset.test_X is None or result.predictions is None:
        return
    predict = getattr(restored_pipeline, "predict", None)
    if not callable(predict):
        raise ArtifactValidationError("serialized final pipeline does not expose predict")
    reproduced = np.asarray(predict(dataset.test_X)).reshape(-1)
    expected = np.asarray(result.predictions).reshape(-1)
    if np.issubdtype(expected.dtype, np.number):
        if not np.allclose(
            np.asarray(reproduced, dtype=float),
            np.asarray(expected, dtype=float),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ArtifactValidationError("serialized pipeline predictions do not reproduce")
    elif not np.array_equal(reproduced, expected):
        raise ArtifactValidationError("serialized pipeline predictions do not reproduce")


def _trial_by_id(trials: list[TrialResult], trial_id: str) -> TrialResult:
    for result in trials:
        if result.trial_id == trial_id:
            return result
    raise RunExecutionError("selected trial is missing from the persisted registry")


def _normalize_config(config: AutoMLConfig) -> AutoMLConfig:
    updates: dict[str, object] = {"output_dir": config.output_dir.expanduser().resolve()}
    if config.test_path is not None:
        updates["test_path"] = config.test_path.expanduser().resolve()
    return config.model_copy(update=updates, deep=True)


def _prepare_new_run_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        if (directory / MANIFEST_FILENAME).exists():
            raise ConfigurationError(
                f"output directory already contains a run; use resume: {directory}"
            )
        raise ConfigurationError(f"output directory must be empty: {directory}")


def _new_run_id(profile: DatasetProfile) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"run-{timestamp}-{profile.dataset_hash[:8]}-{uuid.uuid4().hex[:8]}"


def _dependency_versions() -> dict[str, str]:
    dependencies = ["autonomous-automl", "pandas", "scikit-learn", "optuna", "joblib"]
    values = {"python": platform.python_version()}
    for dependency in dependencies:
        try:
            values[dependency] = version(dependency)
        except PackageNotFoundError:
            values[dependency] = "source-tree"
    return values


def _package_version() -> str:
    try:
        return version("autonomous-automl")
    except PackageNotFoundError:
        return "0.1.0"


def _environment_payload() -> dict[str, JsonValue]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": cast(dict[str, JsonValue], _dependency_versions()),
        "reproduction": [
            "uv sync --frozen",
            "uv run automl validate-artifacts <run-directory>",
        ],
    }


def _interrupted_manifest(
    manifest: RunManifest,
    store: ExperimentStore,
    budget: BudgetManager,
    runtime_telemetry: RuntimeTelemetry,
) -> RunManifest:
    states = store.load_component_states(manifest.run_id)
    progress = store.synchronize_consumed_seconds(manifest.run_id, budget.consumed_seconds)
    states["tracking"] = {
        "checkpoint_revision": progress.checkpoint_revision,
        "consumed_seconds": progress.consumed_seconds,
        "effective_budget_seconds": progress.budget_seconds,
    }
    states[RUNTIME_STATE_KEY] = runtime_telemetry.to_json_value()
    return manifest.model_copy(
        update={
            "status": RunStatus.INTERRUPTED,
            "updated_at": datetime.now(UTC),
            "completed_at": None,
            "scheduler_state": states,
        },
        deep=True,
    )


def _persist_runtime_checkpoint(
    context: _RuntimeContext,
    runtime_telemetry: RuntimeTelemetry,
) -> None:
    current = context.store.get_manifest(context.run_id)
    states = dict(current.scheduler_state)
    states[RUNTIME_STATE_KEY] = runtime_telemetry.to_json_value()
    updated = current.model_copy(
        update={
            "updated_at": datetime.now(UTC),
            "scheduler_state": states,
        },
        deep=True,
    )
    context.store.update_manifest(updated)
    updated.write_json(context.run_directory / MANIFEST_FILENAME)


def _persist_interrupted(
    store: ExperimentStore,
    directory: Path,
    manifest: RunManifest,
    reporter: ProgressReporter,
) -> None:
    current = store.get_manifest(manifest.run_id)
    interrupted = current.model_copy(
        update={
            "status": RunStatus.INTERRUPTED,
            "updated_at": datetime.now(UTC),
            "completed_at": None,
        },
        deep=True,
    )
    store.update_manifest(interrupted)
    interrupted.write_json(directory / MANIFEST_FILENAME)
    reporter.emit(RunStage.INTERRUPTED, "Run interrupted; persisted work is resumable")


def _persist_failed(
    store: ExperimentStore,
    directory: Path,
    manifest: RunManifest,
    reporter: ProgressReporter,
    error: Exception,
) -> None:
    current = store.get_manifest(manifest.run_id)
    failed = current.model_copy(
        update={
            "status": RunStatus.FAILED,
            "updated_at": datetime.now(UTC),
            "completed_at": None,
        },
        deep=True,
    )
    store.update_manifest(failed)
    failed.write_json(directory / MANIFEST_FILENAME)
    reporter.emit(
        RunStage.FAILED,
        f"Run failed with {type(error).__name__}; inspect the command error",
    )


def _monotonic() -> float:
    import time

    return time.monotonic()


__all__ = ["AutoMLRun"]

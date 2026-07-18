"""Typer entry point for the complete Build Week product path."""

from __future__ import annotations

import platform
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from autonomous_automl import AutoMLConfig, AutoMLRun, __version__
from autonomous_automl.api import (
    create_or_validate_trust_artifacts,
    open_run,
    predict_csv,
    validate_run_artifacts,
)
from autonomous_automl.contracts import RunResult, TrialStatus, TrustCertificateStatus
from autonomous_automl.runtime import RunProgressEvent, RunStage
from autonomous_automl.utils.errors import AutoMLError, PlannedInterruption

app = typer.Typer(
    name="automl",
    help="Reproducible, leakage-aware AutoML for tabular CSV data.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
console = Console(highlight=False)


def _version_callback(value: bool) -> None:
    if value:
        console.print(__version__)
        raise typer.Exit


@app.callback()
def root(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = None,
) -> None:
    """Run Autonomous AutoML commands."""


@app.command()
def doctor() -> None:
    """Report the local runtime used by the engine."""

    console.print(f"autonomous-automl {__version__}")
    console.print(f"python {platform.python_version()}")
    console.print(f"sqlite {sqlite3.sqlite_version}")


@app.command("fit")
def fit_command(
    train_csv: Annotated[
        list[Path],
        typer.Argument(help="One or more training CSV files.", exists=True, dir_okay=False),
    ],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="JSON AutoMLConfig file.", exists=True),
    ] = None,
    target: Annotated[
        str | None, typer.Option(help="Target column (required without --config).")
    ] = None,
    task: Annotated[
        str | None,
        typer.Option(help="auto, binary_classification, multiclass_classification, or regression."),
    ] = None,
    metric: Annotated[str | None, typer.Option(help="Metric name or auto.")] = None,
    budget: Annotated[
        str | None,
        typer.Option(
            help=(
                "Search-launch budget (finalization and an already running native "
                "operation may extend total runtime), for example 30s or 2m."
            )
        ),
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="New run directory.")
    ] = None,
    test_csv: Annotated[
        Path | None,
        typer.Option("--test", help="Optional target-free prediction CSV.", exists=True),
    ] = None,
    id_column: Annotated[str | None, typer.Option(help="Optional row identifier column.")] = None,
    n_jobs: Annotated[int | None, typer.Option(min=1, help="Model worker count.")] = None,
    trial_timeout: Annotated[
        int | None,
        typer.Option(min=1, help="Maximum seconds per individual trial."),
    ] = None,
    compute_trust_gap: Annotated[
        bool | None,
        typer.Option(
            "--compute-trust-gap/--no-compute-trust-gap",
            help="Run the isolated post-selection raw-feature diagnostic.",
        ),
    ] = None,
    trust_gap_timeout_seconds: Annotated[
        int | None,
        typer.Option(min=1, help="Maximum seconds for the isolated Trust Gap diagnostic."),
    ] = None,
    interrupt_after_trials: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Create a resumable checkpoint after this many trials.",
        ),
    ] = None,
) -> None:
    """Fit, select, serialize, and report a leakage-aware AutoML run."""

    try:
        run_config = _build_config(
            config_path=config,
            target=target,
            task=task,
            metric=metric,
            budget=budget,
            output=output,
            test_csv=test_csv,
            id_column=id_column,
            n_jobs=n_jobs,
            trial_timeout=trial_timeout,
            compute_trust_gap=compute_trust_gap,
            trust_gap_timeout_seconds=trust_gap_timeout_seconds,
        )
        display = _ProgressDisplay()
        result = AutoMLRun(run_config, progress_callback=display).fit(
            train_csv,
            interrupt_after_trials=interrupt_after_trials,
        )
        _print_result(result)
    except PlannedInterruption as interruption:
        console.print(
            Panel.fit(
                "Checkpoint saved. Resume with:\n"
                f"[bold]uv run automl resume {escape(interruption.run_directory)}[/bold]",
                title="Run intentionally interrupted",
                border_style="yellow",
            )
        )
    except (AutoMLError, ValidationError, ValueError) as error:
        _fail(error)


@app.command("resume")
def resume_command(
    run_directory: Annotated[
        Path,
        typer.Argument(help="Interrupted run directory.", exists=True, file_okay=False),
    ],
    additional_budget: Annotated[
        str | None,
        typer.Option(help="Optional extra budget, for example 15s or 1m."),
    ] = None,
) -> None:
    """Resume a persisted interrupted run without losing completed trials."""

    try:
        extra = None if additional_budget is None else float(_parse_duration(additional_budget))
        result = AutoMLRun.resume(
            run_directory,
            additional_budget_seconds=extra,
            progress_callback=_ProgressDisplay(),
        )
        _print_result(result)
    except (AutoMLError, ValidationError, ValueError) as error:
        _fail(error)


@app.command("inspect")
def inspect_command(
    run_directory: Annotated[
        Path,
        typer.Argument(help="Run directory.", exists=True, file_okay=False),
    ],
) -> None:
    """Show the persisted dataset, validation, leakage, and artifact summary."""

    try:
        directory, store, manifest = open_run(run_directory)
        progress = store.get_run_progress(manifest.run_id)
        trials = store.list_trials(manifest.run_id)
        table = Table(title=f"Run {manifest.run_id}", show_header=False)
        table.add_column("Field", style="cyan")
        table.add_column("Value")
        table.add_row("Status", manifest.status.value)
        table.add_row("Task", manifest.dataset_profile.inferred_task.value)
        table.add_row(
            "Rows / features",
            f"{manifest.dataset_profile.n_rows} / {manifest.dataset_profile.n_features}",
        )
        table.add_row("Validation", manifest.validation_plan.splitter_name)
        table.add_row("Trials", str(len(trials)))
        table.add_row(
            "Budget",
            f"{progress.consumed_seconds:.2f}s / {progress.budget_seconds:.2f}s",
        )
        table.add_row("Leakage alerts", str(len(manifest.leakage_report.findings)))
        table.add_row(
            "Excluded columns",
            ", ".join(manifest.leakage_report.excluded_columns) or "none",
        )
        table.add_row("Artifacts", str(len(store.list_artifacts(manifest.run_id))))
        table.add_row("Directory", str(directory))
        console.print(table)
    except (AutoMLError, ValidationError, ValueError) as error:
        _fail(error)


@app.command("leaderboard")
def leaderboard_command(
    run_directory: Annotated[
        Path,
        typer.Argument(help="Completed run directory.", exists=True, file_okay=False),
    ],
    limit: Annotated[int, typer.Option(min=1, max=100, help="Maximum rows to show.")] = 10,
) -> None:
    """Display the persisted leaderboard."""

    try:
        directory, _, manifest = open_run(run_directory)
        relative = manifest.artifacts.get("leaderboard", "leaderboard.csv")
        frame = pd.read_csv(directory / relative).head(limit)
        _print_leaderboard(frame, title=f"Leaderboard · {manifest.run_id}")
    except (AutoMLError, ValidationError, ValueError, OSError) as error:
        _fail(error)


@app.command("predict")
def predict_command(
    run_directory: Annotated[
        Path,
        typer.Argument(help="Completed run directory.", exists=True, file_okay=False),
    ],
    csv_path: Annotated[
        Path,
        typer.Argument(help="Target-free CSV to score.", exists=True, dir_okay=False),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Destination CSV (must not exist)."),
    ] = None,
) -> None:
    """Generate predictions from the checksum-verified serialized pipeline."""

    try:
        destination = output or (run_directory / "external_predictions.csv")
        result = predict_csv(run_directory, csv_path, output_path=destination)
        console.print(f"[green]Predictions written:[/green] {escape(str(result))}")
    except (AutoMLError, ValidationError, ValueError) as error:
        _fail(error)


@app.command("validate-artifacts")
def validate_artifacts_command(
    run_directory: Annotated[
        Path,
        typer.Argument(help="Run directory.", exists=True, file_okay=False),
    ],
) -> None:
    """Verify SQLite, hashes, model loading, and recorded predictions."""

    try:
        summary = validate_run_artifacts(run_directory)
        table = Table(title=f"Artifact validation · {summary.run_id}", show_header=False)
        table.add_column("Check", style="cyan")
        table.add_column("Result")
        table.add_row("SQLite integrity", "PASS")
        table.add_row("Manifest agreement", "PASS")
        table.add_row("Source hashes", "PASS")
        table.add_row("Registered artifacts", str(summary.artifact_count))
        table.add_row("Pipeline load", "PASS" if summary.pipeline_loadable else "n/a")
        table.add_row(
            "Prediction replay",
            "PASS" if summary.predictions_match else "n/a",
        )
        console.print(table)
    except (AutoMLError, ValidationError, ValueError) as error:
        _fail(error)


@app.command("trust")
def trust_command(
    run_directory: Annotated[
        Path,
        typer.Argument(help="Persisted run directory.", exists=True, file_okay=False),
    ],
) -> None:
    """Recalculate and display the self-verified trust certificate."""

    try:
        result = create_or_validate_trust_artifacts(run_directory, persist_if_missing=True)
        certificate = result.certificate
        table = Table(title=f"Self-verified trust certificate · {certificate.run_id}")
        table.add_column("Field", style="cyan")
        table.add_column("Recorded value", overflow="fold")
        table.add_row("Status", certificate.status.value)
        table.add_row(
            "Verified score",
            "Unavailable"
            if certificate.verified_score is None
            else f"{certificate.verified_score:.6f}",
        )
        table.add_row("Leakage findings", str(len(certificate.leakage_findings)))
        table.add_row(
            "Unresolved critical findings",
            str(len(certificate.unresolved_critical_findings)),
        )
        table.add_row("Artifact validation", certificate.artifact_validation.status.value)
        table.add_row("Prediction replay", certificate.prediction_replay.status.value)
        gap = certificate.observed_trust_gap
        table.add_row(
            "Observed Trust Gap",
            "Unavailable"
            if gap is None
            else (
                gap.status.value
                if gap.observed_trust_gap is None
                else f"{gap.observed_trust_gap:+.6f}"
            ),
        )
        if certificate.runtime_telemetry is not None:
            runtime = certificate.runtime_telemetry
            table.add_row(
                "Search budget",
                f"{runtime.configured_search_budget_seconds:.3f} s",
            )
            table.add_row(
                "Search elapsed",
                _optional_seconds(runtime.search_elapsed_seconds),
            )
            table.add_row(
                "Finalization",
                _optional_seconds(runtime.finalization_elapsed_seconds),
            )
            table.add_row("Total runtime", _optional_seconds(runtime.total_runtime_seconds))
            table.add_row("Stop reason", runtime.stop_reason.value.replace("_", " "))
        table.add_row("JSON", str(result.json_path))
        table.add_row("HTML", str(result.html_path))
        console.print(table)
        if certificate.warnings:
            console.print("[yellow]Warnings:[/yellow]")
            for warning in certificate.warnings:
                console.print(f"  - {escape(warning)}")
        if certificate.status is TrustCertificateStatus.FAILED:
            raise typer.Exit(code=1)
    except (AutoMLError, ValidationError, ValueError, OSError) as error:
        _fail(error)


@app.command("demo")
def demo_command(
    output_root: Annotated[
        Path | None,
        typer.Option("--output-root", help="New parent directory for the three demo runs."),
    ] = None,
    budget: Annotated[str, typer.Option(help="Budget per scenario, for example 6s.")] = "6s",
    compute_trust_gap: Annotated[
        bool,
        typer.Option(
            "--compute-trust-gap/--no-compute-trust-gap",
            help="Enable the isolated diagnostic for the leakage scenario.",
        ),
    ] = True,
    trust_gap_timeout_seconds: Annotated[
        int,
        typer.Option(min=1, help="Maximum seconds for the leakage Trust Gap diagnostic."),
    ] = 10,
) -> None:
    """Run classification+resume, regression, and leakage demos end to end."""

    try:
        seconds = _parse_duration(budget)
        if seconds < 6:
            raise ValueError("the three-scenario demo requires at least 6s per scenario")
        root = output_root or Path("runs") / (
            "build-week-demo-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        )
        _run_demo(
            root.resolve(),
            seconds,
            compute_trust_gap=compute_trust_gap,
            trust_gap_timeout_seconds=trust_gap_timeout_seconds,
        )
    except (AutoMLError, ValidationError, ValueError, OSError) as error:
        _fail(error)


class _ProgressDisplay:
    def __init__(self) -> None:
        self.last_stage = ""
        self.last_failed = -1

    def __call__(self, event: RunProgressEvent) -> None:
        stage = event.stage.value.replace("_", " ")
        total = event.completed_trials + event.failed_trials + event.interrupted_trials
        stage_changed = stage != self.last_stage
        failed_changed = event.failed_trials > self.last_failed
        should_print = (
            event.stage is not RunStage.SEARCH
            or stage_changed
            or failed_changed
            or total <= 1
            or total % 3 == 0
        )
        self.last_stage = stage
        self.last_failed = event.failed_trials
        if not should_print:
            return
        best = "n/a" if event.best_internal_score is None else f"{event.best_internal_score:.5f}"
        counts = (
            f"ok {event.completed_trials} · failed {event.failed_trials} · "
            f"stopped {event.interrupted_trials}"
        )
        message = f" · {escape(event.message)}" if stage_changed or total == 0 else ""
        console.print(
            f"[bold blue]→[/bold blue] [bold]{escape(stage.upper())}[/bold]{message} | "
            f"{counts} | best(search) {best} | {event.elapsed_seconds:.1f}s used · "
            f"{event.remaining_seconds:.1f}s left"
        )


def _build_config(
    *,
    config_path: Path | None,
    target: str | None,
    task: str | None,
    metric: str | None,
    budget: str | None,
    output: Path | None,
    test_csv: Path | None,
    id_column: str | None,
    n_jobs: int | None,
    trial_timeout: int | None,
    compute_trust_gap: bool | None,
    trust_gap_timeout_seconds: int | None,
) -> AutoMLConfig:
    if config_path is None:
        if target is None:
            raise ValueError("--target is required when --config is not supplied")
        base: dict[str, Any] = {
            "target": target,
            "task": task or "auto",
            "metric": metric or "auto",
            "budget_seconds": _parse_duration(budget or "30s"),
            "output_dir": output or _default_output_directory(),
        }
    else:
        base = AutoMLConfig.read_json(config_path).model_dump()
    overrides: dict[str, Any] = {}
    for key, value in (
        ("target", target),
        ("task", task),
        ("metric", metric),
        ("output_dir", output),
        ("test_path", test_csv),
        ("id_column", id_column),
        ("n_jobs", n_jobs),
        ("trial_timeout_seconds", trial_timeout),
        ("compute_trust_gap", compute_trust_gap),
        ("trust_gap_timeout_seconds", trust_gap_timeout_seconds),
    ):
        if value is not None:
            overrides[key] = value
    if budget is not None:
        overrides["budget_seconds"] = _parse_duration(budget)
    return AutoMLConfig.model_validate({**base, **overrides})


def _parse_duration(value: str) -> int:
    normalized = value.strip().lower()
    multiplier = 1
    if normalized.endswith("ms"):
        raise ValueError("budgets shorter than one second are not supported")
    if normalized.endswith("s"):
        normalized = normalized[:-1]
    elif normalized.endswith("m"):
        multiplier = 60
        normalized = normalized[:-1]
    elif normalized.endswith("h"):
        multiplier = 3600
        normalized = normalized[:-1]
    try:
        seconds = float(normalized) * multiplier
    except ValueError as error:
        raise ValueError(f"invalid duration: {value!r}") from error
    if not seconds.is_integer() or seconds < 1:
        raise ValueError("duration must resolve to a positive whole number of seconds")
    return int(seconds)


def _default_output_directory() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / f"automl-{stamp}"


def _print_result(result: RunResult) -> None:
    _, store, manifest = open_run(result.output_dir)
    trials = store.list_trials(manifest.run_id)
    progress = store.get_run_progress(manifest.run_id)
    trust = create_or_validate_trust_artifacts(result.output_dir, persist_if_missing=True)
    console.print(
        Panel.fit(
            f"[bold green]Completed[/bold green] · {escape(result.run_id)}\n"
            f"{result.primary_metric.value} = [bold]{result.best_score:.6f}[/bold]",
            title="AutoML result",
            border_style="green",
        )
    )
    diagnostics = Table(title="Run diagnostics", show_header=False)
    diagnostics.add_column("Field", style="cyan")
    diagnostics.add_column("Value")
    diagnostics.add_row(
        "Dataset",
        escape(", ".join(path.name for path in manifest.dataset.train_paths)),
    )
    diagnostics.add_row("Task", manifest.dataset_profile.inferred_task.value)
    diagnostics.add_row("Metric", result.primary_metric.value)
    diagnostics.add_row(
        "Validation",
        f"{manifest.validation_plan.splitter_name} · {manifest.validation_plan.n_splits} folds",
    )
    diagnostics.add_row("Leakage alerts", str(len(manifest.leakage_report.findings)))
    diagnostics.add_row(
        "Excluded before search",
        escape(", ".join(manifest.leakage_report.excluded_columns) or "none"),
    )
    diagnostics.add_row("Trials", _trial_counts(trials))
    runtime = trust.certificate.runtime_telemetry
    if runtime is None:
        diagnostics.add_row(
            "Budget",
            f"{progress.consumed_seconds:.2f}s used / {progress.budget_seconds:.2f}s recorded",
        )
    else:
        diagnostics.add_row(
            "Search budget",
            f"{runtime.configured_search_budget_seconds:.2f} s",
        )
        diagnostics.add_row("Search elapsed", _optional_seconds(runtime.search_elapsed_seconds))
        diagnostics.add_row(
            "Finalization",
            _optional_seconds(runtime.finalization_elapsed_seconds),
        )
        diagnostics.add_row("Total runtime", _optional_seconds(runtime.total_runtime_seconds))
        diagnostics.add_row("Stop reason", runtime.stop_reason.value.replace("_", " "))
    diagnostics.add_row("Trust status", trust.certificate.status.value)
    console.print(diagnostics)
    if trust.certificate.selection_explanation is not None:
        explanation = trust.certificate.selection_explanation
        console.print(
            Panel.fit(
                f"[bold]Selection reason:[/bold] {escape(explanation.selection_reason)}\n"
                f"{escape(explanation.selection_rule)}"
                + (
                    "\n[bold]Tie-breaker:[/bold] " + escape(explanation.tie_breaker_detail)
                    if explanation.tie_breaker_detail is not None
                    else ""
                ),
                title="Why this pipeline won",
                border_style="blue",
            )
        )
    frame = pd.read_csv(result.leaderboard_path).head(10)
    _print_leaderboard(frame, title="Final leaderboard")
    paths = Table(title="Artifacts", show_header=False)
    paths.add_column("Artifact", style="cyan")
    paths.add_column("Path", overflow="fold")
    paths.add_row("Run directory", str(result.output_dir))
    paths.add_row("Model", str(result.best_pipeline_path))
    paths.add_row("Manifest", str(result.manifest_path))
    paths.add_row("Leaderboard", str(result.leaderboard_path))
    paths.add_row("Predictions", str(result.predictions_path or "n/a"))
    paths.add_row("HTML report", str(result.report_path))
    paths.add_row("Trust certificate JSON", str(trust.json_path))
    paths.add_row("Trust certificate HTML", str(trust.html_path))
    console.print(paths)
    console.print(f"Open report: [bold]{escape(_report_open_command(result.report_path))}[/bold]")


def _print_leaderboard(frame: pd.DataFrame, *, title: str) -> None:
    table = Table(title=title)
    for heading in ("Rank", "Family", "Model", "Fidelity", "Metric", "Score", "Seconds"):
        table.add_column(heading)
    for row in frame.to_dict(orient="records"):
        table.add_row(
            str(row.get("rank", "")),
            str(row.get("family", "")),
            str(row.get("model_name", "")),
            f"L{row.get('fidelity_level', '')}",
            str(row.get("primary_metric", "")),
            _format_number(row.get("metric_value")),
            _format_number(row.get("total_seconds"), digits=3),
        )
    console.print(table)


def _format_number(value: object, *, digits: int = 6) -> str:
    if value is None or bool(pd.isna(value)):
        return "n/a"
    return f"{float(str(value)):.{digits}f}"


def _optional_seconds(value: float | None) -> str:
    return "Unavailable" if value is None else f"{value:.3f} s"


def _trial_counts(trials: list[Any]) -> str:
    completed = sum(trial.status is TrialStatus.COMPLETED for trial in trials)
    failed = sum(trial.status is TrialStatus.FAILED for trial in trials)
    interrupted = sum(trial.status is TrialStatus.INTERRUPTED for trial in trials)
    return f"{completed} successful · {failed} failed · {interrupted} interrupted"


def _report_open_command(report_path: Path) -> str:
    return f"uv run python -m webbrowser {report_path.resolve().as_uri()}"


def _run_demo(
    root: Path,
    budget_seconds: int,
    *,
    compute_trust_gap: bool,
    trust_gap_timeout_seconds: int,
) -> None:
    examples = Path.cwd() / "examples"
    required = (
        "classification_config.json",
        "classification_train.csv",
        "regression_config.json",
        "regression_train.csv",
        "leakage_config.json",
        "leakage_train.csv",
        "leakage_test.csv",
    )
    missing = [name for name in required if not (examples / name).is_file()]
    if missing:
        raise ValueError(
            "demo files are missing; run `uv run python scripts/generate_examples.py`: "
            + ", ".join(missing)
        )
    root.mkdir(parents=True, exist_ok=False)
    console.print(Panel.fit(f"Demo output: {escape(str(root))}", title="Build Week demo"))
    results: list[RunResult] = []

    classification_config = _demo_config(
        examples / "classification_config.json",
        root / "classification-resumed",
        budget_seconds,
        compute_trust_gap=False,
        trust_gap_timeout_seconds=trust_gap_timeout_seconds,
    )
    console.rule("1/3 Binary classification + resume · classification_train.csv")
    try:
        AutoMLRun(classification_config, progress_callback=_ProgressDisplay()).fit(
            examples / "classification_train.csv",
            interrupt_after_trials=2,
        )
    except PlannedInterruption:
        console.print("[yellow]Checkpoint persisted; resuming without repeating trials.[/yellow]")
    results.append(
        AutoMLRun.resume(
            classification_config.output_dir,
            additional_budget_seconds=max(2, budget_seconds // 2),
            progress_callback=_ProgressDisplay(),
        )
    )

    console.rule("2/3 Regression · regression_train.csv")
    regression_config = _demo_config(
        examples / "regression_config.json",
        root / "regression",
        budget_seconds,
        compute_trust_gap=False,
        trust_gap_timeout_seconds=trust_gap_timeout_seconds,
    )
    results.append(
        AutoMLRun(regression_config, progress_callback=_ProgressDisplay()).fit(
            examples / "regression_train.csv"
        )
    )

    console.rule("3/3 Leakage detection · leakage_train.csv")
    leakage_config = _demo_config(
        examples / "leakage_config.json",
        root / "leakage",
        budget_seconds,
        compute_trust_gap=compute_trust_gap,
        trust_gap_timeout_seconds=trust_gap_timeout_seconds,
    )
    results.append(
        AutoMLRun(leakage_config, progress_callback=_ProgressDisplay()).fit(
            examples / "leakage_train.csv"
        )
    )

    summary = Table(title="Build Week demo completed")
    summary.add_column("Scenario")
    summary.add_column("Metric")
    summary.add_column("Best score")
    summary.add_column("Trials ok/fail/stop")
    summary.add_column("Replay")
    manifests = []
    trust_results = []
    for name, result in zip(("classification", "regression", "leakage"), results, strict=True):
        validation = validate_run_artifacts(result.output_dir)
        _, store, manifest = open_run(result.output_dir)
        trust = create_or_validate_trust_artifacts(
            result.output_dir,
            persist_if_missing=True,
        )
        trust_results.append(trust)
        manifests.append(manifest)
        trials = store.list_trials(manifest.run_id)
        summary.add_row(
            name,
            result.primary_metric.value,
            f"{result.best_score:.6f}",
            "/".join(
                str(sum(trial.status is status for trial in trials))
                for status in (
                    TrialStatus.COMPLETED,
                    TrialStatus.FAILED,
                    TrialStatus.INTERRUPTED,
                )
            ),
            "PASS" if validation.predictions_match is True else "n/a",
        )
    leakage_manifest = manifests[-1]
    expected = {"target_copy", "customer_id"}
    if not expected.issubset(leakage_manifest.leakage_report.excluded_columns):
        raise ValueError("demo leakage columns were not neutralized")
    console.print(summary)
    certificate_summary = Table(title="Self-verified trust certificates")
    certificate_summary.add_column("Scenario")
    certificate_summary.add_column("Status")
    for name, trust in zip(
        ("classification", "regression", "leakage"),
        trust_results,
        strict=True,
    ):
        certificate_summary.add_row(name, trust.certificate.status.value)
    console.print(certificate_summary)

    classification_frame = pd.read_csv(results[0].leaderboard_path).head(5)
    _print_leaderboard(classification_frame, title="Classification leaderboard · top 5")

    leakage_table = Table(title="Leakage proof · neutralized before search")
    leakage_table.add_column("Risk")
    leakage_table.add_column("Column")
    leakage_table.add_column("Evidence", overflow="fold")
    leakage_table.add_column("Action", overflow="fold")
    for finding in leakage_manifest.leakage_report.findings:
        evidence = ", ".join(
            f"{key}={value}" for key, value in sorted(finding.evidence_summary.items())
        )
        leakage_table.add_row(
            finding.finding_type.value,
            escape(finding.column or "dataset"),
            escape(evidence or "recorded heuristic"),
            escape(finding.action),
        )
    console.print(leakage_table)
    leakage_trust = trust_results[-1].certificate
    gap = leakage_trust.observed_trust_gap
    console.print(
        "[bold]Score interpretation:[/bold] every main leaderboard score is "
        "post-neutralization. The raw comparison, when available, is a separate "
        "post-selection diagnostic."
    )

    trust_summary = Table(title="TRUST SUMMARY", show_header=False)
    trust_summary.add_column("Field", style="cyan")
    trust_summary.add_column("Recorded value", overflow="fold")
    trust_summary.add_row("Status", leakage_trust.status.value)
    trust_summary.add_row(
        "Verified score",
        "Unavailable"
        if leakage_trust.verified_score is None
        else f"{leakage_trust.verified_score:.6f}",
    )
    trust_summary.add_row(
        "Raw diagnostic score",
        "Unavailable" if gap is None or gap.raw_score is None else f"{gap.raw_score:.6f}",
    )
    trust_summary.add_row(
        "Observed Trust Gap",
        "NOT_COMPUTED"
        if gap is None or gap.observed_trust_gap is None
        else f"{gap.observed_trust_gap:+.6f}",
    )
    trust_summary.add_row("Leakage findings", str(len(leakage_trust.leakage_findings)))
    trust_summary.add_row(
        "Excluded features",
        escape(", ".join(leakage_trust.excluded_columns) or "none"),
    )
    trust_summary.add_row("Artifact replay", leakage_trust.prediction_replay.status.value)
    if leakage_trust.runtime_telemetry is not None:
        runtime = leakage_trust.runtime_telemetry
        trust_summary.add_row(
            "Search budget",
            f"{runtime.configured_search_budget_seconds:.1f} s",
        )
        trust_summary.add_row("Search elapsed", _optional_seconds(runtime.search_elapsed_seconds))
        trust_summary.add_row("Total runtime", _optional_seconds(runtime.total_runtime_seconds))
    trust_summary.add_row("Certificate", str(trust_results[-1].html_path))
    trust_summary.add_row("Full report", str(results[-1].report_path))
    console.print(trust_summary)

    artifacts = Table(title="Run artifacts and reports")
    artifacts.add_column("Scenario")
    artifacts.add_column("Run directory", overflow="fold")
    artifacts.add_column("HTML report", overflow="fold")
    for name, result in zip(("classification", "regression", "leakage"), results, strict=True):
        artifacts.add_row(name, escape(str(result.output_dir)), escape(str(result.report_path)))
    console.print(artifacts)
    console.print(
        f"Open leakage report: [bold]{escape(_report_open_command(results[-1].report_path))}[/bold]"
    )
    console.print(f"[bold green]All scenarios passed.[/bold green] Artifacts: {escape(str(root))}")


def _demo_config(
    path: Path,
    output: Path,
    budget_seconds: int,
    *,
    compute_trust_gap: bool,
    trust_gap_timeout_seconds: int,
) -> AutoMLConfig:
    values = AutoMLConfig.read_json(path).model_dump()
    values.update(
        {
            "output_dir": output,
            "budget_seconds": budget_seconds,
            "compute_trust_gap": compute_trust_gap,
            "trust_gap_timeout_seconds": trust_gap_timeout_seconds,
        }
    )
    return AutoMLConfig.model_validate(values)


def _fail(error: Exception) -> None:
    console.print(
        Panel.fit(
            f"[bold red]{escape(type(error).__name__)}[/bold red]\n{escape(str(error))}",
            title="Command failed",
            border_style="red",
        )
    )
    raise typer.Exit(code=1) from error


def main() -> None:
    """Invoke the Typer application."""

    app()

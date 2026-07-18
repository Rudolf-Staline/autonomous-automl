"""Standalone HTML report rendered only from persisted run state."""

# The embedded HTML template is intentionally kept readable as HTML.
# ruff: noqa: E501

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, select_autoescape

from autonomous_automl.contracts import (
    ContractModel,
    MetricName,
    ObservedTrustGap,
    PipelineSelectionExplanation,
    RunManifest,
    RuntimeTelemetry,
    TrialResult,
    TrialStatus,
    TrustCertificate,
)
from autonomous_automl.evaluation.selection import build_leaderboard
from autonomous_automl.pipelines import pipeline_fingerprint
from autonomous_automl.tracking import (
    ArtifactRecord,
    ArtifactRegistration,
    ArtifactStore,
    ExperimentStore,
)


@dataclass(frozen=True, slots=True)
class ReportTrial:
    rank: int
    trial_id: str
    family: str
    model_name: str
    fidelity_level: int
    status: str
    score: str
    uncertainty: str
    duration: str


@dataclass(frozen=True, slots=True)
class ReportData:
    manifest: RunManifest
    metric: MetricName
    score_label: str
    best_score: str
    consumed_seconds: float
    remaining_seconds: float
    successful_count: int
    failed_count: int
    interrupted_count: int
    families: tuple[str, ...]
    trials: tuple[ReportTrial, ...]
    failed_trials: tuple[TrialResult, ...]
    artifacts: tuple[tuple[str, str], ...]
    best_spec_json: str
    dataset_sources: tuple[str, ...]
    test_source: str | None
    best_family: str
    best_model: str
    trust_certificate: TrustCertificate | None
    runtime_telemetry: RuntimeTelemetry | None
    selection_explanation: PipelineSelectionExplanation | None
    observed_trust_gap: ObservedTrustGap | None


def render_run_report(store: ExperimentStore, run_id: str) -> str:
    """Read the SQLite authority and render a self-contained HTML document."""

    manifest = store.get_manifest(run_id)
    trials = store.list_trials(run_id)
    progress = store.get_run_progress(run_id)
    registrations = store.list_artifacts(run_id)
    data = _build_report_data(
        manifest,
        trials,
        progress.consumed_seconds,
        registrations,
        trust_certificate=_registered_contract(
            store,
            registrations,
            "trust_certificate_json",
            TrustCertificate,
        ),
        runtime_telemetry=_registered_contract(
            store,
            registrations,
            "runtime_telemetry",
            RuntimeTelemetry,
        ),
        selection_explanation=_registered_contract(
            store,
            registrations,
            "selection_explanation",
            PipelineSelectionExplanation,
        ),
        observed_trust_gap=_registered_contract(
            store,
            registrations,
            "trust_gap",
            ObservedTrustGap,
        ),
    )
    environment = Environment(
        autoescape=select_autoescape(default_for_string=True),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return environment.from_string(_REPORT_TEMPLATE).render(
        data=data,
        seconds=_format_seconds,
        timestamp=_format_timestamp,
        boolean=_format_boolean,
    )


def _build_report_data(
    manifest: RunManifest,
    trials: list[TrialResult],
    consumed_seconds: float,
    registrations: list[ArtifactRegistration],
    *,
    trust_certificate: TrustCertificate | None = None,
    runtime_telemetry: RuntimeTelemetry | None = None,
    selection_explanation: PipelineSelectionExplanation | None = None,
    observed_trust_gap: ObservedTrustGap | None = None,
) -> ReportData:
    leaderboard = build_leaderboard(trials)
    metric = leaderboard.primary_metric or _manifest_metric(manifest)
    by_id = {result.trial_id: result for result in trials}
    report_trials = tuple(
        ReportTrial(
            rank=entry.rank,
            trial_id=entry.trial_id,
            family=entry.family,
            model_name=entry.model_name,
            fidelity_level=entry.fidelity.level,
            status=TrialStatus.COMPLETED.value,
            score=_format_score(_user_score(by_id[entry.trial_id], metric)),
            uncertainty=_format_score(entry.std_score),
            duration=f"{entry.total_seconds:.3f}s",
        )
        for entry in leaderboard.entries
    )
    failed = tuple(result for result in trials if result.status is TrialStatus.FAILED)
    interrupted = sum(result.status is TrialStatus.INTERRUPTED for result in trials)
    best_trial = _best_trial(manifest, trials)
    best_score = (
        _format_score(_user_score(best_trial, metric))
        if best_trial is not None
        else _format_score(manifest.metrics.get("best_score"))
    )
    registered_paths = {
        registration.name: registration.relative_path for registration in registrations
    }
    artifact_paths = {**manifest.artifacts, **registered_paths}
    artifacts = tuple(sorted(artifact_paths.items()))
    return ReportData(
        manifest=manifest,
        metric=metric,
        score_label=_metric_label(metric),
        best_score=best_score,
        consumed_seconds=consumed_seconds,
        remaining_seconds=max(
            0.0,
            progress_budget(manifest) - consumed_seconds,
        ),
        successful_count=sum(result.status is TrialStatus.COMPLETED for result in trials),
        failed_count=len(failed),
        interrupted_count=interrupted,
        families=tuple(sorted({result.family for result in trials})),
        trials=report_trials,
        failed_trials=failed,
        artifacts=artifacts,
        best_spec_json=(
            "{}"
            if manifest.best_pipeline is None
            else json.dumps(
                manifest.best_pipeline.to_json_value(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        ),
        dataset_sources=tuple(Path(path).name for path in manifest.dataset.train_paths),
        test_source=(
            None if manifest.dataset.test_path is None else Path(manifest.dataset.test_path).name
        ),
        best_family=("n/a" if manifest.best_pipeline is None else manifest.best_pipeline.family),
        best_model=("n/a" if manifest.best_pipeline is None else manifest.best_pipeline.model_name),
        trust_certificate=trust_certificate,
        runtime_telemetry=runtime_telemetry,
        selection_explanation=selection_explanation,
        observed_trust_gap=observed_trust_gap,
    )


def _registered_contract[ContractT: ContractModel](
    store: ExperimentStore,
    registrations: list[ArtifactRegistration],
    name: str,
    contract_type: type[ContractT],
) -> ContractT | None:
    registration = next(
        (artifact for artifact in registrations if artifact.name == name),
        None,
    )
    if registration is None:
        return None
    record = ArtifactRecord(
        name=registration.name,
        kind=registration.kind,
        relative_path=registration.relative_path,
        sha256=registration.sha256,
        size_bytes=registration.size_bytes,
        media_type=registration.media_type,
    )
    path = ArtifactStore(store.path.parent).validate(record)
    return contract_type.read_json(path)


def progress_budget(manifest: RunManifest) -> float:
    state = manifest.scheduler_state.get("tracking")
    if isinstance(state, dict):
        value = state.get("effective_budget_seconds")
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return float(manifest.configuration.budget_seconds)


def _format_seconds(value: float | None) -> str:
    return "Unavailable" if value is None else f"{value:.3f}s"


def _format_timestamp(value: datetime | None) -> str:
    return "Unavailable" if value is None else value.isoformat()


def _format_boolean(value: bool | None) -> str:
    if value is None:
        return "Unavailable"
    return "Yes" if value else "No"


def _manifest_metric(manifest: RunManifest) -> MetricName:
    if manifest.configuration.metric is not MetricName.AUTO:
        return manifest.configuration.metric
    value = manifest.metrics.get("metric_name")
    if value is not None:
        raise ValueError("manifest metric_name must not be encoded as a float")
    raise ValueError("persisted run contains no resolved primary metric")


def _best_trial(manifest: RunManifest, trials: list[TrialResult]) -> TrialResult | None:
    if manifest.best_pipeline is None:
        return None
    expected = pipeline_fingerprint(manifest.best_pipeline)
    matches = [
        result
        for result in trials
        if result.status is TrialStatus.COMPLETED
        and pipeline_fingerprint(result.pipeline_spec) == expected
    ]
    if not matches:
        return None
    return min(
        matches,
        key=lambda result: (
            -result.fidelity.level,
            -(result.mean_score if result.mean_score is not None else float("-inf")),
            result.trial_id,
        ),
    )


def _user_score(result: TrialResult, metric: MetricName) -> float | None:
    if result.fold_results:
        return sum(item.user_metric_value for item in result.fold_results) / len(
            result.fold_results
        )
    if result.mean_score is None:
        return None
    return (
        -result.mean_score
        if metric in {MetricName.RMSE, MetricName.MAE, MetricName.LOG_LOSS}
        else result.mean_score
    )


def _metric_label(metric: MetricName) -> str:
    return metric.value.replace("_", " ").upper()


def _format_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


_REPORT_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Autonomous AutoML — {{ data.manifest.run_id }}</title>
  <style>
    :root { color-scheme: light; --ink:#152033; --muted:#60708a; --line:#dbe3ef;
      --panel:#f7f9fc; --accent:#3157d5; --good:#087a55; --warn:#a45508; }
    * { box-sizing:border-box; } body { margin:0; font:15px/1.5 Inter,ui-sans-serif,
      system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink);
      background:#eef2f8; } main { max-width:1120px; margin:0 auto; padding:32px 20px 56px; }
    header,.card { background:white; border:1px solid var(--line); border-radius:14px;
      box-shadow:0 8px 28px #23345a12; } header { padding:28px; margin-bottom:18px; }
    h1,h2,h3 { margin-top:0; } h1 { margin-bottom:5px; font-size:30px; }
    h2 { font-size:19px; } p { color:var(--muted); } .grid { display:grid;
      grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin-top:20px; }
    .metric { padding:14px; background:var(--panel); border-radius:10px; }
    .metric small { color:var(--muted); display:block; } .metric strong { font-size:19px; }
    .card { padding:22px; margin:18px 0; overflow:auto; } table { width:100%;
      border-collapse:collapse; min-width:720px; } th,td { padding:9px 10px;
      border-bottom:1px solid var(--line); text-align:left; } th { color:var(--muted);
      font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
    code,pre { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
    pre { padding:15px; overflow:auto; background:#101827; color:#e8eefc; border-radius:9px; }
    .alert { border-left:4px solid var(--warn); padding:10px 13px; background:#fff8ed;
      margin:8px 0; } .ok { color:var(--good); } a { color:var(--accent); }
    .certificate { border-left:5px solid var(--accent); } .status { font-weight:750; }
    ul { padding-left:20px; } footer { color:var(--muted); margin-top:25px; }
  </style>
</head>
<body><main>
  <header>
    <h1>Autonomous AutoML run</h1>
    <p>Persisted run <code>{{ data.manifest.run_id }}</code>. Trial, score and artifact
      information below is rendered from the manifest and SQLite registry.</p>
    <div class="grid">
      <div class="metric"><small>Task</small><strong>{{ data.manifest.dataset_profile.inferred_task.value }}</strong></div>
      <div class="metric"><small>{{ data.score_label }}</small><strong>{{ data.best_score }}</strong></div>
      <div class="metric"><small>Rows x features</small><strong>{{ data.manifest.dataset_profile.n_rows }} x {{ data.manifest.dataset_profile.n_features }}</strong></div>
      <div class="metric"><small>Budget consumed</small><strong>{{ "%.2f"|format(data.consumed_seconds) }}s</strong></div>
      <div class="metric"><small>Selected model</small><strong>{{ data.best_model }}</strong></div>
      {% if data.trust_certificate %}<div class="metric"><small>Trust status</small><strong>{{ data.trust_certificate.status.value }}</strong></div>{% endif %}
    </div>
  </header>

  {% if data.trust_certificate %}<section class="card certificate"><h2>Self-verified trust certificate</h2>
    <p class="status">{{ data.trust_certificate.status.value }}</p>
    <p>{{ data.trust_certificate.disclaimer }}</p>
    <p><a href="trust_certificate.html">Open the standalone trust certificate</a> ·
      <a href="trust_certificate.json">JSON evidence</a></p>
  </section>{% endif %}

  <section class="card"><h2>Dataset and validation</h2>
    <ul>
      <li>Training CSV: <strong>{{ data.dataset_sources|join(", ") }}</strong></li>
      {% if data.test_source %}<li>Prediction CSV: <strong>{{ data.test_source }}</strong></li>{% endif %}
      <li>Resolved metric: <strong>{{ data.metric.value }}</strong></li>
      <li>Validation: <strong>{{ data.manifest.validation_plan.splitter_name }}</strong>, {{ data.manifest.validation_plan.n_splits }} persisted folds</li>
      <li>Estimated memory: {{ "%.3f"|format(data.manifest.dataset_profile.estimated_memory_mb) }} MiB</li>
      <li>Recorded budget: {{ "%.2f"|format(data.consumed_seconds + data.remaining_seconds) }}s</li>
      <li>Remaining recorded budget: {{ "%.2f"|format(data.remaining_seconds) }}s</li>
    </ul>
  </section>

  <section class="card"><h2>Leakage and schema diagnostics</h2>
    {% if data.manifest.leakage_report.findings %}
      {% for finding in data.manifest.leakage_report.findings %}
      <div class="alert"><strong>{{ finding.severity.value }} · {{ finding.finding_type.value }}</strong>
        {% if finding.column %} — column <code>{{ finding.column }}</code>{% endif %}<br>
        Action: {{ finding.action }} · confidence {{ "%.2f"|format(finding.confidence) }}</div>
      {% endfor %}
    {% else %}<p class="ok">No heuristic leakage alert was recorded.</p>{% endif %}
    <p>Excluded before search: {% if data.manifest.leakage_report.excluded_columns %}
      <code>{{ data.manifest.leakage_report.excluded_columns|join(", ") }}</code>{% else %}none{% endif %}</p>
    {% if data.manifest.leakage_report.excluded_columns %}
      <p><strong>Neutralization proof:</strong> the columns above were removed before
        pipeline search and validation. Every leaderboard score is therefore post-neutralization.
        {% if data.observed_trust_gap and data.observed_trust_gap.status.value == "COMPUTED" %}
        A separate post-selection raw diagnostic was calculated and never entered the leaderboard.
        {% else %}No contaminated comparison score was calculated. No raw diagnostic
        score entered search or selection.{% endif %}</p>
    {% endif %}
  </section>

  {% if data.observed_trust_gap %}<section class="card"><h2>Observed Trust Gap</h2>
    <p><strong>{{ data.observed_trust_gap.status.value }}</strong> — {{ data.observed_trust_gap.reason }}</p>
    {% if data.observed_trust_gap.status.value == "COMPUTED" %}
    <div class="grid">
      <div class="metric"><small>Raw diagnostic score</small><strong>{{ "%.6f"|format(data.observed_trust_gap.raw_score) }}</strong></div>
      <div class="metric"><small>Verified score</small><strong>{{ "%.6f"|format(data.observed_trust_gap.verified_score) }}</strong></div>
      <div class="metric"><small>Apparent score inflation under the raw protocol</small><strong>{{ "%+.6f"|format(data.observed_trust_gap.observed_trust_gap) }}</strong></div>
      <div class="metric"><small>Diagnostic duration</small><strong>{{ "%.3f"|format(data.observed_trust_gap.diagnostic_elapsed_seconds) }}s</strong></div>
    </div>
    <p>Differing columns: <code>{{ data.observed_trust_gap.differing_columns|join(", ") }}</code>.
      This protocol-specific observation is not presented as a universal causal effect.</p>
    <p>Raw protocol: <code>{{ data.observed_trust_gap.raw_protocol }}</code> · verified
      protocol: <code>{{ data.observed_trust_gap.verified_protocol }}</code> · splitter
      different: <strong>{{ boolean(data.observed_trust_gap.splitter_different) }}</strong>.</p>
    {% for warning in data.observed_trust_gap.warnings %}<div class="alert">{{ warning }}</div>{% endfor %}
    {% endif %}
  </section>{% endif %}

  <section class="card"><h2>Search summary</h2>
    <p>Families tested: {{ data.families|join(", ") }}.</p>
    <div class="grid">
      <div class="metric"><small>Successful</small><strong>{{ data.successful_count }}</strong></div>
      <div class="metric"><small>Failed</small><strong>{{ data.failed_count }}</strong></div>
      <div class="metric"><small>Interrupted</small><strong>{{ data.interrupted_count }}</strong></div>
    </div>
  </section>

  {% if data.runtime_telemetry %}<section class="card"><h2>Budget and runtime telemetry</h2>
    <p><code>budget_seconds</code> is the search-launch budget. It prevents new trials
      from starting after exhaustion; finalization and an already-running native operation
      may extend total wall-clock runtime.</p>
    <div class="grid">
      <div class="metric"><small>Search budget</small><strong>{{ seconds(data.runtime_telemetry.configured_search_budget_seconds) }}</strong></div>
      <div class="metric"><small>Search elapsed</small><strong>{{ seconds(data.runtime_telemetry.search_elapsed_seconds) }}</strong></div>
      <div class="metric"><small>Finalization</small><strong>{{ seconds(data.runtime_telemetry.finalization_elapsed_seconds) }}</strong></div>
      <div class="metric"><small>Total runtime</small><strong>{{ seconds(data.runtime_telemetry.total_runtime_seconds) }}</strong></div>
      <div class="metric"><small>Budget overshoot</small><strong>{{ seconds(data.runtime_telemetry.budget_overshoot_seconds) }}</strong></div>
      <div class="metric"><small>Budget remaining at search stop</small><strong>{{ seconds(data.runtime_telemetry.budget_remaining_at_search_stop_seconds) }}</strong></div>
      <div class="metric"><small>Stop reason</small><strong>{{ data.runtime_telemetry.stop_reason.value }}</strong></div>
    </div>
    <table><tbody>
      <tr><td>Search started</td><td>{{ timestamp(data.runtime_telemetry.search_started_at) }}</td></tr>
      <tr><td>Search finished</td><td>{{ timestamp(data.runtime_telemetry.search_finished_at) }}</td></tr>
      <tr><td>Trials started / completed / failed</td><td>{{ data.runtime_telemetry.trials_started }} / {{ data.runtime_telemetry.trials_completed }} / {{ data.runtime_telemetry.trials_failed }}</td></tr>
    </tbody></table>
  </section>{% endif %}

  <section class="card"><h2>Validation leaderboard{% if data.manifest.leakage_report.excluded_columns %} — post-neutralization{% endif %}</h2>
    <table><thead><tr><th>Rank</th><th>Family</th><th>Model</th><th>Fidelity</th>
      <th>{{ data.score_label }}</th><th>Std (internal)</th><th>Duration</th></tr></thead><tbody>
      {% for row in data.trials %}<tr><td>{{ row.rank }}</td><td>{{ row.family }}</td>
      <td>{{ row.model_name }}</td><td>L{{ row.fidelity_level }}</td><td>{{ row.score }}</td>
      <td>{{ row.uncertainty }}</td><td>{{ row.duration }}</td></tr>{% endfor %}
    </tbody></table>
  </section>

  {% if data.selection_explanation %}<section class="card"><h2>Why this pipeline won</h2>
    <p><strong>Selection reason:</strong> {{ data.selection_explanation.selection_reason }}</p>
    <p>{{ data.selection_explanation.selection_rule }}</p>
    {% if data.selection_explanation.tie_breaker_used %}<p><strong>Tie-breaker:</strong> {{ data.selection_explanation.tie_breaker_detail }}</p>{% endif %}
    <h3>Supporting context</h3><ul>{% for item in data.selection_explanation.supporting_context %}
      <li><strong>{{ item.label }}:</strong> {{ item.value }} <small>— source: {{ item.provenance }}</small></li>
    {% endfor %}</ul>
  </section>{% endif %}

  <section class="card"><h2>Best PipelineSpec</h2>
    <p>Selected family: <strong>{{ data.best_family }}</strong> · model:
      <strong>{{ data.best_model }}</strong>.</p><pre>{{ data.best_spec_json }}</pre></section>

  <section class="card"><h2>Failed trials</h2>
    {% if data.failed_trials %}<ul>{% for trial in data.failed_trials %}
      <li><code>{{ trial.trial_id }}</code> — {{ trial.failure_type }}: {{ trial.failure_message }}</li>
    {% endfor %}</ul>{% else %}<p class="ok">No failed trial.</p>{% endif %}
  </section>

  <section class="card"><h2>Artifacts</h2><ul>
    {% for name,path in data.artifacts %}<li><a href="{{ path }}">{{ name }}</a> — <code>{{ path }}</code></li>{% endfor %}
  </ul></section>

  <section class="card"><h2>Reproduce and inspect</h2>
    <pre>uv sync --frozen
uv run automl inspect &lt;run-directory&gt;
uv run automl leaderboard &lt;run-directory&gt;
uv run automl validate-artifacts &lt;run-directory&gt;
uv run automl trust &lt;run-directory&gt;</pre>
    <p>To rerun the exact search, keep the recorded CSV bytes unchanged and use
      <code>uv run automl resume &lt;run-directory&gt;</code> for an interrupted run.</p>
  </section>
  <section class="card"><h2>Scope and known limits</h2>
    <ul>
      <li>The score is a persisted cross-validation estimate, not a claim about unseen production data.</li>
      <li>Certified notebook export, inter-dataset memory (M11), and extended framework benchmarks (M14) are outside this frozen release.</li>
      <li>Joblib models are local artifacts and must only be loaded from trusted run directories after artifact validation.</li>
    </ul>
  </section>
  <footer>Generated deterministically from the persisted manifest, SQLite trial registry and artifact inventory.</footer>
</main></body></html>"""


__all__ = ["ReportData", "ReportTrial", "render_run_report"]

"""Deterministic compilation and rendering of a self-verified trust certificate."""

# The embedded HTML template is intentionally kept readable as HTML.
# ruff: noqa: E501

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime

from jinja2 import Environment, select_autoescape

from autonomous_automl.contracts import (
    FindingSeverity,
    MetricName,
    ObservedTrustGap,
    PipelineSelectionExplanation,
    ResumeEvidence,
    RunManifest,
    RuntimeTelemetry,
    SourceHashEvidence,
    TrialResult,
    TrialStatus,
    TrustCertificate,
    VerificationEvidence,
    derive_certificate_status,
)
from autonomous_automl.evaluation.trust_gap import metric_direction


def compile_trust_certificate(
    manifest: RunManifest,
    trials: Sequence[TrialResult],
    *,
    generated_at: datetime,
    git_commit: str | None,
    artifact_validation: VerificationEvidence,
    prediction_replay: VerificationEvidence,
    resume: ResumeEvidence,
    runtime_telemetry: RuntimeTelemetry | None,
    selection_explanation: PipelineSelectionExplanation | None,
    observed_trust_gap: ObservedTrustGap | None,
) -> TrustCertificate:
    """Compile certificate evidence without reading runtime-only model objects."""

    resolved_metric = _resolved_metric(manifest, trials)
    unresolved_critical = [
        finding
        for finding in manifest.leakage_report.findings
        if finding.severity is FindingSeverity.CRITICAL
        and (
            finding.column is None or finding.column not in manifest.leakage_report.excluded_columns
        )
    ]
    warnings = _certificate_warnings(
        manifest,
        runtime_telemetry,
        selection_explanation,
        prediction_replay,
        observed_trust_gap,
    )
    final_pipeline = manifest.best_pipeline
    status = derive_certificate_status(
        run_status=manifest.status,
        final_pipeline_present=final_pipeline is not None,
        artifact_status=artifact_validation.status,
        replay_status=prediction_replay.status,
        unresolved_critical_count=len(unresolved_critical),
        has_warnings=bool(warnings or manifest.leakage_report.findings),
    )
    return TrustCertificate(
        status=status,
        run_id=manifest.run_id,
        run_status=manifest.status,
        generated_at=generated_at,
        package_version=manifest.package_version,
        git_commit=git_commit,
        source_hashes=_source_hash_evidence(manifest),
        task=manifest.dataset_profile.inferred_task,
        metric=resolved_metric,
        metric_direction=(None if resolved_metric is None else metric_direction(resolved_metric)),
        verified_score=manifest.metrics.get("best_score"),
        validation_strategy=manifest.validation_plan.splitter_name,
        n_folds=manifest.validation_plan.n_splits,
        used_columns=_used_columns(manifest),
        excluded_columns=_excluded_columns(manifest),
        leakage_findings=list(manifest.leakage_report.findings),
        unresolved_critical_findings=unresolved_critical,
        final_pipeline=final_pipeline,
        selected_model=None if final_pipeline is None else final_pipeline.model_name,
        trials_completed=sum(trial.status is TrialStatus.COMPLETED for trial in trials),
        trials_failed=sum(trial.status is TrialStatus.FAILED for trial in trials),
        runtime_telemetry=runtime_telemetry,
        artifact_validation=artifact_validation,
        prediction_replay=prediction_replay,
        resume=resume,
        selection_explanation=selection_explanation,
        observed_trust_gap=observed_trust_gap,
        warnings=warnings,
        limitations=[
            "The verified score is a recorded validation estimate, not a production guarantee.",
            "Leakage findings are deterministic engine diagnostics, not a complete domain audit.",
            "A certificate cannot attest to its own bytes; `automl trust` validates its "
            "registered hash when the run is reloaded.",
            "The certificate does not assess commercial, scientific, ethical, or regulatory fitness.",
        ],
    )


def render_trust_certificate(certificate: TrustCertificate) -> str:
    """Render a standalone, autoescaped HTML view from one certificate contract."""

    environment = Environment(
        autoescape=select_autoescape(default_for_string=True),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    best_spec = (
        "Unavailable"
        if certificate.final_pipeline is None
        else json.dumps(
            certificate.final_pipeline.to_json_value(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return environment.from_string(_TRUST_TEMPLATE).render(
        certificate=certificate,
        best_spec=best_spec,
        score=_format_number(certificate.verified_score),
        gap=(
            "Unavailable"
            if certificate.observed_trust_gap is None
            or certificate.observed_trust_gap.observed_trust_gap is None
            else f"{certificate.observed_trust_gap.observed_trust_gap:+.6f}"
        ),
        seconds=_format_seconds,
        timestamp=_format_timestamp,
        boolean=_format_boolean,
    )


def _resolved_metric(
    manifest: RunManifest,
    trials: Sequence[TrialResult],
) -> MetricName | None:
    metrics = {trial.primary_metric for trial in trials if trial.status is TrialStatus.COMPLETED}
    if len(metrics) == 1:
        return next(iter(metrics))
    if manifest.configuration.metric is not MetricName.AUTO:
        return manifest.configuration.metric
    return None


def _source_hash_evidence(manifest: RunManifest) -> list[SourceHashEvidence]:
    evidence: list[SourceHashEvidence] = []
    for index, path in enumerate(manifest.dataset.train_paths, start=1):
        evidence.append(
            SourceHashEvidence(
                role=f"train_{index}",
                filename=path.name,
                sha256=manifest.source_hashes[str(path)],
            )
        )
    if manifest.dataset.test_path is not None:
        evidence.append(
            SourceHashEvidence(
                role="test",
                filename=manifest.dataset.test_path.name,
                sha256=manifest.source_hashes[str(manifest.dataset.test_path)],
            )
        )
    return evidence


def _used_columns(manifest: RunManifest) -> list[str]:
    excluded = set(
        [] if manifest.best_pipeline is None else manifest.best_pipeline.excluded_columns
    )
    return [column for column in manifest.dataset.feature_columns if column not in excluded]


def _excluded_columns(manifest: RunManifest) -> list[str]:
    values = [
        *manifest.dataset.excluded_columns,
        *manifest.leakage_report.excluded_columns,
    ]
    return list(dict.fromkeys(values))


def _certificate_warnings(
    manifest: RunManifest,
    runtime: RuntimeTelemetry | None,
    explanation: PipelineSelectionExplanation | None,
    replay: VerificationEvidence,
    trust_gap: ObservedTrustGap | None,
) -> list[str]:
    warnings: list[str] = []
    if runtime is None:
        warnings.append("Detailed budget and runtime telemetry is unavailable for this run.")
    if explanation is None:
        warnings.append("A persisted selection explanation is unavailable for this run.")
    if replay.status.value == "NOT_AVAILABLE":
        warnings.append("Prediction replay is unavailable because no comparable predictions exist.")
    if manifest.validation_audit is None:
        warnings.append("The persisted validation audit is unavailable.")
    if (
        trust_gap is not None
        and trust_gap.status.value == "NOT_COMPUTED"
        and (manifest.configuration.compute_trust_gap)
    ):
        warnings.append(f"Observed Trust Gap was not computed: {trust_gap.reason}")
    return warnings


def _format_number(value: float | None) -> str:
    return "Unavailable" if value is None else f"{value:.6f}"


def _format_seconds(value: float | None) -> str:
    return "Unavailable" if value is None else f"{value:.3f} s"


def _format_timestamp(value: datetime | None) -> str:
    return "Unavailable" if value is None else value.isoformat()


def _format_boolean(value: bool | None) -> str:
    if value is None:
        return "Unavailable"
    return "Yes" if value else "No"


_TRUST_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Self-verified trust certificate — {{ certificate.run_id }}</title>
  <style>
    :root { color-scheme:light; --ink:#172033; --muted:#596a84; --line:#d9e1ec;
      --panel:#f6f8fb; --good:#087a55; --warn:#9a5805; --bad:#b42318; --blue:#3157d5; }
    * { box-sizing:border-box; } body { margin:0; background:#eef2f7; color:var(--ink);
      font:15px/1.48 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
    main { max-width:1060px; margin:auto; padding:28px 18px 48px; }
    header,.card { background:#fff; border:1px solid var(--line); border-radius:12px;
      box-shadow:0 7px 25px #23345a10; } header,.card { padding:22px; margin-bottom:16px; }
    h1,h2 { margin-top:0; } h1 { margin-bottom:5px; font-size:28px; } h2 { font-size:19px; }
    p { color:var(--muted); } .status { display:inline-block; padding:7px 11px;
      border-radius:7px; background:var(--panel); font-weight:750; }
    .SELF_VERIFIED { color:var(--good); } .SELF_VERIFIED_WITH_WARNINGS { color:var(--warn); }
    .INCOMPLETE,.FAILED { color:var(--bad); } .grid { display:grid;
      grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:10px; }
    .metric { background:var(--panel); border-radius:8px; padding:12px; }
    .metric small { color:var(--muted); display:block; } .metric strong { font-size:17px; }
    table { width:100%; border-collapse:collapse; } th,td { padding:8px;
      border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
    th { color:var(--muted); font-size:12px; text-transform:uppercase; }
    code,pre { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; } pre {
      background:#101827; color:#e8eefc; padding:14px; border-radius:8px; overflow:auto; }
    .disclaimer { border-left:5px solid var(--blue); background:#eef3ff; padding:13px 15px;
      color:var(--ink); font-weight:650; } .warning { border-left:4px solid var(--warn);
      background:#fff8ed; padding:9px 12px; margin:7px 0; } a { color:var(--blue); }
  </style>
</head>
<body><main>
  <header>
    <h1>Self-verified trust certificate</h1>
    <p>Recorded run <code>{{ certificate.run_id }}</code></p>
    <p class="status {{ certificate.status.value }}">{{ certificate.status.value }}</p>
    <div class="grid">
      <div class="metric"><small>Verified score</small><strong>{{ score }}</strong></div>
      <div class="metric"><small>Task</small><strong>{{ certificate.task.value }}</strong></div>
      <div class="metric"><small>Metric</small><strong>{% if certificate.metric %}{{ certificate.metric.value }}{% else %}Unavailable{% endif %}</strong></div>
      <div class="metric"><small>Selected model</small><strong>{{ certificate.selected_model or "Unavailable" }}</strong></div>
      <div class="metric"><small>Observed Trust Gap</small><strong>{{ gap }}</strong></div>
    </div>
  </header>

  <section class="card"><p class="disclaimer">{{ certificate.disclaimer }}</p></section>

  <section class="card"><h2>Verification</h2>
    <table><thead><tr><th>Check</th><th>Status</th><th>Recorded evidence</th></tr></thead><tbody>
      <tr><td>Artifact integrity</td><td>{{ certificate.artifact_validation.status.value }}</td><td>{{ certificate.artifact_validation.detail }}</td></tr>
      <tr><td>Prediction replay</td><td>{{ certificate.prediction_replay.status.value }}</td><td>{{ certificate.prediction_replay.detail }}</td></tr>
      <tr><td>Validation</td><td>{% if certificate.n_folds %}RECORDED{% else %}Unavailable{% endif %}</td><td>{{ certificate.validation_strategy }} · {{ certificate.n_folds }} folds</td></tr>
    </tbody></table>
  </section>

  <section class="card"><h2>Leakage risks and exclusions</h2>
    <p>Used columns: {% if certificate.used_columns %}<code>{{ certificate.used_columns|join(", ") }}</code>{% else %}Unavailable{% endif %}</p>
    <p>Excluded columns: {% if certificate.excluded_columns %}<code>{{ certificate.excluded_columns|join(", ") }}</code>{% else %}none{% endif %}</p>
    {% if certificate.leakage_findings %}<ul>{% for finding in certificate.leakage_findings %}
      <li><strong>{{ finding.severity.value }} · {{ finding.finding_type.value }}</strong>{% if finding.column %} — <code>{{ finding.column }}</code>{% endif %}: {{ finding.action }}</li>
    {% endfor %}</ul>{% else %}<p>No heuristic leakage finding was recorded.</p>{% endif %}
    <p>Unresolved critical findings: <strong>{{ certificate.unresolved_critical_findings|length }}</strong></p>
  </section>

  {% if certificate.selection_explanation %}<section class="card"><h2>Why this pipeline won</h2>
    <p><strong>Selection reason:</strong> {{ certificate.selection_explanation.selection_reason }}</p>
    <p>{{ certificate.selection_explanation.selection_rule }}</p>
    {% if certificate.selection_explanation.tie_breaker_used %}<p><strong>Tie-breaker:</strong> {{ certificate.selection_explanation.tie_breaker_detail }}</p>{% endif %}
    <h3>Supporting context</h3><ul>{% for item in certificate.selection_explanation.supporting_context %}
      <li><strong>{{ item.label }}:</strong> {{ item.value }} <small>— {{ item.provenance }}</small></li>
    {% endfor %}</ul>
  </section>{% endif %}

  {% if certificate.runtime_telemetry %}<section class="card"><h2>Budget and runtime</h2>
    <div class="grid">
      <div class="metric"><small>Search budget</small><strong>{{ seconds(certificate.runtime_telemetry.configured_search_budget_seconds) }}</strong></div>
      <div class="metric"><small>Search elapsed</small><strong>{{ seconds(certificate.runtime_telemetry.search_elapsed_seconds) }}</strong></div>
      <div class="metric"><small>Finalization</small><strong>{{ seconds(certificate.runtime_telemetry.finalization_elapsed_seconds) }}</strong></div>
      <div class="metric"><small>Total runtime</small><strong>{{ seconds(certificate.runtime_telemetry.total_runtime_seconds) }}</strong></div>
      <div class="metric"><small>Budget remaining at search stop</small><strong>{{ seconds(certificate.runtime_telemetry.budget_remaining_at_search_stop_seconds) }}</strong></div>
      <div class="metric"><small>Budget overshoot</small><strong>{{ seconds(certificate.runtime_telemetry.budget_overshoot_seconds) }}</strong></div>
      <div class="metric"><small>Stop reason</small><strong>{{ certificate.runtime_telemetry.stop_reason.value }}</strong></div>
    </div>
    <table><tbody>
      <tr><td>Search started</td><td>{{ timestamp(certificate.runtime_telemetry.search_started_at) }}</td></tr>
      <tr><td>Search finished</td><td>{{ timestamp(certificate.runtime_telemetry.search_finished_at) }}</td></tr>
      <tr><td>Trials started / completed / failed</td><td>{{ certificate.runtime_telemetry.trials_started }} / {{ certificate.runtime_telemetry.trials_completed }} / {{ certificate.runtime_telemetry.trials_failed }}</td></tr>
    </tbody></table>
  </section>{% endif %}

  <section class="card"><h2>Observed Trust Gap</h2>
    {% if certificate.observed_trust_gap %}
      <p><strong>{{ certificate.observed_trust_gap.status.value }}</strong> — {{ certificate.observed_trust_gap.reason }}</p>
      {% if certificate.observed_trust_gap.status.value == "COMPUTED" %}
      <table><tbody>
        <tr><td>Raw diagnostic score</td><td>{{ "%.6f"|format(certificate.observed_trust_gap.raw_score) }}</td></tr>
        <tr><td>Verified score</td><td>{{ "%.6f"|format(certificate.observed_trust_gap.verified_score) }}</td></tr>
        <tr><td>Apparent score inflation under the raw protocol</td><td>{{ "%+.6f"|format(certificate.observed_trust_gap.observed_trust_gap) }}</td></tr>
        <tr><td>Differing columns</td><td>{{ certificate.observed_trust_gap.differing_columns|join(", ") }}</td></tr>
        <tr><td>Raw protocol</td><td>{{ certificate.observed_trust_gap.raw_protocol }}</td></tr>
        <tr><td>Verified protocol</td><td>{{ certificate.observed_trust_gap.verified_protocol }}</td></tr>
        <tr><td>Splitter different</td><td>{{ boolean(certificate.observed_trust_gap.splitter_different) }}</td></tr>
      </tbody></table>
      {% endif %}
      {% for warning in certificate.observed_trust_gap.warnings %}<div class="warning">{{ warning }}</div>{% endfor %}
    {% else %}<p>Unavailable.</p>{% endif %}
  </section>

  <section class="card"><h2>Best PipelineSpec</h2><pre>{{ best_spec }}</pre></section>

  {% if certificate.warnings %}<section class="card"><h2>Warnings</h2>{% for warning in certificate.warnings %}<div class="warning">{{ warning }}</div>{% endfor %}</section>{% endif %}
  <section class="card"><h2>Limits</h2><ul>{% for limit in certificate.limitations %}<li>{{ limit }}</li>{% endfor %}</ul></section>
  <section class="card"><h2>Reproduce</h2><pre>uv sync --frozen
uv run automl validate-artifacts &lt;run-directory&gt;
uv run automl trust &lt;run-directory&gt;</pre><p><a href="report.html">Open the full run report</a></p></section>
</main></body></html>"""


__all__ = ["compile_trust_certificate", "render_trust_certificate"]

# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project follows semantic versioning.

## [Unreleased]

### Added

- Self-verified trust certificates in deterministic JSON and standalone HTML,
  derived from persisted run evidence and exposed through `automl trust`.
- Factual `Why this pipeline won` evidence with the exact existing selection rule
  and sourced supporting context.
- Separate search-launch budget, search, finalization, total-runtime, trial-count,
  overshoot and stop-reason telemetry.
- Optional isolated Observed Trust Gap diagnostics, disabled by default and never
  used by Optuna, the main leaderboard, or final selection.
- Targeted tests for certificate status/corruption, Trust Gap direction and
  isolation, selection explanations, runtime/resume telemetry, old-run loading,
  CLI and HTML consistency.

### Changed

- The three-scenario demo now emits trust certificates for every completed run and
  enables the Trust Gap only for its synthetic leakage scenario.
- The report and CLI expose persisted Trust Layer evidence and clarify that
  `budget_seconds` is a search-launch budget.

### Compatibility

- No SQLite migration or destructive manifest change. The new configuration fields
  are additive; pre-Trust payloads load with the diagnostic disabled.

## [0.1.0] - 2026-07-18

### Added

- Complete Python 3.12/`uv` tabular AutoML core through adaptive multi-fidelity
  search (M0–M9).
- Public `AutoMLRun.fit` and resumable `AutoMLRun.resume` orchestration.
- Checksum-verified Joblib model loading and persisted prediction replay.
- Rich CLI commands for fit, resume, inspection, leaderboard, prediction, artifact
  validation and the three-scenario Build Week demo.
- Standalone HTML reports generated from manifests, SQLite and registered artifacts.
- Deterministic classification, regression and synthetic-leakage examples.
- Build Week judge, video, Devpost and Codex collaboration documentation.
- Optional Docker execution path.

### Changed

- Froze the submission scope after M9; cross-run memory and heavy benchmarks are
  deferred.
- Preloaded the evaluator module in forkserver workers to reduce safe short-run
  process startup overhead.

### Not included

- Certified notebook generation is explicitly deferred from this release.

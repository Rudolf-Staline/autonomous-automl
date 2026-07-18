# Build Week video script — 2:45 target

All spoken copy below is intentionally in natural English. Prepare the environment
off camera with `uv sync --frozen`, use a terminal at least 110 columns wide, and
ensure `runs/video-demo` does not exist.

## 0:00–0:18 — Problem

**On screen:** README title and product statement.

**Say:**

> Most AutoML systems optimize a score. Autonomous AutoML verifies whether that
> score deserves to be trusted. A target leak, preprocessing outside a fold, or a
> stale saved model can make a great score meaningless.

**Transition:** scroll once to the one-command demonstration.

## 0:18–0:48 — Launch the complete path

**On screen:** terminal.

**Run:**

```bash
uv run automl demo --output-root runs/video-demo
```

**Say while it runs:**

> One local command runs three small, deterministic scenarios. The terminal shows
> the active stage, budget remaining, completed and failed trials, and the current
> search objective. Classification deliberately checkpoints after two trials and
> resumes from SQLite without repeating them.

**Expected screen:** the three scenario headings, `Checkpoint persisted`, the
summary table, a top-five leaderboard, leakage proof, and three `PASS` checks. The
final audited commands took 16.21–19.53 seconds.

## 0:48–1:15 — Leakage evidence

**On screen:** keep the final leakage-proof table visible, then run:

```bash
uv run automl inspect runs/video-demo/leakage
```

**Say:**

> This synthetic CSV contains an exact target copy, a post-outcome copy, and a
> unique customer identifier. The engine records the reason and action, excludes
> all three before pipeline generation, and uses persisted stratified folds. The
> displayed score is post-neutralization. We do not invent a contaminated score.

**Transition:** switch back to the demo summary or the leaderboard command.

## 1:15–1:38 — Search and leaderboard

**Run:**

```bash
uv run automl leaderboard runs/video-demo/classification-resumed --limit 5
```

**Say:**

> Compatible PipelineSpecs combine fold-local preprocessing with several
> scikit-learn model families. Optuna suggests parameters, while multi-fidelity and
> adaptive allocation spend the short budget. A naive baseline is always included,
> and a candidate failure remains in the registry instead of ending the run.

## 1:38–2:00 — Resume and artifact proof

**On screen:** show the run directory, including `registry.sqlite3`,
`best_pipeline_spec.json`, `best_pipeline.joblib`, and `predictions.csv`.

**Run:**

```bash
uv run automl validate-artifacts runs/video-demo/classification-resumed
```

**Say:**

> Validation checks SQLite, source hashes, every registered artifact, and the final
> Joblib before loading it. The loaded pipeline replays the test CSV, and its
> predictions must match the persisted file within the recorded tolerance.

## 2:00–2:24 — Standalone report

**Open:** `runs/video-demo/leakage/report.html` directly in a browser.

**On screen:** dataset and validation, leakage diagnostics, post-neutralization
leaderboard, best PipelineSpec, failed trials, then artifact links.

**Say:**

> This standalone HTML is generated only from the persisted manifest, SQLite trial
> registry, and artifact inventory. It explains the validation plan, exclusions,
> failures, selected PipelineSpec, budget, and reproduction commands without a
> server.

## 2:24–2:38 — Codex and GPT-5.6 disclosure

**Open:** `docs/CODEX_COLLABORATION.md`.

**Say:**

> Codex implemented and release-audited the repository, measured real demos, and
> corrected failures against the gates. The available record does not establish a
> separate GPT-5.6 Sol session, so the submission makes no unsupported attribution.

## 2:38–2:45 — Close

**Return to:** README product statement.

**Say:**

> Autonomous AutoML does not just save a model. It saves the evidence needed to
> decide whether the model can be trusted.

## Backup plan

- If dependency installation is slow, keep it off camera; the video starts after
  the already documented `uv sync --frozen` step.
- If the demo exceeds 30 seconds, let it finish and use a transparent time cut in
  the recording. Do not cut away failures or the final `PASS` summary.
- If Windows process startup causes a short-budget failure, rerun before recording
  with `uv run automl demo --budget 15s --output-root runs/video-demo`.
- If browser launch integration is unavailable, open
  `runs/video-demo/leakage/report.html` using the file manager; no server is needed.
- Keep a completed `runs/video-demo` only as an on-camera fallback for the inspect,
  leaderboard, validation, and report segments. State clearly if the live search
  was recorded separately.

# Build Week video script — 2:38 target

All spoken copy below is exact, short, and intended to be easy to pronounce in
English. Prepare off camera with `uv sync --frozen`. Use a terminal at least 110
columns wide and ensure `runs/video-demo` does not exist.

## 0:00–0:17 — Problem

**On screen:** README title and product statement.

**Say:**

> Most AutoML systems optimize a score. Autonomous AutoML verifies whether that
> score deserves to be trusted. A target leak or a stale saved model can make a
> great score meaningless.

**Transition:** move to the terminal.

## 0:17–0:45 — Launch the complete path

**Run:**

```bash
uv run automl demo --output-root runs/video-demo
```

**Say while it runs:**

> One local command runs classification, regression, and a synthetic leakage
> attack. The terminal shows the stage, search budget, trial counts, and current
> objective. Classification checkpoints after two trials, then resumes from
> SQLite without repeating them.

**Expected screen:** three scenario headings, `Checkpoint persisted`, the final
summary, leakage table, leaderboard, and `TRUST SUMMARY`. The audited Trust Layer
demo took 16.76 seconds.

## 0:45–1:12 — Leakage and Observed Trust Gap

**On screen:** keep `TRUST SUMMARY` visible.

**Say:**

> The raw CSV contains two target copies and a unique customer identifier. The
> engine excludes them before search. Every main leaderboard score is verified
> after neutralization. After selection, a separate diagnostic evaluates the same
> pipeline and folds with the recorded risk columns restored. Here, the raw score
> is one, the verified score is about zero point eight four, and the Observed Trust
> Gap is about plus zero point one six. This is apparent inflation under this
> protocol, not a universal causal claim.

**Transition:** show the classification leaderboard, then the trust command.

## 1:12–1:35 — Search and why the pipeline won

**Run:**

```bash
uv run automl leaderboard runs/video-demo/classification-resumed --limit 5
uv run automl trust runs/video-demo/leakage
```

**Say:**

> Optuna, multi-fidelity search, and adaptive family allocation evaluate compatible
> PipelineSpecs. A naive baseline is always present. The certificate explains why
> the final pipeline won by repeating the real selection rule. Supporting facts
> name their persisted source. The Trust Gap never enters this leaderboard or the
> selection rule.

## 1:35–1:57 — Resume, budget, and artifact proof

**Run:**

```bash
uv run automl validate-artifacts runs/video-demo/classification-resumed
```

**Say:**

> Validation checks SQLite, source hashes, registered artifacts, and Joblib before
> loading. It replays the test CSV and compares saved predictions. Search time and
> finalization time are recorded separately. The configured value is a
> search-launch budget, not a hard total-runtime promise.

## 1:57–2:20 — Certificate and full report

**Open first:** `runs/video-demo/leakage/trust_certificate.html`.

**Then open:** `runs/video-demo/leakage/report.html`.

**Say:**

> This Self-verified trust certificate is generated only from recorded run
> evidence. It shows validation, exclusions, replay, runtime, Trust Gap, warnings,
> and limits. It is not an external, security, or regulatory certification. The
> full standalone report adds the dataset profile, trials, leaderboard, selected
> PipelineSpec, and reproduction commands. Neither file needs a server.

## 2:20–2:33 — Codex disclosure

**Open:** `docs/CODEX_COLLABORATION.md`.

**Say:**

> This is a solo project assisted by Codex. Codex implemented and audited the
> repository, including this Trust Layer and its adversarial tests. The available
> record does not prove a separate GPT-5.6 Sol session, so I make no unsupported
> attribution.

## 2:33–2:38 — Close

**Return to:** the certificate status or README statement.

**Say:**

> Autonomous AutoML returns a model, and the evidence needed to question its score.

## Backup plan

- Keep dependency installation off camera; it is already documented and tested.
- If the live demo takes longer than 30 seconds, let it finish and use a transparent
  time cut. Never hide a failure or omit the final summary.
- If Windows process startup exhausts the short allowance, rerun with
  `uv run automl demo --budget 15s --output-root runs/video-demo`.
- Keep one completed `runs/video-demo` as a clearly disclosed fallback for the
  leaderboard, trust, validation, certificate, and report screens.
- If a browser file URL does not open from the terminal, use the file manager. Both
  HTML files work without a server.

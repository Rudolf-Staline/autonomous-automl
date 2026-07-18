# Autonomous AutoML

**Most AutoML systems optimize a score. Autonomous AutoML verifies whether the
score deserves to be trusted.**

Autonomous AutoML turns tabular CSV data into a leakage-aware diagnosis, a
budgeted search, a selected scikit-learn pipeline, predictions, and a standalone
run report. It runs locally without an API key or external service.

## The problem

Tabular ML can produce an impressive but invalid score when a target copy, a
post-outcome field, or an identifier reaches validation. It can also become
irreproducible when preprocessing is fitted outside folds, failed trials disappear,
or the saved model no longer matches the experiment that selected it.

This project makes those risks persisted, testable decisions. Leakage safety,
reproducibility, and recoverability take priority over a marginally higher score.

## Quick demonstration

From a clone of this repository:

```bash
uv sync --frozen
uv run automl demo
```

The command runs binary classification with interruption/resume, regression, and a
synthetic leakage attack. Across the final Linux RC gates it completed in **16.21 to
19.53 seconds** after installation, with a maximum observed resident set of about
**225 MiB**. A recent laptop should normally finish in 20–30 seconds; process startup
can be slower on Windows.

The CLI prints the demo root; the leakage report is
`<demo-root>/leakage/report.html`. The frozen scope is tabular CSV: certified
notebook export, M11/M14, and a web UI are not included.

## Installation

Requirements: Git, internet access for the first dependency sync, and
[`uv`](https://docs.astral.sh/uv/getting-started/installation/). The committed
lockfile and `.python-version` select Python 3.12.

```bash
git clone <repository-url>
cd AutoML
uv sync --frozen
```

Official `uv` installer commands:

Linux and macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

No `.env`, API key, GPU, Docker daemon, cloud account, or database server is
required. Minimum practical resources are 2 CPU cores, 2 GiB RAM, and about 1 GiB
of free disk for the environment.

Troubleshooting:

- If `uv` is not found after installation, restart the shell or use the path shown
  by the installer.
- Run outputs are immutable. Choose a new `--output-root` or `--output` when a
  directory already exists.
- A source-hash mismatch means the input CSV bytes changed; restore them or start a
  new run.
- A recorded trial timeout near a short budget boundary is not a run failure.
- On Windows, use `uv run automl demo --budget 15s` if process startup consumes too
  much of the six-second per-scenario default.

## One-command path

After installation, the complete judge path is:

```bash
uv run automl demo
```

## Expected result

The terminal shows each stage, consumed/remaining budget, trial counts, the current
search objective, a final top-five leaderboard, leakage evidence/actions, artifact
checks, and the exact report command. It ends with:

```text
All scenarios passed.
```

The invariant results are:

- classification selects ROC AUC, intentionally checkpoints after two trials,
  resumes without duplicating them, and writes 36 predictions;
- regression selects RMSE and writes 36 finite predictions;
- the leakage scenario excludes `target_copy`, `approved_after_review`, and
  `customer_id` before search;
- every final model loads through its registered size/SHA-256 record;
- replayed test predictions match the persisted predictions;
- each run contains a standalone `report.html`.

Exact short-budget scores and the family selected at a time boundary can vary with
hardware. The score-independent assertions are recorded in
[`examples/EXPECTED_RESULTS.md`](examples/EXPECTED_RESULTS.md).

## Features in the frozen release

- binary and multiclass classification, plus regression;
- one or more training CSVs and an isolated optional target-free test CSV;
- train-only profiling and automatic task/metric resolution;
- explainable target-copy, near-copy, suspicious-name, post-outcome, and identifier
  diagnostics;
- stratified, K-fold, group, temporal, and predefined validation plans;
- fold-local imputation and encoding with unknown-category-safe inference;
- compatible-only, JSON-serializable `PipelineSpec` candidates;
- a mandatory naive baseline and multiple core scikit-learn model families;
- persistent Optuna studies, four fidelity levels, and adaptive family allocation;
- per-trial timeout/error containment with failed trials retained in SQLite;
- interruption/resume with budget and completed-trial preservation;
- final full-data fit, Joblib registration, prediction replay, and HTML reporting.

Optional XGBoost, LightGBM, and CatBoost adapters are lazy. Their absence does not
break the required scikit-learn core.

## Architecture

```text
CSV -> profile -> leakage guard -> persisted validation folds
    -> compatible PipelineSpecs -> Optuna + multi-fidelity allocation
    -> leaderboard -> final PipelineSpec -> Joblib + predictions
    -> SQLite/manifest verification -> standalone HTML report
```

The package separates descriptions from executables:

```text
contracts/   versioned JSON contracts and PipelineSpec source of truth
data/        explicit CSV loading, hashes, target/test firewalls
profiling/   train-only statistics, task inference, leakage findings
validation/  persisted folds and overlap/group/time audits
pipelines/   compatibility grammar and deterministic sklearn factories
evaluation/  fold-local fitting, metrics, OOF, timeouts, final training
search/      Optuna, fidelity scheduler, budget, family allocator
tracking/    SQLite transactions, resume, confined/checksummed artifacts
api/         fit, resume, safe load, prediction, replay validation
reporting/   artifact-backed leaderboard and standalone HTML
cli/         Typer/Rich presentation
```

SQLite is the authority for mutable run state. `manifest.json` is a readable
inventory, while `PipelineSpec` is the reconstructible logical source of truth for
the selected pipeline.

## Leakage guarantees

- The target is removed before features reach profiling or pipeline factories.
- Test CSV data never influences profiling, folds, candidate generation, search,
  ranking, or final selection.
- Exact/near target copies and high-confidence identifiers are excluded before
  pipeline generation and recorded with evidence and action.
- The report labels leakage-scenario leaderboard scores as post-neutralization; it
  does not invent a contaminated comparison score.
- Learned imputers and encoders live inside the pipeline fitted on each training
  fold.
- One-hot encoding ignores unknown categories; other encoders define fallbacks.
- Target encoding is intentionally absent because this release does not contain a
  certified cross-fitted implementation.
- CSV values are never evaluated as code, and structured logs omit raw rows.

## CLI

```text
automl fit                 run CSV-to-report training
automl resume              continue an interrupted run
automl inspect             show persisted diagnostics and budget state
automl leaderboard         print the persisted ranking
automl predict             score a target-free CSV with the verified model
automl validate-artifacts  verify SQLite, hashes, Joblib, and prediction replay
automl demo                run the three Build Week scenarios
automl doctor              show package, Python, and SQLite versions
```

Fit a CSV directly:

```bash
uv run automl fit data/train.csv \
  --target target \
  --task auto \
  --metric auto \
  --budget 30s \
  --test data/test.csv \
  --output runs/my-run
```

Use a JSON configuration and demonstrate resume:

```bash
uv run automl fit examples/classification_train.csv \
  --config examples/classification_config.json \
  --output runs/resume-demo \
  --interrupt-after-trials 2
uv run automl resume runs/resume-demo --additional-budget 3s
uv run automl inspect runs/resume-demo
uv run automl validate-artifacts runs/resume-demo
```

Run `uv run automl --help` or `uv run automl COMMAND --help` for all options.

## Python API

```python
from autonomous_automl import AutoMLConfig, AutoMLRun

run = AutoMLRun(
    AutoMLConfig(
        target="target",
        task="auto",
        metric="auto",
        budget_seconds=30,
        random_seed=42,
        output_dir="runs/python-api",
    )
)
result = run.fit("data/train.csv")
print(result.best_pipeline_path, result.report_path)
```

## Artifacts produced

```text
run-directory/
├── manifest.json
├── configuration.json
├── dataset_profile.json
├── leakage_report.json
├── validation_plan.json
├── validation_audit.json
├── registry.sqlite3
├── optuna.sqlite3
├── trials.csv
├── leaderboard.csv
├── best_pipeline_spec.json
├── best_pipeline.joblib
├── predictions.csv              # only when a test CSV is supplied
├── environment.json
├── logs/run.jsonl
└── report.html
```

Joblib is loaded only from a locally generated run after registration checks. Do
not load arbitrary external Joblib files.

## Tests and release gates

```bash
uv sync --frozen
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

`uv run python scripts/quality.py` runs the aggregate quality checks. The suite
covers unit, integration, leakage, reproducibility, resume, corruption detection,
and artifact/prediction replay behavior. Heavy comparative benchmarks are not part
of the standard gate.

## Known limitations

- The release is limited to tabular CSV data; there is no web UI, NLP, vision, or
  distributed search.
- M11 inter-dataset memory/meta-ranking and M14 extended framework benchmarks are
  deferred.
- A certified generated notebook is not delivered. `RunResult.notebook_path` is
  `None`; reproducibility is verified through the manifest, `PipelineSpec`,
  registered Joblib, and prediction replay.
- Short budgets are soft global envelopes with protected finalization reserves; a
  native operation already in progress can finish slightly beyond a boundary.
- Cross-machine floating-point behavior can change short-budget scores or candidate
  ordering even with fixed seeds; persisted predictions are checked within the run.
- The provided Dockerfile is an optional convenience. The audited release path is
  `uv sync --frozen` followed by `uv run automl demo`.

See the [judge guide](docs/JUDGE_GUIDE.md), [video script](docs/DEMO_SCRIPT.md),
[Devpost copy](docs/DEVPOST_SUBMISSION.md), [submission checklist](docs/SUBMISSION_CHECKLIST.md),
and [implementation evidence](IMPLEMENTATION_STATUS.md).

## Codex and GPT-5.6 Sol usage

Codex acted as the repository implementation and release-audit agent. It followed
the supplied specification order, built and integrated M0–M9, created tests and
examples, measured real demos, diagnosed failures, verified resume/corruption
behavior, and synchronized the CLI, report, and submission documentation with
observed outputs.

The available transcript and repository do not identify a separate, verifiable
GPT-5.6 Sol session. No work is therefore attributed to GPT-5.6 Sol. The honest
chronology and the decisions Codex accelerated are in
[`docs/CODEX_COLLABORATION.md`](docs/CODEX_COLLABORATION.md).

This is a solo Build Week project assisted by Codex. The project owner supplied the
direction, specifications, and delivery pivots; no separate human implementation
team is claimed.

## License

MIT License. See [`LICENSE`](LICENSE).

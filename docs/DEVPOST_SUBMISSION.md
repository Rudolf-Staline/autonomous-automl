# Devpost submission — final copy

The recommended track is **Developer Tools**, because the official track covers
developer-facing testing, workflows, and security. Confirm that selection manually
on Devpost before submission.

## Project name

Autonomous AutoML

## Tagline

Most AutoML systems optimize a score. Autonomous AutoML verifies whether the score
deserves to be trusted.

## Inspiration

An AutoML score can look excellent for the wrong reason. Target copies,
post-outcome fields, unique identifiers, preprocessing fitted outside validation
folds, and stale model artifacts can all create results that do not survive real
use. We wanted a local tabular AutoML engine where leakage safety, recoverability,
and reproducibility are part of the product rather than afterthoughts.

## What it does

Autonomous AutoML accepts one or more training CSVs, a target, task/metric choices
or `auto`, and a wall-time budget. It profiles only the training data, records
explainable leakage and identifier findings, excludes high-confidence risks before
search, and materializes a validation plan. It generates compatible PipelineSpecs
whose learned preprocessing stays inside each training fold.

The engine evaluates a mandatory baseline and multiple scikit-learn model families.
Optuna suggests hyperparameters while a four-level multi-fidelity scheduler and an
adaptive family allocator manage the short budget. Individual candidate failures
are retained in SQLite without ending the run. Interrupted runs resume from their
persisted state without duplicating completed trials.

The selected PipelineSpec is rebuilt, fitted on all training rows, and saved as a
locally generated Joblib artifact. The system checks its registered size and
SHA-256 before loading it, replays the target-free test CSV, and compares reproduced
predictions with the saved predictions. A standalone HTML report presents the real
profile, validation, exclusions, leaderboard, failures, budget, selected spec, and
artifact links.

The one-command demo covers classification with resume, regression, and a synthetic
leakage attack in about 20 seconds on the audited machine.

## How we built it

The Python 3.12 project uses versioned Pydantic contracts to separate persistent
decisions from executable objects. pandas provides mature CSV/scikit-learn
interoperability; scikit-learn Pipelines keep imputation and encoding fold-local.
`PipelineSpec` is the reconstructible source of truth, while small factories build
the executable pipelines.

Optuna provides persistent ask/tell studies. The project adds its own fidelity,
promotion, budget-reserve, and family-allocation logic. SQLite stores manifests,
trials, attempts, leases, checkpoints, and artifact registrations. Typer and Rich
present the CLI; Jinja2 renders a single-file report from persisted state. `uv`
locks the environment, and pytest, Ruff, and Pyright enforce the release gates.

## Challenges

- Preventing every learned transformation from seeing validation rows.
- Orienting losses for optimization while showing positive RMSE/log-loss to users.
- Containing individual model timeouts without losing the full run.
- Reconciling interrupted Optuna work and fidelity promotions exactly once.
- Reserving enough time for confirmation, final fitting, and artifact verification
  under six-second demonstration budgets.
- Reducing safe worker-process import overhead only after measuring it.
- Making a report from persisted records so presentation cannot diverge from the
  actual experiment.
- Keeping the Build Week scope honest by deferring unverified notebook and
  cross-dataset-memory work.

## Accomplishments

- A complete CSV-to-diagnosis-to-model-to-report local workflow.
- Real interruption/resume with prior trial IDs and budget consumption preserved.
- Multi-family, multi-fidelity search with a mandatory naive baseline.
- Explainable neutralization of exact target copies, post-outcome copies, and
  identifiers before search.
- A reconstructible best PipelineSpec and checksum-gated Joblib load.
- Persisted predictions that are replayed and compared during artifact validation.
- Classification, regression, and leakage demos in one command.
- A standalone report and comprehensive automated quality gates.

## What we learned

- Serializable decisions are easier to test, resume, and explain than estimator
  objects used as state.
- Short-budget AutoML needs explicit phase reserves or it may search successfully
  but fail to deliver a usable final artifact.
- Process startup is part of the budget and must be measured.
- A controlled failed trial is useful evidence when the system proves it continued.
- Honest scope cuts improve reliability more than decorative unfinished features.

## What's next

- Generate and execute a certified notebook from the final manifest.
- Add M11 cross-run dataset memory and warm-start ranking.
- Run M14 extended comparisons with other AutoML frameworks outside the fast gate.
- Expand calibrated-probability and temporal diagnostics.
- Consider a web experience only after the local evidence path remains verifiable.

These are future items, not features claimed by this release.

## Technologies used

Python 3.12, uv, pandas, NumPy, scikit-learn, Joblib, Optuna, Pydantic v2,
SQLite, Typer, Rich, Jinja2, psutil, pytest, Ruff, and Pyright. nbformat and nbclient
remain dependencies from the original technical baseline, but certified notebook
generation is explicitly not delivered.

## Category justification

**Developer Tools.** Autonomous AutoML is a local engineering tool for developers
and data practitioners. It automates a repeatable ML workflow while adding testing,
artifact-integrity, failure-recovery, and leakage-safety evidence. Judges can install
and exercise it directly with two commands and committed sample data.

## Codex and GPT-5.6 usage

Codex served as the implementation and release-audit agent. It read the supplied
specifications, implemented and integrated the M0–M9 engine, wrote regression tests,
generated deterministic examples, ran real short-budget demos, diagnosed process
and resume failures, adversarially tested artifact corruption, and synchronized the
CLI, report, and documentation with measured behavior.

The available transcript and repository metadata do not identify a separate,
verifiable GPT-5.6 Sol session. We do not attribute unrecorded work to that model.
Before submission, the project owner must ensure that any required GPT-5.6 use is
real, documented, and described consistently in the video and Devpost form. The
evidence-based chronology is in `docs/CODEX_COLLABORATION.md`.

## Installation instructions

```bash
git clone <repository-url>
cd AutoML
uv sync --frozen
uv run automl demo
```

Requirements are Git, `uv`, 2 CPU cores, 2 GiB RAM, about 1 GiB free disk, and
internet access for the first sync. No API key, `.env`, GPU, Docker, or external
service is required at runtime.

## Known limitations

- Tabular CSV only; no web UI, NLP, vision, or distributed execution.
- Certified notebook generation is deferred; reproducibility is proven through the
  manifest, PipelineSpec, registered Joblib, and prediction replay.
- M11 inter-dataset memory and M14 extended benchmarks are deferred.
- Short wall budgets are soft around native operations already in progress.
- Fixed seeds do not eliminate every cross-machine floating-point or timing
  difference near a short budget boundary.
- Joblib artifacts are intended only for locally generated, trusted run directories.

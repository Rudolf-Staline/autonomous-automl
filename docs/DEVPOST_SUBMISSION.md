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
or `auto`, and a search-launch budget. It profiles only the training data, records
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

Every successful run also produces a Self-verified trust certificate in JSON and
standalone HTML. Its status is derived from recorded completion, artifact integrity,
prediction replay, and unresolved critical findings. A factual “Why this pipeline
won” section mirrors the real selection profile instead of inventing a narrative.
Search, finalization, and total runtime are persisted separately.

An optional Observed Trust Gap runs only after final selection. It evaluates the
selected algorithm and parameters on the persisted folds while restoring recorded
risk columns when that comparison is valid. It saves its protocol and predictions
under a separate namespace and never changes Optuna, the main trial registry,
leaderboard, or selected PipelineSpec. A positive value means the raw protocol
appeared better for the recorded metric; it is not a universal causal leakage
penalty.

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
present the CLI; Jinja2 renders standalone report and certificate files from
persisted state. `uv` locks the environment, and pytest, Ruff, and Pyright enforce
the release gates.

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
- Producing a useful certificate without circularly claiming that a file certifies
  its own bytes.
- Reusing exact persisted fold execution seeds after the raw PipelineSpec changes,
  so the diagnostic comparison does not add avoidable random variation.
- Keeping the Trust Gap outside the scheduler, main budget, trial registry,
  leaderboard, and selection while enforcing a separate timeout.
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
- Deterministic JSON/HTML Self-verified trust certificates with artifact replay,
  selection provenance, runtime telemetry, warnings, and explicit limits.
- An audited synthetic leakage run with raw diagnostic score 1.000000, verified ROC
  AUC 0.837941, and Observed Trust Gap +0.162059.

## What we learned

- Serializable decisions are easier to test, resume, and explain than estimator
  objects used as state.
- Short-budget AutoML needs explicit phase reserves or it may search successfully
  but fail to deliver a usable final artifact.
- Process startup is part of the budget and must be measured.
- A controlled failed trial is useful evidence when the system proves it continued.
- Honest scope cuts improve reliability more than decorative unfinished features.
- A diagnostic comparison is trustworthy only when its metric, folds, fidelity,
  execution seeds, features, and limitations are persisted.
- A self-generated engineering certificate must describe its limits and must not be
  marketed as an external assessment.

## What's next

- Generate and execute a certified notebook from the final manifest.
- Add M11 cross-run dataset memory and warm-start ranking.
- Run M14 extended comparisons with other AutoML frameworks outside the fast gate.
- Expand calibrated-probability and temporal diagnostics.
- Consider a web experience only after the local evidence path remains verifiable.
- Evaluate additional explicitly documented diagnostic protocols only if their
  comparability can be preserved.

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

This was a solo Build Week project assisted by Codex. Codex served as the
implementation and release-audit agent. It read the supplied
specifications, implemented and integrated the M0–M9 engine, wrote regression tests,
generated deterministic examples, ran real short-budget demos, diagnosed process
and resume failures, adversarially tested artifact corruption, and synchronized the
CLI, report, and documentation with measured behavior.

Codex later implemented the isolated Trust Layer on `feat/trust-layer`: it audited
the evidence already persisted, preserved the selector and scheduler, added
versioned optional contracts, found and fixed the diagnostic identifier barrier,
reused persisted execution seeds, added corruption/compatibility/isolation tests,
and measured the complete demo and diagnostic cost. These statements are supported
by the working transcript, branch diff, tests, and `IMPLEMENTATION_STATUS.md`.

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
- `budget_seconds` is a search-launch budget. Finalization and a native operation
  already running may extend total wall-clock runtime.
- The Self-verified trust certificate is generated by this engine. It is not an
  external, regulatory, security, scientific, ethical, or commercial certification.
- Observed Trust Gap is disabled by default, may be `NOT_COMPUTED`, and is a
  protocol-specific observation rather than a proven leakage impact.
- Fixed seeds do not eliminate every cross-machine floating-point or timing
  difference near a short budget boundary.
- Joblib artifacts are intended only for locally generated, trusted run directories.

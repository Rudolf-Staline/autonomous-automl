# Codex collaboration record

This file records only contributions supported by the working transcript and the
repository's implementation journal. It does not infer authorship from names or
invent a separate human/AI role.

## Context received

The repository began with specification documents and permanent `AGENTS.md` rules.
Those documents defined the priorities, contracts, architecture, milestone order,
acceptance tests and benchmark plan. Their authorship is not established by the
available evidence, so it is described only as supplied project context.

## Evidence-based chronology

### Specification intake and architecture

Codex read the required documents in their mandated order, reformulated the design
in `IMPLEMENTATION_STATUS.md`, identified leakage/reproducibility risks and planned
the dependency milestones. The resulting architecture separates serializable
contracts from executable scikit-learn objects and keeps `PipelineSpec` as the
pipeline source of truth.

### M0–M3: foundation and data safety

Codex created the Python 3.12/`uv` project, strict Pydantic contracts, CSV loader,
train-only profiler, leakage findings and persisted validation plans. It added
tests for target isolation, test firewalls, identifiers, copies of target,
group/time constraints and reproducible folds, and ran the quality gates after
each phase.

### M4–M6: executable pipelines and durable runs

Codex implemented compatible scikit-learn preprocessing/model factories, metric and
OOF evaluation, per-trial timeout/error containment, SQLite migrations, atomic
artifacts, hashes, leases and resume fencing. It deliberately excluded target
encoding until a cross-fitted implementation could be proven and isolated optional
boosting libraries behind lazy adapters.

### M7–M9: autonomous search

Codex connected persistent Optuna ask/tell studies, four fidelity levels,
successive promotions, a cost-aware family allocator and a crash-safe search
controller. Tests covered deterministic suggestions, failed/pruned trials, budget
reserves, retries and exact state restoration.

### Build Week delivery pivot

When the priority changed, Codex audited the actual repository instead of
reimplementing working components. It documented what was complete, identified the
missing public orchestration/CLI/report/examples, froze M11 and M14, and treated the
notebook as deferred unless it could be certified.

Codex then:

- connected `AutoMLRun.fit` and `resume` through data, profile, leakage, validation,
  search, confirmation, final selection, Joblib and prediction replay;
- added progress callbacks and structured JSON Lines events;
- generated a persisted-state-only standalone HTML report;
- implemented Typer/Rich commands for fit, resume, inspect, leaderboard, predict,
  artifact validation and the one-command demo;
- created deterministic classification, regression and leakage datasets/configs;
- executed real runs and found that safe `forkserver` startup consumed most of a
  six-second demo budget;
- fixed that measured bottleneck by preloading the fold worker module, then reran
  the demo successfully; the later RC audit measured 19.27 seconds;
- added public-path integration tests, installation guidance and submission docs.

### Quality and correction loop

The journal records repeated pytest, Ruff, formatter and Pyright runs. During the
delivery work Codex corrected type errors, short-budget behavior, resume
reconciliation and CLI/report issues based on actual command output. Failed model
trials observed during demos were retained rather than hidden.

### Adversarial release-candidate audit

Codex then switched from implementation to a strict feature-frozen audit. It found
that the working implementation had not yet been added to Git, which meant a real
clone would contain only the original specifications. It also scanned the release
tree for credentials, personal paths, caches, databases, model files and oversized
artifacts; exercised a deliberate resume; corrupted a copied Joblib artifact and
confirmed rejection before loading; rebuilt the selected pipeline from its
persisted `PipelineSpec`; and compared replayed predictions.

The same audit measured the three-scenario CLI, corrected truncated/overly verbose
release presentation, made post-neutralization scoring explicit in the report,
removed documentation overclaims, and prepared the repository for a commit-backed
fresh-clone gate. No M11, M14, notebook, or web feature was added during this work.

## Decisions accelerated or improved by Codex

- Reuse the tested M0–M9 services and add a thin public orchestration layer.
- Freeze nonessential memory/benchmark milestones before the deadline.
- Make reports read SQLite/manifests instead of live Python objects.
- Verify the serialized model by loading through its checksum registration and
  comparing predictions.
- Demonstrate resume with a deliberate checkpoint rather than claiming it from unit
  tests alone.
- Keep Rich confined to the CLI so core contracts and logging remain stable.
- Preload scientific modules in the safe fold process strategy after measuring the
  actual startup bottleneck.
- State that the notebook is deferred instead of creating an example-like notebook
  that would violate the project's reproducibility claim.

## GPT-5.6 Sol disclosure

The Build Week request explicitly asks how Codex and GPT-5.6 Sol were used. The
available system context identifies the implementation agent as Codex, and neither
the transcript nor repository metadata provides a separable GPT-5.6 Sol session or
its outputs. Consequently:

- Codex contributions are recorded above;
- no work is attributed to GPT-5.6 Sol without evidence;
- no prompt, model run, review or decision is fabricated to fill that section.

If verifiable GPT-5.6 Sol work is later added, this chronology should be extended
with the date, task, outputs and validation performed.

## Human contribution disclosure

The project direction, specifications and Build Week pivot were provided to Codex.
The available record does not name or distinguish individual human contributors,
so this document does not assign them unverified implementation work.

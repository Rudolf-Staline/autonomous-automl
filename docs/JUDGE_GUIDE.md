# Judge guide — under five minutes

## Prerequisites

- a clone of this repository;
- Git and `uv`;
- internet access for the first dependency sync;
- Python 3.12 is provisioned by `uv` from `.python-version`;
- no API key, `.env`, GPU, Docker daemon, or external service.

## Two commands

Run from the repository root:

```bash
uv sync --frozen
uv run automl demo
```

The first sync is network-dependent and commonly takes 1–3 minutes. The audited
demo took 19.27 seconds on Linux and should normally take 20–30 seconds on a recent
laptop.

## What should be visible

- classification, regression, and synthetic-leakage scenarios;
- a classification checkpoint followed by SQLite resume;
- current stage, budget, successful/failed trial counts, and search objective;
- a final leaderboard and three artifact-validation `PASS` results;
- `target_copy`, `approved_after_review`, and `customer_id` neutralized before
  search;
- exact run and report paths, ending with `All scenarios passed.`

The demo writes to a timestamped directory under `runs/`. Open the leakage report
printed by the CLI, or open:

```text
<demo-root>/leakage/report.html
```

It is a standalone file with inline styling and relative artifact links.

## Optional checks

Replace `<demo-root>` with the path printed by the demo:

```bash
uv run automl inspect <demo-root>/leakage
uv run automl leaderboard <demo-root>/classification-resumed --limit 5
uv run automl validate-artifacts <demo-root>/classification-resumed
```

On Windows, if process startup exhausts a six-second scenario budget, use
`uv run automl demo --budget 15s`.

## Hardware limits

Recommended minimum: 2 CPU cores, 2 GiB RAM, and about 1 GiB free disk. The audited
demo used one model worker and peaked near 225 MiB RSS, excluding the installed
environment. Certified notebook export, M11, M14, and a web UI are not part of this
release.

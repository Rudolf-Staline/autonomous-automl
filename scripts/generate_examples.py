"""Generate the small deterministic datasets used by the Build Week demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
SEED = 42


def _write_frame(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(EXAMPLES / name, index=False, lineterminator="\n")


def _write_json(name: str, value: dict[str, Any]) -> None:
    (EXAMPLES / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _features(rng: np.random.Generator, rows: int, *, start: int) -> pd.DataFrame:
    age = rng.integers(19, 76, size=rows).astype(float)
    income = rng.lognormal(mean=10.45, sigma=0.42, size=rows)
    tenure = rng.integers(0, 18, size=rows).astype(float)
    region = rng.choice(["north", "south", "east", "west"], size=rows)
    channel = rng.choice(["web", "store", "partner"], size=rows, p=[0.55, 0.3, 0.15])
    age[rng.random(rows) < 0.08] = np.nan
    income[rng.random(rows) < 0.07] = np.nan
    region = region.astype(object)
    region[rng.random(rows) < 0.05] = None
    return pd.DataFrame(
        {
            "row_id": [f"row-{index:04d}" for index in range(start, start + rows)],
            "age": age,
            "income": income,
            "tenure_years": tenure,
            "region": region,
            "channel": channel,
        }
    )


def _classification(rng: np.random.Generator) -> None:
    train = _features(rng, 180, start=0)
    score = (
        0.055 * np.nan_to_num(train["age"].to_numpy(), nan=45.0)
        + 0.000045 * np.nan_to_num(train["income"].to_numpy(), nan=35_000.0)
        + 0.12 * train["tenure_years"].to_numpy()
        + 0.8 * (train["channel"].to_numpy() == "web")
        + rng.normal(0.0, 1.0, len(train))
    )
    target = (score > np.median(score)).astype(int)
    train["churned"] = target
    test = _features(rng, 36, start=10_000)
    test.loc[0, "region"] = "central"
    _write_frame("classification_train.csv", train)
    _write_frame("classification_test.csv", test)
    _write_json(
        "classification_config.json",
        {
            "target": "churned",
            "task": "auto",
            "metric": "auto",
            "budget_seconds": 6,
            "random_seed": SEED,
            "n_jobs": 1,
            "id_column": "row_id",
            "test_path": "examples/classification_test.csv",
            "output_dir": "runs/demo-classification",
            "trial_timeout_seconds": 3,
        },
    )


def _regression(rng: np.random.Generator) -> None:
    train = _features(rng, 180, start=20_000)
    clean_income = np.nan_to_num(train["income"].to_numpy(), nan=35_000.0)
    clean_age = np.nan_to_num(train["age"].to_numpy(), nan=45.0)
    region_effect = pd.Series(train["region"]).map(
        {"north": 12.0, "south": -8.0, "east": 5.0, "west": 0.0}
    )
    target = (
        35.0
        + 0.0015 * clean_income
        + 0.7 * clean_age
        + 2.4 * train["tenure_years"].to_numpy()
        + region_effect.fillna(0.0).to_numpy()
        + rng.normal(0.0, 7.0, len(train))
    )
    train["annual_spend"] = target
    test = _features(rng, 36, start=30_000)
    test.loc[0, "channel"] = "mobile"
    _write_frame("regression_train.csv", train)
    _write_frame("regression_test.csv", test)
    _write_json(
        "regression_config.json",
        {
            "target": "annual_spend",
            "task": "regression",
            "metric": "rmse",
            "budget_seconds": 6,
            "random_seed": SEED,
            "n_jobs": 1,
            "id_column": "row_id",
            "test_path": "examples/regression_test.csv",
            "output_dir": "runs/demo-regression",
            "trial_timeout_seconds": 3,
        },
    )


def _leakage(rng: np.random.Generator) -> None:
    train = _features(rng, 160, start=40_000).drop(columns=["row_id"])
    score = (
        0.04 * np.nan_to_num(train["age"].to_numpy(), nan=45.0)
        + 0.15 * train["tenure_years"].to_numpy()
        + rng.normal(0.0, 1.0, len(train))
    )
    target = (score > np.median(score)).astype(int)
    train.insert(0, "customer_id", [f"customer-{index:04d}" for index in range(len(train))])
    train["target_copy"] = target
    train["approved_after_review"] = target
    train["outcome"] = target
    _write_frame("leakage_train.csv", train)
    _write_json(
        "leakage_config.json",
        {
            "target": "outcome",
            "task": "binary_classification",
            "metric": "roc_auc",
            "budget_seconds": 6,
            "random_seed": SEED,
            "n_jobs": 1,
            "output_dir": "runs/demo-leakage",
            "trial_timeout_seconds": 3,
        },
    )


def main() -> None:
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    _classification(rng)
    _regression(rng)
    _leakage(rng)


if __name__ == "__main__":
    main()

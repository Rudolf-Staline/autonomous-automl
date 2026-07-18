"""Deterministic finalist selection tests using synthetic TrialResult values."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from autonomous_automl.contracts import FidelitySpec, MetricName, PipelineSpec, TrialResult
from autonomous_automl.evaluation.selection import (
    build_leaderboard,
    pareto_front,
    select_finalist,
)


def _trial(
    trial_id: str,
    score: float,
    *,
    std: float = 0.01,
    level: int = 2,
    fit: float = 1.0,
    predict: float = 0.1,
    memory: float | None = 64.0,
    family: str = "linear",
    model: str = "ridge",
    status: str = "completed",
    metric: str = "rmse",
    spec_key: str | None = None,
) -> TrialResult:
    spec = PipelineSpec(
        family=family,
        task="regression",
        numeric_imputer="median",
        numeric_scaler="standard",
        categorical_imputer="constant",
        categorical_encoder="ordinal",
        datetime_transformer="calendar",
        model_name=model,
        model_params={"synthetic_key": spec_key if spec_key is not None else trial_id},
        random_seed=42,
    )
    fidelity = FidelitySpec(
        level=level,
        sample_fraction=1.0 if level >= 2 else 0.5,
        n_folds=5 if level >= 2 else 2,
        max_iterations=100,
        seeds=[42, 43] if level == 3 else [42],
    )
    now = datetime.now(UTC)
    values: dict[str, object] = {
        "trial_id": trial_id,
        "family": family,
        "pipeline_spec": spec,
        "fidelity": fidelity,
        "status": status,
        "primary_metric": metric,
        "fit_seconds": fit,
        "predict_seconds": predict,
        "peak_memory_mb": memory,
        "started_at": now,
        "finished_at": now,
    }
    if status == "completed":
        values.update(
            {
                "mean_score": score,
                "std_score": std,
                "fold_scores": [score - std, score + std],
            }
        )
    elif status == "failed":
        values.update(
            {
                "failure_type": "SyntheticFailure",
                "failure_message": "aggregate synthetic failure",
            }
        )
    return TrialResult.model_validate(values)


def test_leaderboard_keeps_internal_greater_is_better_orientation() -> None:
    trials = [
        _trial("worse-rmse", -2.0),
        _trial("better-rmse", -1.0),
        _trial("failed", 100.0, status="failed"),
    ]

    leaderboard = build_leaderboard(trials)

    assert [entry.trial_id for entry in leaderboard.entries] == ["better-rmse", "worse-rmse"]
    assert leaderboard.entries[0].rank == 1
    assert list(leaderboard.to_frame()["trial_id"]) == ["better-rmse", "worse-rmse"]


def test_low_fidelity_trials_are_visible_but_never_finalists() -> None:
    low = _trial("low", 0.99, level=0)
    full = _trial("full", 0.70, level=2)
    leaderboard = build_leaderboard([low, full])

    selection = select_finalist(leaderboard, "accuracy")

    assert [entry.trial_id for entry in leaderboard.entries] == ["low", "full"]
    assert [entry.trial_id for entry in leaderboard.finalists] == ["full"]
    assert selection.selected.trial_id == "full"


def test_confirmation_supersedes_full_result_for_the_same_pipeline() -> None:
    obsolete_full = _trial("obsolete-full", 0.99, level=2, spec_key="same")
    confirmation_z = _trial("z-confirmation", 0.70, level=3, spec_key="same")
    confirmation_a = _trial("a-confirmation", 0.70, level=3, spec_key="same")

    forward = build_leaderboard([obsolete_full, confirmation_z, confirmation_a])
    reverse = build_leaderboard([confirmation_a, confirmation_z, obsolete_full])

    assert [entry.trial_id for entry in forward.finalists] == ["a-confirmation"]
    assert forward.finalists == reverse.finalists
    assert select_finalist(forward, "accuracy").selected.trial_id == "a-confirmation"


def test_leaderboard_requires_one_completed_metric_and_exposes_it() -> None:
    clean = build_leaderboard([_trial("first", 0.7), _trial("second", 0.8)])

    assert clean.primary_metric is MetricName.RMSE
    assert set(clean.to_frame()["primary_metric"]) == {MetricName.RMSE.value}

    mixed = [_trial("rmse", -1.0), _trial("mae", -0.9, metric="mae")]
    with pytest.raises(ValueError, match="one primary_metric"):
        build_leaderboard(mixed)


def test_pareto_front_maximizes_score_and_minimizes_resources() -> None:
    high_score = _trial("high", 0.90, fit=10.0, predict=1.0, memory=100.0)
    efficient = _trial("efficient", 0.80, fit=2.0, predict=0.2, memory=50.0)
    dominated = _trial("dominated", 0.70, fit=5.0, predict=0.5, memory=80.0)
    leaderboard = build_leaderboard([dominated, efficient, high_score])

    front = pareto_front(leaderboard.finalists)

    assert [entry.trial_id for entry in front] == ["high", "efficient"]


def test_optimization_profiles_make_coherent_tradeoffs() -> None:
    accurate = _trial("accurate", 0.95, fit=100.0, predict=10.0, memory=500.0)
    balanced = _trial("balanced", 0.88, fit=10.0, predict=1.0, memory=100.0)
    fast = _trial("fast", 0.75, fit=1.0, predict=0.1, memory=50.0)
    leaderboard = build_leaderboard([fast, accurate, balanced])

    assert select_finalist(leaderboard, "accuracy").selected.trial_id == "accurate"
    assert select_finalist(leaderboard, "balanced").selected.trial_id == "balanced"
    assert select_finalist(leaderboard, "fast").selected.trial_id == "fast"


def test_accuracy_profile_penalizes_uncertainty() -> None:
    risky = _trial("risky", 1.0, std=0.30)
    stable = _trial("stable", 0.90, std=0.01)

    selection = select_finalist(build_leaderboard([risky, stable]), "accuracy")

    assert selection.selected.trial_id == "stable"
    assert selection.selected.adjusted_score == pytest.approx(0.89)


def test_baseline_is_identifiable_and_compared_only_at_full_fidelity() -> None:
    baseline = _trial("dummy-full", 0.50, family="baseline", model="dummy")
    candidate = _trial("model-full", 0.70)

    selection = select_finalist(build_leaderboard([baseline, candidate]), "balanced")

    assert selection.baseline is not None
    assert selection.baseline.is_baseline
    assert selection.baseline_comparable
    assert selection.baseline_score_delta == pytest.approx(0.20)
    assert selection.beats_baseline is True

    low_baseline = _trial("dummy-low", 0.60, level=0, family="baseline", model="dummy")
    without_comparable = select_finalist(
        build_leaderboard([low_baseline, candidate]),
        "accuracy",
    )
    assert not without_comparable.baseline_comparable
    assert without_comparable.baseline_score_delta is None
    assert without_comparable.beats_baseline is None


def test_input_order_and_exact_ties_have_stable_trial_id_breaks() -> None:
    first = _trial("a-trial", 0.80, std=0.02, fit=2.0)
    second = _trial("b-trial", 0.80, std=0.02, fit=2.0)

    forward = build_leaderboard([first, second])
    reverse = build_leaderboard([second, first])

    assert [entry.trial_id for entry in forward.entries] == ["a-trial", "b-trial"]
    assert forward.entries == reverse.entries
    assert select_finalist(forward, "accuracy").selected.trial_id == "a-trial"


def test_selection_requires_full_fidelity_and_duplicate_ids_are_rejected() -> None:
    low = _trial("low-only", 0.8, level=1)
    with pytest.raises(ValueError, match="full-fidelity"):
        select_finalist(build_leaderboard([low]), "accuracy")

    duplicate = _trial("duplicate", 0.7)
    with pytest.raises(ValueError, match="duplicate completed trial id"):
        build_leaderboard([duplicate, duplicate])

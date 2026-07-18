"""The explanation must mirror, never reinterpret, finalist selection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from autonomous_automl.contracts import FidelitySpec, FoldResult, PipelineSpec, TrialResult
from autonomous_automl.evaluation import explain_pipeline_selection
from autonomous_automl.evaluation.selection import build_leaderboard, select_finalist

STARTED = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
FINISHED = STARTED + timedelta(seconds=1)


def _trial(
    trial_id: str,
    score: float,
    *,
    std: float = 0.01,
    fit: float = 1.0,
    predict: float = 0.1,
    memory: float | None = 32.0,
    status: str = "completed",
) -> TrialResult:
    spec = PipelineSpec(
        family="linear",
        task="binary_classification",
        numeric_imputer="median",
        numeric_scaler="standard",
        categorical_imputer="constant",
        categorical_encoder="one_hot",
        datetime_transformer="drop",
        model_name="logistic_regression",
        model_params={"C": 1.0, "diagnostic_key": trial_id},
        random_seed=42,
    )
    fidelity = FidelitySpec(
        level=2,
        sample_fraction=1.0,
        n_folds=2,
        max_iterations=100,
        seeds=[42],
    )
    values: dict[str, object] = {
        "trial_id": trial_id,
        "family": "linear",
        "pipeline_spec": spec,
        "fidelity": fidelity,
        "status": status,
        "primary_metric": "roc_auc",
        "fit_seconds": fit,
        "predict_seconds": predict,
        "peak_memory_mb": memory,
        "started_at": STARTED,
        "finished_at": FINISHED,
    }
    if status == "completed":
        fold_values = [score - std, score + std]
        values.update(
            mean_score=score,
            std_score=std,
            fold_scores=fold_values,
            fold_results=[
                FoldResult(
                    fold_index=index,
                    seed=100 + index,
                    score=value,
                    user_metric_value=value,
                    fit_seconds=fit / 2,
                    predict_seconds=predict / 2,
                    n_train_rows=50,
                    n_validation_rows=25,
                )
                for index, value in enumerate(fold_values)
            ],
        )
    else:
        values.update(
            failure_type="SyntheticFailure",
            failure_message="recorded unit-test failure",
        )
    return TrialResult.model_validate(values)


def test_accuracy_reason_states_the_real_uncertainty_adjusted_rule() -> None:
    risky = _trial("risky", 0.95, std=0.20)
    selected = _trial("selected", 0.88, std=0.01)
    leaderboard = build_leaderboard([risky, selected])

    explanation = explain_pipeline_selection(
        leaderboard,
        [risky, selected],
        "accuracy",
        selected_trial_id="selected",
    )

    assert explanation.selection_reason.startswith("Highest uncertainty-adjusted")
    assert explanation.eligible_trial_count == 2
    assert "faster" not in explanation.selection_reason.casefold()
    assert "simpler" not in explanation.selection_reason.casefold()
    assert "more stable" not in explanation.selection_reason.casefold()


def test_exact_tie_reports_the_existing_deterministic_tie_breaker() -> None:
    first = _trial("a-trial", 0.80, std=0.02, fit=2.0)
    second = _trial("b-trial", 0.80, std=0.02, fit=2.0)
    leaderboard = build_leaderboard([second, first])

    explanation = explain_pipeline_selection(
        leaderboard,
        [second, first],
        "accuracy",
        selected_trial_id="a-trial",
    )

    assert explanation.tie_breaker_used
    assert explanation.tie_breaker_detail is not None
    assert explanation.runner_up_trial_id == "b-trial"
    assert "trial id" in explanation.tie_breaker_detail


def test_runner_up_margin_and_every_context_provenance_are_recorded() -> None:
    winner = _trial("winner", 0.84)
    runner_up = _trial("runner-up", 0.81)
    failed = _trial("failed", 0.0, status="failed")

    explanation = explain_pipeline_selection(
        build_leaderboard([winner, runner_up, failed]),
        [winner, runner_up, failed],
        "accuracy",
        selected_trial_id="winner",
        pipeline_size_bytes=2048,
    )
    context = {item.label: item for item in explanation.supporting_context}

    assert explanation.runner_up_trial_id == "runner-up"
    assert context["Absolute verified metric difference from runner-up"].value == pytest.approx(
        0.03
    )
    assert context["Failed trials in selected family"].value == 1
    assert context["Serialized pipeline size bytes"].value == 2048
    assert all(item.provenance for item in explanation.supporting_context)


def test_missing_optional_context_is_omitted_not_estimated() -> None:
    only = _trial("only", 0.80, memory=None)

    explanation = explain_pipeline_selection(
        build_leaderboard([only]),
        [only],
        "accuracy",
        selected_trial_id="only",
        pipeline_size_bytes=None,
    )
    labels = {item.label for item in explanation.supporting_context}

    assert explanation.runner_up_trial_id is None
    assert "Serialized pipeline size bytes" not in labels
    assert "Absolute verified metric difference from runner-up" not in labels


@pytest.mark.parametrize("profile", ["balanced", "fast"])
def test_profile_explanation_uses_the_same_selected_trial_as_production(profile: str) -> None:
    accurate = _trial("accurate", 0.95, fit=20, predict=2, memory=200)
    efficient = _trial("efficient", 0.82, fit=1, predict=0.1, memory=30)
    leaderboard = build_leaderboard([accurate, efficient])
    selected = select_finalist(leaderboard, profile).selected

    explanation = explain_pipeline_selection(
        leaderboard,
        [accurate, efficient],
        profile,
        selected_trial_id=selected.trial_id,
    )

    assert explanation.selected_trial_id == selected.trial_id
    assert profile in explanation.selection_reason.casefold()


def test_explanation_refuses_a_trial_not_selected_by_the_real_rule() -> None:
    winner = _trial("winner", 0.90)
    loser = _trial("loser", 0.70)

    with pytest.raises(ValueError, match="differs from the existing selection rule"):
        explain_pipeline_selection(
            build_leaderboard([winner, loser]),
            [winner, loser],
            "accuracy",
            selected_trial_id="loser",
        )

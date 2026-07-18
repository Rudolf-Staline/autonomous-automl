"""Deterministic, training-free family allocation tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from autonomous_automl.contracts import FidelitySpec, PipelineSpec, TrialResult
from autonomous_automl.search.allocator import FamilyAllocator
from autonomous_automl.utils.errors import ResumeError
from autonomous_automl.utils.json import JsonValue, canonical_json_dumps


def _result(
    trial_id: str,
    family: str,
    *,
    score: float = 0.5,
    cost: float = 1.0,
    status: str = "completed",
    model_name: str | None = None,
) -> TrialResult:
    spec = PipelineSpec(
        family=family,
        task="regression",
        numeric_imputer="median",
        numeric_scaler="none",
        categorical_imputer="constant",
        categorical_encoder="ordinal",
        datetime_transformer="calendar",
        model_name=model_name or f"{family}_model",
        model_params={},
        random_seed=42,
    )
    fidelity = FidelitySpec(
        level=0,
        sample_fraction=0.5,
        n_folds=2,
        max_iterations=50,
        seeds=[42],
    )
    now = datetime.now(UTC)
    values: dict[str, object] = {
        "trial_id": trial_id,
        "family": family,
        "pipeline_spec": spec,
        "fidelity": fidelity,
        "status": status,
        "primary_metric": "rmse",
        "fit_seconds": cost,
        "started_at": now,
        "finished_at": now,
    }
    if status == "completed":
        values.update(
            {
                "mean_score": score,
                "std_score": 0.01,
                "fold_scores": [score - 0.01, score + 0.01],
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


def test_minimum_exploration_selects_every_compatible_family_before_repeat() -> None:
    allocator = FamilyAllocator(["baseline", "linear", "trees"])

    selected = [allocator.select_next() for _ in range(3)]

    assert selected == ["baseline", "linear", "trees"]
    assert all(allocator.state_for(name).pending == 1 for name in allocator.families)


def test_cancel_pending_releases_only_an_active_selection() -> None:
    allocator = FamilyAllocator(["linear"])
    assert allocator.select_next() == "linear"

    allocator.cancel_pending("linear")

    assert allocator.state_for("linear").pending == 0
    with pytest.raises(RuntimeError, match="no pending"):
        allocator.cancel_pending("linear")


def test_ucb_prioritizes_better_performance_without_real_training() -> None:
    allocator = FamilyAllocator(
        ["strong", "weak"],
        improvement_weight=0.0,
        cost_weight=0.0,
        uncertainty_weight=0.0,
        diversity_weight=0.0,
    )
    allocator.update(_result("strong-1", "strong", score=0.9))
    allocator.update(_result("weak-1", "weak", score=0.2))

    assert allocator.select_next() == "strong"
    assert allocator.priorities()["strong"] > allocator.priorities()["weak"]


def test_cost_penalty_prefers_equally_good_cheaper_family() -> None:
    allocator = FamilyAllocator(
        ["fast", "slow"],
        performance_weight=0.0,
        improvement_weight=0.0,
        cost_weight=1.0,
        uncertainty_weight=0.0,
        diversity_weight=0.0,
    )
    allocator.update(_result("fast-1", "fast", score=0.8, cost=1.0))
    allocator.update(_result("slow-1", "slow", score=0.8, cost=100.0))

    assert allocator.select_next() == "fast"


def test_improvement_and_uncertainty_are_real_priority_terms() -> None:
    improving = FamilyAllocator(
        ["rising", "flat"],
        performance_weight=0.0,
        improvement_weight=1.0,
        cost_weight=0.0,
        uncertainty_weight=0.0,
        diversity_weight=0.0,
    )
    improving.update(_result("rising-1", "rising", score=0.4))
    improving.update(_result("rising-2", "rising", score=0.8))
    improving.update(_result("flat-1", "flat", score=0.8))
    improving.update(_result("flat-2", "flat", score=0.8))
    assert improving.select_next() == "rising"

    uncertain = FamilyAllocator(
        ["rare", "known"],
        performance_weight=0.0,
        improvement_weight=0.0,
        cost_weight=0.0,
        uncertainty_weight=1.0,
        diversity_weight=0.0,
    )
    uncertain.update(_result("rare-1", "rare"))
    for index in range(4):
        uncertain.update(_result(f"known-{index}", "known"))
    assert uncertain.select_next() == "rare"


def test_diversity_bonus_avoids_immediate_family_repetition() -> None:
    allocator = FamilyAllocator(
        ["first", "second"],
        performance_weight=0.0,
        improvement_weight=0.0,
        cost_weight=0.0,
        uncertainty_weight=0.0,
        diversity_weight=1.0,
    )
    allocator.update(_result("first-1", "first"))
    allocator.update(_result("second-1", "second"))

    assert allocator.select_next() == "first"
    allocator.update(_result("first-2", "first"))
    assert allocator.select_next() == "second"


def test_repeated_failures_suspend_then_reactivate_family() -> None:
    allocator = FamilyAllocator(
        ["fragile", "healthy"],
        failure_threshold=2,
        cooldown_selections=2,
    )
    allocator.update(_result("fragile-1", "fragile", status="failed"))
    allocator.update(_result("fragile-2", "fragile", status="failed"))

    suspended = allocator.state_for("fragile")
    assert suspended.suspended_until_step == 3
    assert allocator.select_next() == "healthy"
    allocator.update(_result("healthy-1", "healthy"))
    assert allocator.select_next() == "healthy"
    allocator.update(_result("healthy-2", "healthy"))

    assert allocator.select_next(["fragile"]) == "fragile"
    assert allocator.state_for("fragile").suspended_until_step is None


def test_online_state_tracks_rewards_cost_variance_and_models() -> None:
    allocator = FamilyAllocator(["linear"])
    allocator.update(_result("linear-1", "linear", score=0.4, cost=2.0, model_name="ridge"))
    allocator.update(_result("linear-2", "linear", score=0.8, cost=4.0, model_name="elastic_net"))

    state = allocator.state_for("linear")
    assert state.attempts == state.successes == 2
    assert state.best_score == 0.8
    assert state.mean_score == pytest.approx(0.6)
    assert state.score_variance == pytest.approx(0.08)
    assert state.mean_cost_seconds == pytest.approx(3.0)
    assert state.models_seen == {"ridge", "elastic_net"}


def test_snapshot_restore_is_canonical_strict_and_preserves_next_choice() -> None:
    allocator = FamilyAllocator(["linear", "trees"], failure_threshold=2)
    allocator.update(_result("linear-1", "linear", score=0.7, cost=1.0))
    allocator.update(_result("trees-1", "trees", score=0.8, cost=3.0))
    first_choice = allocator.select_next()
    allocator.update(_result(f"{first_choice}-2", first_choice, score=0.75, cost=2.0))
    snapshot = allocator.snapshot()
    serialized = canonical_json_dumps(snapshot)

    restored = FamilyAllocator(["linear", "trees"], failure_threshold=2)
    restored.restore(snapshot)

    assert canonical_json_dumps(restored.snapshot()) == serialized
    assert restored.select_next() == allocator.select_next()

    corrupted: dict[str, JsonValue] = dict(snapshot)
    corrupted["unexpected"] = True
    with pytest.raises(ResumeError, match="fields"):
        FamilyAllocator(["linear", "trees"], failure_threshold=2).restore(corrupted)


def test_restore_rejects_configuration_or_family_mismatch() -> None:
    source = FamilyAllocator(["a", "b"], cost_weight=0.5)
    snapshot = source.snapshot()

    with pytest.raises(ResumeError, match="configuration"):
        FamilyAllocator(["a", "b"], cost_weight=0.1).restore(snapshot)
    with pytest.raises(ResumeError, match="families"):
        FamilyAllocator(["a", "c"], cost_weight=0.5).restore(snapshot)

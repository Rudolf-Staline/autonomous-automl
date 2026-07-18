"""M9 acceptance tests for deterministic adaptive family allocation."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta

from autonomous_automl.contracts import (
    FidelitySpec,
    MetricName,
    PipelineSpec,
    TaskType,
    TrialResult,
    TrialStatus,
)
from autonomous_automl.search.allocator import FamilyAllocator

STARTED_AT = datetime(2026, 7, 17, 11, 0, tzinfo=UTC)
FINISHED_AT = STARTED_AT + timedelta(milliseconds=10)
FIDELITY = FidelitySpec(
    level=0,
    sample_fraction=0.25,
    n_folds=2,
    max_iterations=25,
    seeds=[42],
)


def make_pipeline(family: str) -> PipelineSpec:
    return PipelineSpec(
        family=family,
        task=TaskType.BINARY_CLASSIFICATION,
        numeric_imputer="median",
        numeric_scaler="standard",
        categorical_imputer="constant",
        categorical_encoder="one_hot",
        datetime_transformer="drop",
        model_name=f"{family}_model",
        model_params={},
        random_seed=42,
    )


def make_result(
    family: str,
    index: int,
    *,
    score: float | None,
    cost_seconds: float,
    status: TrialStatus = TrialStatus.COMPLETED,
) -> TrialResult:
    values: dict[str, object] = {
        "trial_id": f"{family}-{index}",
        "family": family,
        "pipeline_spec": make_pipeline(family),
        "fidelity": FIDELITY,
        "status": status,
        "primary_metric": MetricName.ROC_AUC,
        "fit_seconds": cost_seconds * 0.8,
        "predict_seconds": cost_seconds * 0.2,
        "started_at": STARTED_AT,
        "finished_at": FINISHED_AT,
    }
    if status is TrialStatus.COMPLETED:
        assert score is not None
        values.update(
            mean_score=score,
            std_score=0.0,
            fold_scores=[score, score],
        )
    elif status is TrialStatus.FAILED:
        values.update(
            failure_type="SyntheticFailure",
            failure_message="deterministic allocator simulation",
        )
    return TrialResult.model_validate(values)


def select_and_update(
    allocator: FamilyAllocator,
    index: int,
    outcomes: dict[str, tuple[float, float]],
) -> str:
    family = allocator.select_next()
    score, cost = outcomes[family]
    allocator.update(make_result(family, index, score=score, cost_seconds=cost))
    return family


def test_j1_every_compatible_family_receives_minimum_exploration() -> None:
    families = ["linear", "random_forest", "hist_gradient_boosting"]
    allocator = FamilyAllocator(families, minimum_exploration_trials=1)
    selected: list[str] = []

    for index in range(len(families)):
        family = allocator.select_next()
        selected.append(family)
        allocator.update(make_result(family, index, score=0.7, cost_seconds=1.0))

    assert selected == families
    assert set(selected) == set(families)
    assert all(allocator.state_for(family).attempts == 1 for family in families)


def test_j2_promising_low_cost_family_receives_more_deterministic_selections() -> None:
    families = ["promising", "middle", "weak_expensive"]
    outcomes = {
        "promising": (0.90, 1.0),
        "middle": (0.72, 3.0),
        "weak_expensive": (0.55, 8.0),
    }

    def simulate() -> list[str]:
        allocator = FamilyAllocator(families)
        return [select_and_update(allocator, index, outcomes) for index in range(45)]

    first = simulate()
    second = simulate()
    counts = Counter(first)

    assert first == second
    assert counts["promising"] > counts["middle"]
    assert counts["promising"] > counts["weak_expensive"]
    assert set(counts) == set(families)


def test_j2_equal_reward_penalizes_the_more_expensive_family() -> None:
    allocator = FamilyAllocator(
        ["cheap", "expensive"],
        performance_weight=1.0,
        cost_weight=1.0,
        uncertainty_weight=0.0,
        diversity_weight=0.0,
    )
    outcomes = {
        "cheap": (0.8, 1.0),
        "expensive": (0.8, 20.0),
    }
    assert select_and_update(allocator, 0, outcomes) == "cheap"
    assert select_and_update(allocator, 1, outcomes) == "expensive"

    priorities = allocator.priorities()

    assert priorities["cheap"] > priorities["expensive"]
    assert allocator.select_next() == "cheap"


def test_j3_repeated_failures_suspend_then_cooldown_reactivates_family() -> None:
    allocator = FamilyAllocator(
        ["fragile", "stable"],
        failure_threshold=2,
        cooldown_selections=2,
    )

    assert allocator.select_next() == "fragile"
    allocator.update(
        make_result(
            "fragile",
            0,
            score=None,
            cost_seconds=0.1,
            status=TrialStatus.FAILED,
        )
    )
    assert allocator.select_next() == "stable"
    allocator.update(make_result("stable", 1, score=0.8, cost_seconds=1.0))

    assert allocator.select_next(["fragile"]) == "fragile"
    allocator.update(
        make_result(
            "fragile",
            2,
            score=None,
            cost_seconds=0.1,
            status=TrialStatus.FAILED,
        )
    )
    suspended_until = allocator.state_for("fragile").suspended_until_step
    assert suspended_until is not None

    for index in (3, 4):
        assert allocator.select_next() == "stable"
        allocator.update(make_result("stable", index, score=0.8, cost_seconds=1.0))

    assert allocator.selection_step == suspended_until - 1
    assert allocator.select_next(["fragile"]) == "fragile"
    assert allocator.state_for("fragile").suspended_until_step is None


def test_j4_policy_is_driven_entirely_by_synthetic_trial_results() -> None:
    allocator = FamilyAllocator(["family_a", "family_b"])
    outcomes = {"family_a": (0.75, 1.0), "family_b": (0.65, 2.0)}

    selections = [select_and_update(allocator, index, outcomes) for index in range(12)]

    assert len(selections) == 12
    assert sum(allocator.state_for(family).attempts for family in allocator.families) == 12
    assert sum(allocator.state_for(family).pending for family in allocator.families) == 0
    assert all(allocator.state_for(family).successes > 0 for family in allocator.families)


def test_snapshot_restore_produces_the_same_next_decision() -> None:
    configuration: dict[str, object] = {
        "minimum_exploration_trials": 1,
        "failure_threshold": 2,
        "cooldown_selections": 3,
        "performance_weight": 1.0,
        "improvement_weight": 0.25,
        "cost_weight": 0.5,
        "uncertainty_weight": 0.35,
        "diversity_weight": 0.15,
        "diversity_window": 4,
    }
    families = ["linear", "forest", "boosting"]
    outcomes = {
        "linear": (0.78, 1.0),
        "forest": (0.82, 4.0),
        "boosting": (0.85, 10.0),
    }
    original = FamilyAllocator(families, **configuration)  # type: ignore[arg-type]
    for index in range(9):
        select_and_update(original, index, outcomes)
    checkpoint = original.snapshot()

    restored = FamilyAllocator(**configuration)  # type: ignore[arg-type]
    restored.restore(checkpoint)

    assert restored.snapshot() == checkpoint
    original_next = original.select_next()
    restored_next = restored.select_next()
    assert restored_next == original_next
    assert restored.snapshot() == original.snapshot()

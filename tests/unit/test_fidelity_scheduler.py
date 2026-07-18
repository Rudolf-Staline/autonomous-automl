"""Multi-fidelity policy and successive-halving scheduler tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from autonomous_automl.contracts import PipelineSpec, TrialResult
from autonomous_automl.search import FidelityPolicy, FidelityScheduler
from autonomous_automl.utils.errors import ResumeError
from autonomous_automl.utils.json import canonical_json_dumps


def _spec(*, family: str = "trees", model: str = "random_forest", token: int = 0) -> PipelineSpec:
    return PipelineSpec(
        family=family,
        task="regression",
        numeric_imputer="median",
        numeric_scaler="none",
        categorical_imputer="constant",
        categorical_encoder="ordinal",
        datetime_transformer="calendar",
        model_name=model,
        model_params={"token": token},
        random_seed=42,
    )


def _result(
    trial_id: str,
    policy: FidelityPolicy,
    *,
    level: int = 0,
    mean: float = 0.8,
    std: float = 0.01,
    status: str = "completed",
    spec: PipelineSpec | None = None,
) -> TrialResult:
    pipeline_spec = spec or _spec(token=int(trial_id.removeprefix("trial-") or 0))
    now = datetime.now(UTC)
    values: dict[str, object] = {
        "trial_id": trial_id,
        "family": pipeline_spec.family,
        "pipeline_spec": pipeline_spec,
        "fidelity": policy.level(level),
        "status": status,
        "primary_metric": "rmse",
        "started_at": now,
        "finished_at": now,
    }
    if status == "completed":
        values.update(
            {
                "mean_score": mean,
                "std_score": std,
                "fold_scores": [mean - std, mean + std],
            }
        )
    elif status == "failed":
        values.update(
            {
                "failure_type": "SyntheticFailure",
                "failure_message": "aggregate-only synthetic failure",
            }
        )
    return TrialResult.model_validate(values)


def test_policy_materializes_four_levels_adapted_to_available_folds() -> None:
    policy = FidelityPolicy(available_folds=10, random_seed=42, confirmation_seed_count=2)

    assert [level.level for level in policy.levels] == [0, 1, 2, 3]
    assert [level.n_folds for level in policy.levels] == [2, 3, 5, 5]
    assert [level.sample_fraction for level in policy.levels] == [0.25, 0.60, 1.0, 1.0]
    assert len(policy.level(3).seeds) == 2
    assert len(set(policy.level(3).seeds)) == 2

    constrained = FidelityPolicy(available_folds=2, random_seed=42)
    assert [level.n_folds for level in constrained.levels] == [2, 2, 2, 2]


@pytest.mark.parametrize(
    "updates",
    [
        {"available_folds": 1},
        {"confirmation_seed_count": 1},
        {"low_sample_fraction": 0.8, "medium_sample_fraction": 0.6},
    ],
)
def test_policy_rejects_invalid_levels(updates: dict[str, object]) -> None:
    values: dict[str, object] = {"available_folds": 5, "random_seed": 42}
    values.update(updates)
    with pytest.raises(ValueError, match=r"must|sample fractions"):
        FidelityPolicy(**values)  # type: ignore[arg-type]


def test_successive_halving_uses_mean_minus_uncertainty_and_same_spec() -> None:
    policy = FidelityPolicy(available_folds=5, random_seed=7)
    scheduler = FidelityScheduler(policy, reduction_factor=3, uncertainty_weight=1.0)
    high_variance = _result("trial-1", policy, mean=0.95, std=0.20)
    stable = _result("trial-2", policy, mean=0.90, std=0.01)
    lower = _result("trial-3", policy, mean=0.80, std=0.01)

    scheduler.observe(high_variance)
    scheduler.observe(stable)
    promotions = scheduler.observe(lower)

    assert len(promotions) == 1
    assert promotions[0].parent_trial_id == stable.trial_id
    assert promotions[0].adjusted_score == pytest.approx(0.89)
    assert promotions[0].pipeline_spec == stable.pipeline_spec
    assert promotions[0].from_fidelity == policy.level(0)
    assert promotions[0].to_fidelity == policy.level(1)
    assert scheduler.should_promote(stable, [high_variance, lower])
    assert not scheduler.should_promote(high_variance, [stable, lower])


def test_failed_trials_never_count_toward_or_receive_promotion() -> None:
    policy = FidelityPolicy(available_folds=3, random_seed=9)
    scheduler = FidelityScheduler(policy, reduction_factor=3)
    scheduler.observe(_result("trial-1", policy, mean=0.9))
    scheduler.observe(_result("trial-2", policy, status="failed"))
    scheduler.observe(_result("trial-3", policy, mean=0.8))

    assert scheduler.pending_promotions() == ()
    scheduler.observe(_result("trial-4", policy, mean=0.7))

    promoted_ids = {promotion.parent_trial_id for promotion in scheduler.pending_promotions()}
    assert promoted_ids == {"trial-1"}
    assert "trial-2" not in promoted_ids


def test_promotion_slots_preserve_family_and_model_diversity() -> None:
    policy = FidelityPolicy(available_folds=5, random_seed=12)
    scheduler = FidelityScheduler(policy, reduction_factor=2)
    same_group = [
        _result(
            f"trial-{index}",
            policy,
            mean=1.0 - index / 100,
            spec=_spec(family="trees", model="random_forest", token=index),
        )
        for index in range(1, 4)
    ]
    other_group = _result(
        "trial-4",
        policy,
        mean=0.5,
        spec=_spec(family="linear", model="ridge", token=4),
    )

    for result in [*same_group, other_group]:
        scheduler.observe(result)

    promotions = scheduler.pending_promotions()
    groups = {(item.pipeline_spec.family, item.pipeline_spec.model_name) for item in promotions}
    assert len(promotions) == 2
    assert groups == {("trees", "random_forest"), ("linear", "ridge")}


def test_confirmation_results_are_terminal_for_the_scheduler() -> None:
    policy = FidelityPolicy(available_folds=5, random_seed=14)
    scheduler = FidelityScheduler(policy, reduction_factor=2)

    scheduler.observe(_result("trial-1", policy, level=3, mean=0.9))
    scheduler.observe(_result("trial-2", policy, level=3, mean=0.8))

    assert scheduler.next_level(policy.level(3)) is None
    assert scheduler.pending_promotions() == ()


def test_snapshot_restore_is_exact_and_preserves_pending_order() -> None:
    policy = FidelityPolicy(available_folds=5, random_seed=17)
    scheduler = FidelityScheduler(policy, reduction_factor=2)
    scheduler.observe(_result("trial-1", policy, mean=0.9))
    scheduler.observe(_result("trial-2", policy, mean=0.8))
    snapshot = scheduler.snapshot()
    serialized = canonical_json_dumps(snapshot)

    restored = FidelityScheduler(policy, reduction_factor=2)
    restored.restore(snapshot)

    assert canonical_json_dumps(restored.snapshot()) == serialized
    assert restored.pending_promotions() == scheduler.pending_promotions()
    assert restored.pop_next_promotion() == scheduler.pop_next_promotion()


def test_restore_rejects_policy_mismatch_and_observe_is_idempotent() -> None:
    policy = FidelityPolicy(available_folds=5, random_seed=21)
    scheduler = FidelityScheduler(policy, reduction_factor=2)
    result = _result("trial-1", policy)
    assert scheduler.observe(result) == ()
    assert scheduler.observe(result) == ()
    snapshot = scheduler.snapshot()

    incompatible = FidelityScheduler(
        FidelityPolicy(available_folds=4, random_seed=21),
        reduction_factor=2,
    )
    with pytest.raises(ResumeError, match="policy"):
        incompatible.restore(snapshot)

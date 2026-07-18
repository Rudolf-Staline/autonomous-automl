"""M8 acceptance tests for multi-fidelity sampling and promotion."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autonomous_automl.contracts import (
    DatasetBundle,
    FidelitySpec,
    FoldAssignment,
    MetricName,
    PipelineSpec,
    TaskType,
    TrialResult,
    TrialStatus,
    ValidationPlan,
)
from autonomous_automl.data import LoadedDataset
from autonomous_automl.pipelines import pipeline_fingerprint
from autonomous_automl.search import FidelityPolicy, FidelityScheduler
from autonomous_automl.search.sampling import FidelitySampler

STARTED_AT = datetime(2026, 7, 17, 10, 0, tzinfo=UTC)
FINISHED_AT = STARTED_AT + timedelta(seconds=1)


def make_dataset(
    target: list[object],
    *,
    groups: list[object] | None = None,
    time_values: list[object] | pd.DatetimeIndex | None = None,
    test_values: list[float] | None = None,
) -> LoadedDataset:
    n_rows = len(target)
    test_frame = None if test_values is None else pd.DataFrame({"signal": test_values})
    bundle = DatasetBundle(
        train_paths=[Path("/synthetic/train.csv")],
        test_path=None if test_frame is None else Path("/synthetic/test.csv"),
        target="target",
        feature_columns=["signal"],
        n_rows=n_rows,
        n_test_rows=None if test_frame is None else len(test_frame),
        source_hashes={
            "train": "a" * 64,
            **({"test": "b" * 64} if test_frame is not None else {}),
        },
    )
    return LoadedDataset(
        X=pd.DataFrame({"signal": np.arange(n_rows, dtype=float)}),
        y=pd.Series(target, name="target"),
        test_X=test_frame,
        row_ids=None,
        test_row_ids=None,
        groups=None if groups is None else pd.Series(groups, name="group_id"),
        time_values=(None if time_values is None else pd.Series(time_values, name="event_time")),
        predefined_folds=None,
        bundle=bundle,
    )


def make_plan(
    folds: list[FoldAssignment],
    *,
    group_column: str | None = None,
    time_column: str | None = None,
) -> ValidationPlan:
    if group_column is not None:
        splitter_name = "group_kfold"
    elif time_column is not None:
        splitter_name = "time_series_split"
    else:
        splitter_name = "stratified_kfold"
    shuffle = group_column is None and time_column is None
    return ValidationPlan(
        splitter_name=splitter_name,
        n_splits=len(folds),
        shuffle=shuffle,
        random_seed=42 if shuffle else None,
        group_column=group_column,
        time_column=time_column,
        rationale=["synthetic multi-fidelity acceptance plan"],
        folds=folds,
    )


def make_pipeline(
    *,
    family: str = "linear",
    model_name: str = "logistic_regression",
    random_seed: int = 42,
) -> PipelineSpec:
    return PipelineSpec(
        family=family,
        task=TaskType.BINARY_CLASSIFICATION,
        numeric_imputer="median",
        numeric_scaler="standard",
        categorical_imputer="constant",
        categorical_encoder="one_hot",
        datetime_transformer="drop",
        model_name=model_name,
        model_params={"regularization": 1.0},
        random_seed=random_seed,
    )


def make_result(
    trial_id: str,
    fidelity: FidelitySpec,
    *,
    score: float,
    uncertainty: float = 0.0,
    family: str = "linear",
    model_name: str = "logistic_regression",
) -> TrialResult:
    pipeline = make_pipeline(family=family, model_name=model_name)
    return TrialResult(
        trial_id=trial_id,
        family=family,
        pipeline_spec=pipeline,
        fidelity=fidelity,
        status=TrialStatus.COMPLETED,
        primary_metric=MetricName.ROC_AUC,
        mean_score=score,
        std_score=uncertainty,
        fold_scores=[score] * fidelity.n_folds,
        started_at=STARTED_AT,
        finished_at=FINISHED_AT,
    )


def test_policy_exposes_low_medium_full_confirmation_chain() -> None:
    policy = FidelityPolicy(available_folds=7, random_seed=42)
    scheduler = FidelityScheduler(policy)
    low, medium, full, confirmation = policy.levels

    assert [level.level for level in policy.levels] == [0, 1, 2, 3]
    assert 0 < low.sample_fraction < medium.sample_fraction < 1
    assert full.sample_fraction == confirmation.sample_fraction == 1
    assert [low.n_folds, medium.n_folds, full.n_folds, confirmation.n_folds] == [2, 3, 5, 5]
    assert low.max_iterations < medium.max_iterations < full.max_iterations
    assert confirmation.max_iterations == full.max_iterations
    assert scheduler.initial_level() == low
    assert scheduler.next_level(low) == medium
    assert scheduler.next_level(medium) == full
    assert scheduler.next_level(full) == confirmation
    assert scheduler.next_level(confirmation) is None


def test_confirmation_uses_multiple_reproducible_unique_seeds() -> None:
    first = FidelityPolicy(
        available_folds=5,
        random_seed=73,
        confirmation_seed_count=4,
    )
    second = FidelityPolicy(
        available_folds=5,
        random_seed=73,
        confirmation_seed_count=4,
    )

    assert first.level(3).seeds == second.level(3).seeds
    assert len(first.level(3).seeds) == 4
    assert len(set(first.level(3).seeds)) == 4
    assert len(first.level(2).seeds) == 1


def test_only_top_fraction_is_promoted_by_adjusted_score() -> None:
    scheduler = FidelityScheduler(
        FidelityPolicy(available_folds=5, random_seed=42),
        reduction_factor=3,
        uncertainty_weight=1.0,
    )
    low = scheduler.initial_level()
    results = [
        make_result("best", low, score=0.95, uncertainty=0.01),
        make_result("second", low, score=0.91, uncertainty=0.01),
        make_result("uncertain", low, score=0.99, uncertainty=0.20),
        make_result("fourth", low, score=0.80),
        make_result("fifth", low, score=0.70),
        make_result("sixth", low, score=0.60),
    ]

    assert sum(scheduler.should_promote(result, results) for result in results) == 2
    assert {result.trial_id for result in results if scheduler.should_promote(result, results)} == {
        "best",
        "second",
    }

    for result in results:
        scheduler.observe(result)

    assert {promotion.parent_trial_id for promotion in scheduler.pending_promotions()} == {
        "best",
        "second",
    }


def test_promotion_quota_preserves_model_diversity() -> None:
    scheduler = FidelityScheduler(
        FidelityPolicy(available_folds=5, random_seed=42),
        reduction_factor=3,
    )
    low = scheduler.initial_level()
    results = [
        make_result(f"linear-{index}", low, score=0.99 - index / 100) for index in range(3)
    ] + [
        make_result(
            f"forest-{index}",
            low,
            score=0.90 - index / 100,
            family="random_forest",
            model_name="random_forest",
        )
        for index in range(3)
    ]

    promoted = [result for result in results if scheduler.should_promote(result, results)]

    assert len(promoted) == 2
    assert {(result.family, result.pipeline_spec.model_name) for result in promoted} == {
        ("linear", "logistic_regression"),
        ("random_forest", "random_forest"),
    }


def test_snapshot_restore_preserves_results_queue_and_idempotence() -> None:
    policy = FidelityPolicy(available_folds=5, random_seed=42)
    scheduler = FidelityScheduler(policy, reduction_factor=3)
    low = scheduler.initial_level()
    results = [make_result(f"trial-{index}", low, score=0.90 - index / 100) for index in range(3)]
    for result in results:
        scheduler.observe(result)
    snapshot = scheduler.snapshot()

    restored = FidelityScheduler(policy, reduction_factor=3)
    restored.restore(snapshot)

    assert restored.snapshot() == snapshot
    assert restored.pending_promotions() == scheduler.pending_promotions()
    assert restored.observe(results[0]) == ()
    assert restored.snapshot() == snapshot
    assert restored.pop_next_promotion() == scheduler.pop_next_promotion()
    assert restored.pop_next_promotion() is None


def test_promotion_keeps_exact_pipeline_spec_while_only_fidelity_changes() -> None:
    scheduler = FidelityScheduler(
        FidelityPolicy(available_folds=5, random_seed=42),
        reduction_factor=3,
    )
    low = scheduler.initial_level()
    winner = make_result("winner", low, score=0.95)
    original_json = winner.pipeline_spec.to_json()
    original_fingerprint = pipeline_fingerprint(winner.pipeline_spec)

    scheduler.observe(winner)
    scheduler.observe(make_result("peer-1", low, score=0.70))
    promotions = scheduler.observe(make_result("peer-2", low, score=0.60))

    assert len(promotions) == 1
    promotion = promotions[0]
    assert promotion.parent_trial_id == winner.trial_id
    assert promotion.pipeline_spec == winner.pipeline_spec
    assert promotion.pipeline_spec.to_json() == original_json
    assert pipeline_fingerprint(promotion.pipeline_spec) == original_fingerprint
    assert promotion.from_fidelity == low
    assert promotion.to_fidelity == scheduler.next_level(low)
    assert promotion.to_fidelity != promotion.from_fidelity


@pytest.mark.parametrize("available_folds", [2, 3, 4])
def test_policy_adapts_fold_counts_to_small_validation_plans(available_folds: int) -> None:
    policy = FidelityPolicy(available_folds=available_folds, random_seed=42)

    assert all(level.n_folds <= available_folds for level in policy.levels)
    assert policy.level(0).n_folds == 2
    assert policy.level(3).sample_fraction == 1.0
    assert len(policy.level(3).seeds) >= 2


def test_same_seed_and_changed_test_frame_keep_the_same_training_sample() -> None:
    first_dataset = make_dataset(
        [index % 2 for index in range(20)],
        test_values=[10.0, 11.0],
    )
    changed_test_dataset = make_dataset(
        [index % 2 for index in range(20)],
        test_values=[-999_999.0, 999_999.0],
    )
    folds = [
        FoldAssignment(
            fold_index=0,
            train_positions=list(range(16)),
            validation_positions=list(range(16, 20)),
        ),
        FoldAssignment(
            fold_index=1,
            train_positions=list(range(4, 20)),
            validation_positions=list(range(4)),
        ),
    ]
    plan = make_plan(folds)
    fidelity = FidelityPolicy(available_folds=2, random_seed=42).level(0)

    first = FidelitySampler.sample_training_positions(
        first_dataset,
        plan,
        folds[0],
        fidelity,
        seed=321,
        task=TaskType.BINARY_CLASSIFICATION,
    )
    repeated = FidelitySampler.sample_training_positions(
        first_dataset,
        plan,
        folds[0],
        fidelity,
        seed=321,
        task=TaskType.BINARY_CLASSIFICATION,
    )
    changed_test = FidelitySampler.sample_training_positions(
        changed_test_dataset,
        plan,
        folds[0],
        fidelity,
        seed=321,
        task=TaskType.BINARY_CLASSIFICATION,
    )

    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(first, changed_test)
    assert set(first).issubset(folds[0].train_positions)
    assert set(first_dataset.y.iloc[first]) == {0, 1}


def test_group_sampling_selects_only_complete_training_groups() -> None:
    groups = [group for group in "ABCDEF" for _ in range(3)]
    dataset = make_dataset(
        [index % 2 for index in range(len(groups))],
        groups=groups,
    )
    folds = [
        FoldAssignment(
            fold_index=0,
            train_positions=list(range(12)),
            validation_positions=list(range(12, 15)),
        ),
        FoldAssignment(
            fold_index=1,
            train_positions=list(range(3, 18)),
            validation_positions=list(range(3)),
        ),
    ]
    plan = make_plan(folds, group_column="group_id")
    fidelity = FidelityPolicy(available_folds=2, random_seed=42).level(0)

    selected = FidelitySampler.sample_training_positions(
        dataset,
        plan,
        folds[0],
        fidelity,
        seed=91,
        task=TaskType.BINARY_CLASSIFICATION,
    )

    selected_set = set(selected.tolist())
    assert selected_set
    assert selected_set.issubset(folds[0].train_positions)
    for group in sorted({groups[index] for index in folds[0].train_positions}):
        group_positions = {index for index in folds[0].train_positions if groups[index] == group}
        assert selected_set.isdisjoint(group_positions) or group_positions.issubset(selected_set)


def test_time_sampling_returns_an_ordered_training_prefix() -> None:
    dataset = make_dataset(
        [float(index) for index in range(12)],
        time_values=pd.date_range("2026-01-01", periods=12, freq="D", tz="UTC"),
    )
    folds = [
        FoldAssignment(
            fold_index=0,
            train_positions=[5, 0, 3, 1, 4, 2],
            validation_positions=[6, 7],
        ),
        FoldAssignment(
            fold_index=1,
            train_positions=[7, 2, 5, 0, 6, 3, 1, 4],
            validation_positions=[8, 9, 10, 11],
        ),
    ]
    plan = make_plan(folds, time_column="event_time")
    fidelity = FidelityPolicy(
        available_folds=2,
        random_seed=42,
        low_sample_fraction=0.5,
    ).level(0)

    selected = FidelitySampler.sample_training_positions(
        dataset,
        plan,
        folds[0],
        fidelity,
        seed=42,
        task=TaskType.REGRESSION,
    )

    assert selected.tolist() == [0, 1, 2]
    assert dataset.time_values is not None
    assert dataset.time_values.iloc[selected].is_monotonic_increasing
    assert (
        dataset.time_values.iloc[selected].max()
        < dataset.time_values.iloc[folds[0].validation_positions].min()
    )


def test_small_and_rare_class_samples_remain_executable() -> None:
    rare_dataset = make_dataset([0, 0, 0, 0, 1, 2, 0, 1])
    rare_folds = [
        FoldAssignment(
            fold_index=0,
            train_positions=list(range(6)),
            validation_positions=[6, 7],
        ),
        FoldAssignment(
            fold_index=1,
            train_positions=list(range(2, 8)),
            validation_positions=[0, 1],
        ),
    ]
    rare_plan = make_plan(rare_folds)
    low = FidelityPolicy(available_folds=2, random_seed=42).level(0)

    rare_sample = FidelitySampler.sample_training_positions(
        rare_dataset,
        rare_plan,
        rare_folds[0],
        low,
        seed=7,
        task=TaskType.MULTICLASS_CLASSIFICATION,
    )

    assert len(rare_sample) == 3
    assert set(rare_dataset.y.iloc[rare_sample]) == {0, 1, 2}

    tiny_dataset = make_dataset([0.1, 0.2])
    tiny_folds = [
        FoldAssignment(fold_index=0, train_positions=[0], validation_positions=[1]),
        FoldAssignment(fold_index=1, train_positions=[1], validation_positions=[0]),
    ]
    tiny_plan = make_plan(tiny_folds)
    tiny_sample = FidelitySampler.sample_training_positions(
        tiny_dataset,
        tiny_plan,
        tiny_folds[0],
        low,
        seed=7,
        task=TaskType.REGRESSION,
    )

    assert tiny_sample.tolist() == [0]

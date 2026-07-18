"""M5 acceptance tests for metrics, OOF alignment, and isolated CV evaluation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from autonomous_automl.contracts import (
    AutoMLConfig,
    DatasetProfile,
    FidelitySpec,
    MetricName,
    PipelineSpec,
    TaskType,
    TrialStatus,
    ValidationPlan,
)
from autonomous_automl.data import LoadedDataset, load_dataset
from autonomous_automl.evaluation import (
    OOFAccumulator,
    PipelineEvaluator,
    resolve_metric,
    score_metric,
)
from autonomous_automl.profiling import profile_dataset
from autonomous_automl.utils.errors import MetricError
from autonomous_automl.validation import ValidationPlanner


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    """Minimal classifier that records which source markers reach ``fit``."""

    fit_markers: ClassVar[list[tuple[int, ...]]] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RecordingClassifier:
        type(self).fit_markers.append(tuple(int(value) for value in X["row_marker"]))
        self.classes_ = np.asarray(sorted(pd.unique(y).tolist()), dtype=object)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.repeat(self.classes_[0], len(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full((len(X), len(self.classes_)), 1.0 / len(self.classes_))


class ExplodingClassifier(RecordingClassifier):
    """Estimator used to ensure model failures stay inside one trial."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ExplodingClassifier:
        raise RuntimeError("intentional model failure")


class SlowClassifier(RecordingClassifier):
    """Estimator used to exercise cooperative timeout recording."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SlowClassifier:
        time.sleep(1.0)
        super().fit(X, y)
        return self


class RecordingMedianTransformer(TransformerMixin, BaseEstimator):
    """Fold-local transformer exposing the statistic learned by each clone."""

    fitted_medians: ClassVar[list[float]] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> RecordingMedianTransformer:
        median = float(X["signal"].median())
        self.median_ = median
        type(self).fitted_medians.append(median)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return X[["signal"]].fillna(self.median_).to_numpy()


def make_problem(
    tmp_path: Path,
) -> tuple[
    LoadedDataset,
    DatasetProfile,
    ValidationPlan,
    PipelineSpec,
    FidelitySpec,
]:
    n_rows = 60
    frame = pd.DataFrame(
        {
            "row_marker": np.arange(n_rows, dtype=int),
            "signal": [float(np.sin(index / 4)) for index in range(n_rows)],
            "category": [f"category-{index % 5}" for index in range(n_rows)],
            "target": [index % 2 for index in range(n_rows)],
        }
    )
    train_path = tmp_path / "train.csv"
    frame.to_csv(train_path, index=False)
    config = AutoMLConfig(
        target="target",
        task="binary_classification",
        metric="roc_auc",
        budget_seconds=30,
        random_seed=42,
        n_jobs=1,
        output_dir=tmp_path / "run",
    )
    dataset = load_dataset(train_path, config)
    profile = profile_dataset(dataset, TaskType.BINARY_CLASSIFICATION)
    plan = ValidationPlanner(max_splits=3).plan(dataset, profile, config)
    specification = PipelineSpec(
        family="linear",
        task="binary_classification",
        numeric_imputer="median",
        numeric_scaler="standard",
        categorical_imputer="constant",
        categorical_encoder="one_hot",
        datetime_transformer="drop",
        feature_selector=None,
        model_name="logistic_regression",
        model_params={"C": 1.0, "class_weight": None, "max_iter": 200},
        excluded_columns=["row_marker"],
        random_seed=42,
    )
    fidelity = FidelitySpec(
        level=2,
        sample_fraction=1.0,
        n_folds=plan.n_splits,
        max_iterations=100,
        seeds=[42],
    )
    return dataset, profile, plan, specification, fidelity


def test_auto_metric_policy_uses_task_and_imbalance_profile(tmp_path: Path) -> None:
    _, balanced_profile, _, _, _ = make_problem(tmp_path)
    imbalanced_profile = balanced_profile.model_copy(
        update={"target_profile": {**balanced_profile.target_profile, "imbalance_ratio": 9.0}}
    )

    balanced = resolve_metric(MetricName.AUTO, TaskType.BINARY_CLASSIFICATION, balanced_profile)
    imbalanced = resolve_metric(
        MetricName.AUTO,
        TaskType.BINARY_CLASSIFICATION,
        imbalanced_profile,
    )
    multiclass = resolve_metric(MetricName.AUTO, TaskType.MULTICLASS_CLASSIFICATION)
    regression = resolve_metric(MetricName.AUTO, TaskType.REGRESSION)

    assert balanced.name is MetricName.ROC_AUC
    assert balanced.requires_probabilities
    assert imbalanced.name is MetricName.AVERAGE_PRECISION
    assert multiclass.name is MetricName.MACRO_F1
    assert regression.name is MetricName.RMSE
    assert regression.is_loss


@pytest.mark.parametrize(
    ("metric", "truth", "predictions"),
    [
        (MetricName.RMSE, [1.0, 2.0, 3.0], [2.0, 2.0, 5.0]),
        (MetricName.MAE, [1.0, 2.0, 3.0], [2.0, 2.0, 5.0]),
    ],
)
def test_loss_metrics_are_negative_internally_and_positive_for_users(
    metric: MetricName,
    truth: list[float],
    predictions: list[float],
) -> None:
    definition = resolve_metric(metric, TaskType.REGRESSION)

    value = score_metric(definition, truth, predictions=predictions)

    assert value.user_value > 0
    assert value.internal_value == pytest.approx(-value.user_value)


def test_log_loss_uses_the_same_negative_internal_orientation() -> None:
    definition = resolve_metric(MetricName.LOG_LOSS, TaskType.BINARY_CLASSIFICATION)

    value = score_metric(
        definition,
        [0, 1, 1, 0],
        probabilities=np.asarray([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.8, 0.2]]),
        classes=[0, 1],
    )

    assert value.user_value > 0
    assert value.internal_value == pytest.approx(-value.user_value)


def test_invalid_probability_matrix_is_rejected() -> None:
    definition = resolve_metric(MetricName.ROC_AUC, TaskType.BINARY_CLASSIFICATION)

    with pytest.raises(MetricError, match="non-negative"):
        score_metric(
            definition,
            [0, 1],
            probabilities=np.asarray([[1.1, -0.1], [0.2, 0.8]]),
            classes=[0, 1],
        )


def test_incompatible_metric_is_rejected_before_pipeline_execution() -> None:
    with pytest.raises(MetricError, match="incompatible"):
        resolve_metric(MetricName.RMSE, TaskType.BINARY_CLASSIFICATION)


def test_oof_accumulator_aligns_local_probability_columns_to_global_classes() -> None:
    accumulator = OOFAccumulator(
        4,
        TaskType.MULTICLASS_CLASSIFICATION,
        classes=["ant", "bird", "cat"],
        row_ids=pd.Series(["row-a", "row-b", "row-c", "row-d"]),
    )

    first_aligned = accumulator.add_classification(
        [0, 1],
        ["cat", "ant"],
        np.asarray([[0.8, 0.2], [0.1, 0.9]]),
        model_classes=["cat", "ant"],
    )
    second_aligned = accumulator.add_classification(
        [2, 3],
        ["bird", "cat"],
        np.asarray([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]),
        model_classes=["bird", "cat", "ant"],
    )
    frame = accumulator.to_frame()

    np.testing.assert_allclose(first_aligned, [[0.2, 0.0, 0.8], [0.9, 0.0, 0.1]])
    np.testing.assert_allclose(second_aligned, [[0.1, 0.7, 0.2], [0.1, 0.1, 0.8]])
    assert accumulator.is_complete
    assert accumulator.counts.tolist() == [1, 1, 1, 1]
    assert frame["row_id"].tolist() == ["row-a", "row-b", "row-c", "row-d"]
    assert frame["prediction"].tolist() == ["cat", "ant", "bird", "cat"]
    assert "target" not in frame.columns


def test_oof_labels_preserve_model_predictions_instead_of_probability_argmax() -> None:
    accumulator = OOFAccumulator(1, TaskType.BINARY_CLASSIFICATION, classes=[0, 1])

    accumulator.add_classification(
        [0],
        [0],
        np.asarray([[0.1, 0.9]]),
        model_classes=[0, 1],
    )

    assert accumulator.to_frame().loc[0, "prediction"] == 0


def test_evaluator_produces_complete_reproducible_oof_predictions(tmp_path: Path) -> None:
    dataset, profile, plan, specification, fidelity = make_problem(tmp_path)
    evaluator = PipelineEvaluator(n_jobs=1)

    first = evaluator.evaluate(
        "trial-reproducible",
        specification,
        fidelity,
        dataset,
        profile,
        plan,
        MetricName.ROC_AUC,
    )
    repeated = evaluator.evaluate(
        "trial-reproducible",
        specification,
        fidelity,
        dataset,
        profile,
        plan,
        MetricName.ROC_AUC,
    )

    assert first.trial_result.status is TrialStatus.COMPLETED
    assert first.trial_result.fold_scores == repeated.trial_result.fold_scores
    assert first.trial_result.mean_score == repeated.trial_result.mean_score
    assert len(first.trial_result.fold_results) == plan.n_splits
    assert first.oof_predictions is not None
    assert repeated.oof_predictions is not None
    pd.testing.assert_frame_equal(first.oof_predictions, repeated.oof_predictions)
    assert len(first.oof_predictions) == len(dataset.X)
    assert first.oof_predictions["oof_count"].eq(1).all()
    assert first.oof_predictions.filter(like="probability_").notna().all().all()
    assert "target" not in first.oof_predictions.columns


def test_execution_seed_is_independent_from_tracking_trial_id(tmp_path: Path) -> None:
    dataset, profile, plan, specification, fidelity = make_problem(tmp_path)
    evaluator = PipelineEvaluator(n_jobs=1)

    first = evaluator.evaluate(
        "tracking-id-one",
        specification,
        fidelity,
        dataset,
        profile,
        plan,
        MetricName.ROC_AUC,
    )
    second = evaluator.evaluate(
        "tracking-id-two",
        specification,
        fidelity,
        dataset,
        profile,
        plan,
        MetricName.ROC_AUC,
    )

    assert [fold.seed for fold in first.trial_result.fold_results] == [
        fold.seed for fold in second.trial_result.fold_results
    ]
    assert first.trial_result.fold_scores == second.trial_result.fold_scores


def test_every_fit_receives_only_its_materialized_training_fold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, profile, plan, specification, fidelity = make_problem(tmp_path)
    RecordingClassifier.fit_markers.clear()

    def build_recording_pipeline(*args: object, **kwargs: object) -> Pipeline:
        return Pipeline([("model", RecordingClassifier())])

    monkeypatch.setattr(
        "autonomous_automl.evaluation.evaluator.build_pipeline",
        build_recording_pipeline,
    )
    outcome = PipelineEvaluator(n_jobs=1).evaluate(
        "trial-fold-local",
        specification,
        fidelity,
        dataset,
        profile,
        plan,
        MetricName.ACCURACY,
    )

    expected_markers = [
        tuple(int(value) for value in dataset.X.iloc[fold.train_positions]["row_marker"])
        for fold in plan.folds
    ]
    assert outcome.trial_result.status is TrialStatus.COMPLETED
    assert RecordingClassifier.fit_markers == expected_markers
    assert all(len(markers) < len(dataset.X) for markers in RecordingClassifier.fit_markers)
    assert tuple(range(len(dataset.X))) not in RecordingClassifier.fit_markers


def test_imputation_statistic_is_learned_independently_inside_each_fold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, profile, plan, specification, fidelity = make_problem(tmp_path)
    dataset.X.loc[::7, "signal"] = np.nan
    RecordingMedianTransformer.fitted_medians.clear()

    def build_recording_imputer(*args: object, **kwargs: object) -> Pipeline:
        return Pipeline(
            [
                ("median", RecordingMedianTransformer()),
                ("model", DummyClassifier(strategy="prior")),
            ]
        )

    monkeypatch.setattr(
        "autonomous_automl.evaluation.evaluator.build_pipeline",
        build_recording_imputer,
    )
    outcome = PipelineEvaluator().evaluate(
        "trial-fold-imputation",
        specification,
        fidelity,
        dataset,
        profile,
        plan,
        MetricName.ACCURACY,
    )

    expected = [
        float(dataset.X.iloc[fold.train_positions]["signal"].median()) for fold in plan.folds
    ]
    global_median = float(dataset.X["signal"].median())
    assert outcome.trial_result.status is TrialStatus.COMPLETED
    assert RecordingMedianTransformer.fitted_medians == expected
    assert any(median != global_median for median in expected)


def test_model_error_is_recorded_as_a_failed_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, profile, plan, specification, fidelity = make_problem(tmp_path)

    def build_exploding_pipeline(*args: object, **kwargs: object) -> Pipeline:
        return Pipeline([("model", ExplodingClassifier())])

    monkeypatch.setattr(
        "autonomous_automl.evaluation.evaluator.build_pipeline",
        build_exploding_pipeline,
    )
    outcome = PipelineEvaluator().evaluate(
        "trial-model-error",
        specification,
        fidelity,
        dataset,
        profile,
        plan,
        MetricName.ACCURACY,
    )

    assert outcome.trial_result.status is TrialStatus.FAILED
    assert outcome.trial_result.failure_type == "RuntimeError"
    assert "intentional model failure" in (outcome.trial_result.failure_message or "")
    assert outcome.trial_result.mean_score is None
    assert outcome.oof_predictions is None


def test_timeout_is_recorded_as_a_failed_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, profile, plan, specification, fidelity = make_problem(tmp_path)

    def build_slow_pipeline(*args: object, **kwargs: object) -> Pipeline:
        return Pipeline([("model", SlowClassifier())])

    monkeypatch.setattr(
        "autonomous_automl.evaluation.evaluator.build_pipeline",
        build_slow_pipeline,
    )
    started = time.monotonic()
    outcome = PipelineEvaluator().evaluate(
        "trial-timeout",
        specification,
        fidelity,
        dataset,
        profile,
        plan,
        MetricName.ACCURACY,
        timeout_seconds=0.005,
    )
    elapsed = time.monotonic() - started

    assert outcome.trial_result.status is TrialStatus.FAILED
    assert outcome.trial_result.failure_type == "TrialTimeoutError"
    assert "time limit" in (outcome.trial_result.failure_message or "")
    assert outcome.oof_predictions is None
    assert elapsed < 0.5


def test_evaluator_rejects_global_metric_mismatch_instead_of_recording_trial(
    tmp_path: Path,
) -> None:
    dataset, profile, plan, specification, fidelity = make_problem(tmp_path)

    with pytest.raises(MetricError, match="incompatible"):
        PipelineEvaluator().evaluate(
            "trial-invalid-global-config",
            specification,
            fidelity,
            dataset,
            profile,
            plan,
            MetricName.RMSE,
        )

"""Integration tests for deterministic, leakage-safe final training."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from autonomous_automl.contracts import DatasetBundle, DatasetProfile, PipelineSpec, TaskType
from autonomous_automl.data import LoadedDataset
from autonomous_automl.evaluation.finalization import FinalTrainer
from autonomous_automl.tracking.artifacts import ArtifactStore
from autonomous_automl.tracking.store import ArtifactRegistration


def make_problem(
    task: TaskType,
    *,
    test_frame: pd.DataFrame | None,
) -> tuple[LoadedDataset, DatasetProfile]:
    n_rows = 72
    signal = np.linspace(-3.0, 3.0, n_rows)
    categories = np.asarray(["alpha", "beta", "gamma"] * (n_rows // 3), dtype=object)
    X = pd.DataFrame({"signal": signal, "category": categories})
    if task is TaskType.REGRESSION:
        target = pd.Series(
            2.5 * signal + np.where(categories == "alpha", 0.75, -0.25),
            name="target",
        )
    else:
        target = pd.Series(
            np.where((signal > 0.0) | (categories == "alpha"), "yes", "no"),
            name="target",
        )
    bundle = DatasetBundle(
        train_paths=[Path("/synthetic/final-train.csv")],
        test_path=None if test_frame is None else Path("/synthetic/final-test.csv"),
        target="target",
        feature_columns=["signal", "category"],
        n_rows=n_rows,
        n_test_rows=None if test_frame is None else len(test_frame),
        source_hashes={
            "train": "a" * 64,
            **({"test": "b" * 64} if test_frame is not None else {}),
        },
    )
    dataset = LoadedDataset(
        X=X,
        y=target,
        test_X=None if test_frame is None else test_frame.copy(),
        row_ids=pd.Series([f"train-{index}" for index in range(n_rows)]),
        test_row_ids=(
            None
            if test_frame is None
            else pd.Series([f"test-{index}" for index in range(len(test_frame))])
        ),
        groups=None,
        time_values=None,
        predefined_folds=None,
        bundle=bundle,
    )
    profile = DatasetProfile(
        dataset_hash="a" * 64,
        n_rows=n_rows,
        n_features=2,
        inferred_task=task,
        inferred_types={"signal": "numeric", "category": "categorical"},
        numeric_columns=["signal"],
        categorical_columns=["category"],
        missing_ratios={"signal": 0.0, "category": 0.0},
        cardinalities={"signal": n_rows, "category": 3},
        unique_ratios={"signal": 1.0, "category": 3 / n_rows},
        numeric_skewness={"signal": 0.0},
        infinite_counts={"signal": 0},
        target_profile={"n_classes": 2} if task is not TaskType.REGRESSION else {},
        landmarks={"numeric_fraction": 0.5, "categorical_fraction": 0.5},
        estimated_memory_mb=0.01,
    )
    return dataset, profile


def classification_spec() -> PipelineSpec:
    return PipelineSpec(
        family="random_forest",
        task=TaskType.BINARY_CLASSIFICATION,
        numeric_imputer="median",
        numeric_scaler="none",
        categorical_imputer="constant",
        categorical_encoder="one_hot",
        datetime_transformer="drop",
        model_name="random_forest",
        model_params={
            "n_estimators": 30,
            "max_depth": 6,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },
        random_seed=1234,
    )


def regression_spec() -> PipelineSpec:
    return PipelineSpec(
        family="linear",
        task=TaskType.REGRESSION,
        numeric_imputer="median",
        numeric_scaler="standard",
        categorical_imputer="constant",
        categorical_encoder="one_hot",
        datetime_transformer="drop",
        model_name="ridge",
        model_params={"alpha": 0.5},
        random_seed=91,
    )


def test_classification_fits_once_on_all_train_rows_and_aligns_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_frame = pd.DataFrame(
        {
            "signal": [-4.0, 0.2, 4.0],
            "category": ["never-seen", "alpha", "beta"],
        }
    )
    dataset, profile = make_problem(TaskType.BINARY_CLASSIFICATION, test_frame=test_frame)
    specification = classification_spec()
    original_specification = specification.to_json()
    original_fit = Pipeline.fit
    fitted_row_counts: list[int] = []

    def recording_fit(
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        **params: object,
    ) -> Pipeline:
        fitted_row_counts.append(len(X))
        return original_fit(pipeline, X, y, **params)

    monkeypatch.setattr(Pipeline, "fit", recording_fit)

    result = FinalTrainer(n_jobs=1).fit(specification, dataset, profile)

    assert fitted_row_counts == [len(dataset.X)]
    assert result.pipeline_spec.to_json() == original_specification
    assert result.n_train_rows == len(dataset.X)
    assert result.n_test_rows == len(test_frame)
    assert result.fit_seconds >= 0.0
    assert result.predict_seconds >= 0.0
    assert result.classes == ("no", "yes")
    assert result.probability_columns == ("probability_0", "probability_1")
    assert result.predictions is not None
    assert result.probabilities is not None
    assert result.prediction_frame is not None
    assert result.prediction_frame.columns.tolist() == [
        "row_position",
        "row_id",
        "prediction",
        "probability_0",
        "probability_1",
    ]
    assert result.prediction_frame["row_id"].tolist() == ["test-0", "test-1", "test-2"]
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0)


def test_fixed_seed_is_stable_and_test_changes_cannot_change_fitted_pipeline() -> None:
    first_test = pd.DataFrame({"signal": [-2.0, 2.0], "category": ["never-seen", "alpha"]})
    changed_test = pd.DataFrame(
        {"signal": [-1_000_000.0, 1_000_000.0], "category": ["other", "other"]}
    )
    first_dataset, profile = make_problem(
        TaskType.BINARY_CLASSIFICATION,
        test_frame=first_test,
    )
    changed_dataset, _ = make_problem(
        TaskType.BINARY_CLASSIFICATION,
        test_frame=changed_test,
    )
    trainer = FinalTrainer(n_jobs=1)
    specification = classification_spec()

    first = trainer.fit(specification, first_dataset, profile)
    repeated = trainer.fit(specification, first_dataset, profile)
    changed = trainer.fit(specification, changed_dataset, profile)

    assert joblib.hash(first.pipeline) == joblib.hash(repeated.pipeline)
    assert joblib.hash(first.pipeline) == joblib.hash(changed.pipeline)
    assert first.predictions is not None
    assert repeated.predictions is not None
    assert first.probabilities is not None
    assert repeated.probabilities is not None
    np.testing.assert_array_equal(first.predictions, repeated.predictions)
    np.testing.assert_allclose(first.probabilities, repeated.probabilities, rtol=0.0, atol=0.0)
    assert changed.pipeline_spec == specification


def test_regression_predictions_are_stable_and_test_inference_is_optional() -> None:
    test_frame = pd.DataFrame(
        {
            "signal": [-1.5, 0.0, 2.5],
            "category": ["never-seen", "gamma", "alpha"],
        }
    )
    dataset, profile = make_problem(TaskType.REGRESSION, test_frame=test_frame)
    trainer = FinalTrainer(n_jobs=1)

    result = trainer.fit(regression_spec(), dataset, profile)
    repeated = trainer.fit(regression_spec(), dataset, profile)
    skipped = trainer.fit(regression_spec(), dataset, profile, predict_test=False)

    assert result.predictions is not None
    assert repeated.predictions is not None
    np.testing.assert_allclose(result.predictions, repeated.predictions, rtol=0.0, atol=0.0)
    assert result.probabilities is None
    assert result.classes is None
    assert result.probability_columns == ()
    assert result.prediction_frame is not None
    assert result.prediction_frame.columns.tolist() == [
        "row_position",
        "row_id",
        "prediction",
    ]
    assert skipped.pipeline is not None
    assert skipped.predictions is None
    assert skipped.probabilities is None
    assert skipped.prediction_frame is None
    assert skipped.n_test_rows == 0
    assert skipped.predict_seconds == 0.0


def test_joblib_round_trip_uses_atomic_artifact_store_and_verified_loader(
    tmp_path: Path,
) -> None:
    test_frame = pd.DataFrame({"signal": [-2.0, 2.0], "category": ["never-seen", "alpha"]})
    dataset, profile = make_problem(TaskType.BINARY_CLASSIFICATION, test_frame=test_frame)
    trainer = FinalTrainer(n_jobs=1)
    result = trainer.fit(classification_spec(), dataset, profile)
    writing_store = ArtifactStore(tmp_path / "run")

    record = trainer.save_pipeline(result, writing_store)
    registration = ArtifactRegistration(
        artifact_id=1,
        run_id="run-1",
        name=record.name,
        kind=record.kind,
        relative_path=record.relative_path,
        sha256=record.sha256,
        size_bytes=record.size_bytes,
        media_type=record.media_type,
    )
    registry = Mock()
    registry.get_artifact.return_value = registration
    loading_store = ArtifactStore(
        writing_store.run_directory,
        registry=registry,
        run_id="run-1",
    )

    restored = trainer.load_pipeline(loading_store, registration)

    assert result.predictions is not None
    assert result.probabilities is not None
    np.testing.assert_array_equal(restored.predict(test_frame), result.predictions)
    np.testing.assert_allclose(restored.predict_proba(test_frame), result.probabilities)
    registry.get_artifact.assert_called_once_with("run-1", "best_pipeline")

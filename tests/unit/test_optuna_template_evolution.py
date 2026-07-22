"""Compatibility tests for additive optimizer template evolution."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from autonomous_automl.components import ModelRegistry
from autonomous_automl.contracts import DatasetProfile, FidelitySpec, PipelineSpec
from autonomous_automl.pipelines import pipeline_fingerprint
from autonomous_automl.search import OptunaFamilyOptimizer
from autonomous_automl.utils.errors import ResumeError


def _profile(*, dataset_hash: str = "a" * 64) -> DatasetProfile:
    return DatasetProfile(
        dataset_hash=dataset_hash,
        n_rows=120,
        n_features=80,
        inferred_task="regression",
        inferred_types={f"feature_{index}": "numeric" for index in range(80)},
        numeric_columns=[f"feature_{index}" for index in range(80)],
        missing_ratios={f"feature_{index}": 0.0 for index in range(80)},
        cardinalities={f"feature_{index}": 120 for index in range(80)},
        unique_ratios={f"feature_{index}": 1.0 for index in range(80)},
        numeric_skewness={f"feature_{index}": 0.0 for index in range(80)},
        infinite_counts={f"feature_{index}": 0 for index in range(80)},
        estimated_memory_mb=0.1,
    )


def _template(feature_selector: str | None) -> PipelineSpec:
    return PipelineSpec(
        family="linear",
        task="regression",
        numeric_imputer="mean",
        numeric_scaler="standard",
        categorical_imputer="constant",
        categorical_encoder="ordinal",
        datetime_transformer="calendar",
        feature_selector=feature_selector,
        model_name="ridge",
        model_params={"alpha": 1.0},
        random_seed=42,
    )


def _fidelity() -> FidelitySpec:
    return FidelitySpec(
        level=0,
        sample_fraction=0.5,
        n_folds=2,
        max_iterations=50,
        seeds=[42],
    )


def _optimizer(
    database: Path,
    templates: list[PipelineSpec],
    *,
    dataset_hash: str = "a" * 64,
    prefix: str = "template-evolution",
) -> OptunaFamilyOptimizer:
    return OptunaFamilyOptimizer(
        "linear",
        templates,
        _profile(dataset_hash=dataset_hash),
        database,
        random_seed=7,
        registry=ModelRegistry.default(include_optional=False),
        study_prefix=prefix,
    )


def _persisted_fingerprints(optimizer: OptunaFamilyOptimizer) -> list[str]:
    identity = optimizer.study_for_model("ridge").user_attrs["optimizer_identity"]
    assert isinstance(identity, dict)
    fingerprints = identity["template_fingerprints"]
    assert isinstance(fingerprints, list)
    return cast(list[str], fingerprints)


def test_resumed_study_keeps_its_original_template_universe(tmp_path: Path) -> None:
    database = tmp_path / "resume.sqlite3"
    original = _template(None)
    added = _template("univariate_50")
    first = _optimizer(database, [original])
    original_fingerprints = _persisted_fingerprints(first)

    resumed = _optimizer(database, [original, added])
    candidate = resumed.ask(_fidelity())

    assert original_fingerprints == [pipeline_fingerprint(original)]
    assert _persisted_fingerprints(resumed) == original_fingerprints
    assert candidate.template_fingerprint == pipeline_fingerprint(original)
    assert candidate.pipeline_spec.feature_selector is None


def test_new_study_records_all_current_templates(tmp_path: Path) -> None:
    original = _template(None)
    added = _template("univariate_50")
    optimizer = _optimizer(tmp_path / "new.sqlite3", [original, added], prefix="new-study")

    assert _persisted_fingerprints(optimizer) == [
        pipeline_fingerprint(original),
        pipeline_fingerprint(added),
    ]


def test_resume_rejects_removal_of_a_persisted_template(tmp_path: Path) -> None:
    database = tmp_path / "removed.sqlite3"
    original = _template(None)
    replacement = _template("univariate_50")
    _optimizer(database, [original], prefix="removed")

    with pytest.raises(ResumeError, match="template is unavailable"):
        _optimizer(database, [replacement], prefix="removed")


def test_additive_template_compatibility_does_not_weaken_dataset_identity(tmp_path: Path) -> None:
    database = tmp_path / "dataset.sqlite3"
    original = _template(None)
    added = _template("univariate_50")
    _optimizer(database, [original], dataset_hash="a" * 64, prefix="dataset")

    with pytest.raises(ResumeError, match="identity"):
        _optimizer(
            database,
            [original, added],
            dataset_hash="b" * 64,
            prefix="dataset",
        )

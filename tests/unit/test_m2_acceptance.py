"""Cross-component M2 acceptance tests for safe loading and profiling."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pytest

from autonomous_automl.contracts import AutoMLConfig, DatasetProfile, TaskType
from autonomous_automl.data import LoadedDataset, load_dataset
from autonomous_automl.profiling import DatasetProfiler, profile_dataset
from autonomous_automl.utils.errors import DataValidationError


def write_csv(path: Path, frame: pd.DataFrame, *, separator: str = ",") -> Path:
    frame.to_csv(path, index=False, sep=separator)
    return path


def training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            "city": ["rabat", "casa", "rabat", "casa"] * 3,
            "active": [True, False] * 6,
            "target": [0, 1, 0, 1] * 3,
        }
    )


def make_config(tmp_path: Path, **updates: object) -> AutoMLConfig:
    values: dict[str, object] = {
        "target": "target",
        "task": "auto",
        "metric": "auto",
        "budget_seconds": 30,
        "random_seed": 42,
        "n_jobs": 1,
        "output_dir": tmp_path / "run",
    }
    values.update(updates)
    return AutoMLConfig.model_validate(values)


def source_digest(dataset: LoadedDataset) -> str:
    hashes: Mapping[str, str] = dataset.bundle.source_hashes
    assert len(hashes) == 1
    return next(iter(hashes.values()))


def test_loader_establishes_a_target_feature_firewall(tmp_path: Path) -> None:
    train_path = write_csv(tmp_path / "train.csv", training_frame())

    dataset = load_dataset([train_path], make_config(tmp_path))

    assert dataset.X.columns.tolist() == ["amount", "city", "active"]
    assert "target" not in dataset.X
    assert "target" not in dataset.bundle.feature_columns
    assert dataset.y.name == "target"
    assert dataset.y.tolist() == training_frame()["target"].tolist()
    assert len(dataset.X) == len(dataset.y) == dataset.bundle.n_rows


def test_test_values_and_unknown_categories_cannot_change_training_profile(
    tmp_path: Path,
) -> None:
    train_path = write_csv(tmp_path / "train.csv", training_frame())
    test_path = write_csv(
        tmp_path / "test.csv",
        pd.DataFrame(
            {
                "amount": [1.0e300, -1.0e300],
                "city": ["category_only_in_test", "another_unknown_category"],
                "active": [True, False],
            }
        ),
    )
    train_only = load_dataset([train_path], make_config(tmp_path))
    with_extreme_test = load_dataset(
        [train_path],
        make_config(tmp_path, test_path=test_path),
    )

    direct_profile = DatasetProfiler().profile(train_only, task=TaskType.AUTO)
    profile_without_test = profile_dataset(train_only)
    profile_with_test = profile_dataset(with_extreme_test)

    assert direct_profile == profile_without_test
    assert profile_with_test == profile_without_test
    assert with_extreme_test.test_X is not None
    assert "category_only_in_test" not in profile_with_test.target_profile.values()


def test_source_hash_is_stable_and_changes_with_file_bytes(tmp_path: Path) -> None:
    train_path = write_csv(tmp_path / "train.csv", training_frame())
    config = make_config(tmp_path)

    first = source_digest(load_dataset([train_path], config))
    second = source_digest(load_dataset([train_path], config))
    changed_frame = training_frame()
    changed_frame.loc[0, "amount"] = 999.0
    write_csv(train_path, changed_frame)
    changed = source_digest(load_dataset([train_path], config))

    assert first == second
    assert len(first) == 64
    assert changed != first


def test_loader_detects_semicolon_delimited_csv(tmp_path: Path) -> None:
    train_path = write_csv(
        tmp_path / "semicolon.csv",
        training_frame(),
        separator=";",
    )

    dataset = load_dataset([train_path], make_config(tmp_path))

    assert dataset.X.columns.tolist() == ["amount", "city", "active"]
    assert dataset.y.tolist() == training_frame()["target"].tolist()


@pytest.mark.parametrize(
    "test_frame",
    [
        pd.DataFrame({"amount": [1.0], "active": [True]}),
        pd.DataFrame(
            {
                "amount": [1.0],
                "city": ["rabat"],
                "active": [True],
                "unexpected": ["extra"],
            }
        ),
        pd.DataFrame({"amount": [1.0], "city": ["rabat"], "active": [True], "target": [0]}),
    ],
    ids=["missing", "extra", "target-in-test"],
)
def test_loader_requires_the_exact_training_feature_schema_for_test_data(
    tmp_path: Path,
    test_frame: pd.DataFrame,
) -> None:
    train_path = write_csv(tmp_path / "train.csv", training_frame())
    test_path = write_csv(tmp_path / "test.csv", test_frame)

    with pytest.raises(DataValidationError, match=r"(?i)(schema|column|target)"):
        load_dataset([train_path], make_config(tmp_path, test_path=test_path))


def test_loader_realigns_an_exact_but_reordered_test_schema(tmp_path: Path) -> None:
    train_path = write_csv(tmp_path / "train.csv", training_frame())
    test_path = write_csv(
        tmp_path / "test.csv",
        pd.DataFrame({"city": ["unknown"], "amount": [123.0], "active": [True]}),
    )

    dataset = load_dataset([train_path], make_config(tmp_path, test_path=test_path))

    assert dataset.test_X is not None
    assert dataset.test_X.columns.tolist() == dataset.X.columns.tolist()
    assert dataset.test_X.loc[0, "city"] == "unknown"


def test_profile_is_a_versioned_json_contract_derived_from_training_only(
    tmp_path: Path,
) -> None:
    train_path = write_csv(tmp_path / "train.csv", training_frame())
    dataset = load_dataset([train_path], make_config(tmp_path))

    profile = profile_dataset(dataset)
    restored = DatasetProfile.from_json(profile.to_json())

    assert restored == profile
    assert profile.schema_version == 1
    assert profile.n_rows == len(dataset.X)
    assert profile.n_features == dataset.X.shape[1]
    assert set(profile.inferred_types) == set(dataset.X.columns)
    assert len(profile.dataset_hash) == 64


def test_profile_json_file_round_trip_preserves_column_roles(tmp_path: Path) -> None:
    train_path = write_csv(tmp_path / "train.csv", training_frame())
    profile = profile_dataset(load_dataset([train_path], make_config(tmp_path)))
    destination = tmp_path / "artifacts" / "dataset_profile.json"

    profile.write_json(destination)
    restored = DatasetProfile.read_json(destination)

    assert restored == profile
    assert restored.numeric_columns == ["amount"]
    assert restored.categorical_columns == ["city"]
    assert restored.boolean_columns == ["active"]

"""M2 tests for safe CSV loading and the target firewall."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from autonomous_automl import AutoMLConfig
from autonomous_automl.data import DataLoader, load_dataset
from autonomous_automl.utils.errors import DataValidationError


def _write_csv(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _config(**updates: object) -> AutoMLConfig:
    values: dict[str, object] = {"target": "target", "budget_seconds": 60}
    values.update(updates)
    return AutoMLConfig.model_validate(values)


def test_loader_separates_target_and_reserved_columns(tmp_path: Path) -> None:
    train = _write_csv(
        tmp_path / "train.csv",
        "row_id,group_id,event_time,fold,num,cat,target\n"
        "a,g1,2024-01-01,0,1,x,0\n"
        "b,g2,2024-01-02,1,2,y,1\n",
    )
    config = _config(
        id_column="row_id",
        group_column="group_id",
        time_column="event_time",
        predefined_fold_column="fold",
    )

    dataset = load_dataset(train, config)

    assert list(dataset.X.columns) == ["event_time", "num", "cat"]
    assert "target" not in dataset.X
    assert dataset.y.tolist() == [0, 1]
    assert dataset.row_ids is not None
    assert dataset.row_ids.tolist() == ["a", "b"]
    assert dataset.groups is not None
    assert dataset.groups.tolist() == ["g1", "g2"]
    assert dataset.predefined_folds is not None
    assert dataset.predefined_folds.tolist() == [0, 1]
    assert dataset.time_values is not None
    assert dataset.time_values.tolist() == ["2024-01-01", "2024-01-02"]
    assert dataset.bundle.excluded_columns == ["fold", "group_id", "row_id"]


@pytest.mark.parametrize("delimiter", [",", ";", "\t", "|"])
def test_loader_detects_supported_delimiters(tmp_path: Path, delimiter: str) -> None:
    train = _write_csv(
        tmp_path / "train.csv",
        f"number{delimiter}category{delimiter}target\n"
        f"1{delimiter}a{delimiter}0\n"
        f"2{delimiter}b{delimiter}1\n",
    )

    dataset = DataLoader().load(train, _config())

    assert list(dataset.X.columns) == ["number", "category"]
    assert dataset.X.shape == (2, 2)


def test_loader_concatenates_reordered_compatible_training_files(tmp_path: Path) -> None:
    first = _write_csv(tmp_path / "first.csv", "a,b,target\n1,x,0\n2,y,1\n")
    second = _write_csv(tmp_path / "second.csv", "target,b,a\n1,z,3\n0,w,4\n")

    dataset = load_dataset([first, second], _config())

    assert list(dataset.X.columns) == ["a", "b"]
    assert dataset.X["a"].tolist() == [1, 2, 3, 4]
    assert dataset.y.tolist() == [0, 1, 1, 0]
    assert len(dataset.bundle.train_paths) == 2
    assert len(dataset.bundle.source_hashes) == 2


def test_loader_accepts_all_missing_feature(tmp_path: Path) -> None:
    train = _write_csv(tmp_path / "train.csv", "all_missing,value,target\n,1,0\n,2,1\n")

    dataset = load_dataset(train, _config())

    assert dataset.X["all_missing"].isna().all()


def test_unknown_test_category_is_preserved_without_training_contamination(tmp_path: Path) -> None:
    train = _write_csv(tmp_path / "train.csv", "value,category,target\n1,a,0\n2,b,1\n")
    test = _write_csv(tmp_path / "test.csv", "value,category\n3,unseen\n")

    dataset = load_dataset(train, _config(test_path=test))

    assert dataset.test_X is not None
    assert dataset.test_X["category"].tolist() == ["unseen"]
    assert "unseen" not in set(dataset.X["category"])
    assert dataset.bundle.n_test_rows == 1


def test_loader_preserves_test_ids_and_feature_order(tmp_path: Path) -> None:
    train = _write_csv(tmp_path / "train.csv", "id,a,b,target\nt1,1,x,0\nt2,2,y,1\n")
    test = _write_csv(tmp_path / "test.csv", "b,id,a\nz,s1,3\n")

    dataset = load_dataset(train, _config(test_path=test, id_column="id"))

    assert dataset.test_X is not None
    assert list(dataset.test_X.columns) == ["a", "b"]
    assert dataset.test_row_ids is not None
    assert dataset.test_row_ids.tolist() == ["s1"]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("a,label\n1,0\n", "target column is absent"),
        ("a,target\n1,\n2,1\n", "target column contains 1 missing"),
    ],
)
def test_loader_rejects_invalid_target(tmp_path: Path, content: str, message: str) -> None:
    train = _write_csv(tmp_path / "train.csv", content)

    with pytest.raises(DataValidationError, match=message):
        load_dataset(train, _config())


def test_loader_rejects_inconsistent_training_schema(tmp_path: Path) -> None:
    first = _write_csv(tmp_path / "first.csv", "a,b,target\n1,x,0\n")
    second = _write_csv(tmp_path / "second.csv", "a,c,target\n2,y,1\n")

    with pytest.raises(DataValidationError, match="schema mismatch"):
        load_dataset([first, second], _config())


@pytest.mark.parametrize(
    "test_content",
    [
        "a,target\n2,1\n",
        "wrong\n2\n",
        "a,unexpected\n2,x\n",
    ],
)
def test_loader_rejects_unsafe_test_schema(tmp_path: Path, test_content: str) -> None:
    train = _write_csv(tmp_path / "train.csv", "a,target\n1,0\n3,1\n")
    test = _write_csv(tmp_path / "test.csv", test_content)

    with pytest.raises(DataValidationError):
        load_dataset(train, _config(test_path=test))


def test_source_hash_changes_when_source_changes(tmp_path: Path) -> None:
    train = _write_csv(tmp_path / "train.csv", "a,target\n1,0\n2,1\n")
    first = load_dataset(train, _config()).bundle.source_hashes[str(train)]

    _write_csv(train, "a,target\n1,0\n3,1\n")
    second = load_dataset(train, _config()).bundle.source_hashes[str(train)]

    assert first != second


def test_loader_rejects_duplicate_paths(tmp_path: Path) -> None:
    train = _write_csv(tmp_path / "train.csv", "a,target\n1,0\n2,1\n")

    with pytest.raises(DataValidationError, match="duplicates"):
        load_dataset([train, train], _config())


def test_loader_rejects_empty_or_non_csv_input(tmp_path: Path) -> None:
    empty = _write_csv(tmp_path / "empty.csv", "")
    text = _write_csv(tmp_path / "train.txt", "a,target\n1,0\n")

    with pytest.raises(DataValidationError, match="empty"):
        load_dataset(empty, _config())
    with pytest.raises(DataValidationError, match=r"only \.csv"):
        load_dataset(text, _config())


def test_loader_does_not_mutate_runtime_data_via_bundle_serialization(tmp_path: Path) -> None:
    train = _write_csv(tmp_path / "train.csv", "a,b,target\n1,x,0\n2,y,1\n")
    dataset = load_dataset(train, _config())

    serialized = dataset.bundle.to_json()

    assert "DataFrame" not in serialized
    assert '"target":[' not in serialized
    assert pd.api.types.is_integer_dtype(dataset.X["a"])

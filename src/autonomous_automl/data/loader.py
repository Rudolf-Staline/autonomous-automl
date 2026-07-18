"""Safe CSV loading, schema validation, hashing, and target isolation."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from autonomous_automl.contracts import AutoMLConfig, DatasetBundle
from autonomous_automl.data.models import LoadedDataset
from autonomous_automl.utils.errors import DataValidationError
from autonomous_automl.utils.hashing import sha256_file

_SUPPORTED_DELIMITERS = ",;\t|"


class DataLoader:
    """Load only explicitly supplied CSV sources into a leakage-safe bundle."""

    def load(
        self,
        train_paths: str | Path | Sequence[str | Path],
        config: AutoMLConfig,
    ) -> LoadedDataset:
        """Load, concatenate, validate, and split one or more training CSV files."""
        paths = _normalize_paths(train_paths)
        frames = [_read_csv(path) for path in paths]
        train = _concatenate_training_frames(frames, paths)
        _validate_training_schema(train, config)

        source_hashes = {str(path): sha256_file(path) for path in paths}
        test_path = config.test_path
        test = None
        if test_path is not None:
            normalized_test_path = _validate_csv_path(test_path)
            if normalized_test_path in paths:
                raise DataValidationError("test_path must be distinct from every training path")
            test = _read_csv(normalized_test_path)
            source_hashes[str(normalized_test_path)] = sha256_file(normalized_test_path)
            test_path = normalized_test_path

        y = _get_series(train, config.target).copy()
        train = train.drop(columns=[config.target])
        reserved = _reserved_training_columns(config)
        row_ids = _extract_optional(train, config.id_column)
        groups = _extract_optional(train, config.group_column)
        predefined_folds = _extract_optional(train, config.predefined_fold_column)
        time_values = _get_series(train, config.time_column).copy() if config.time_column else None

        feature_columns = [str(column) for column in train.columns if column not in reserved]
        if not feature_columns:
            raise DataValidationError("no feature columns remain after removing reserved columns")
        X = train.loc[:, feature_columns].copy()

        test_X: pd.DataFrame | None = None
        test_row_ids: pd.Series | None = None
        if test is not None:
            test_X, test_row_ids = _prepare_test_frame(test, feature_columns, config)

        excluded_columns = sorted(column for column in reserved if column in frames[0].columns)
        bundle = DatasetBundle(
            train_paths=paths,
            test_path=test_path,
            target=config.target,
            feature_columns=feature_columns,
            excluded_columns=excluded_columns,
            n_rows=len(X),
            n_test_rows=None if test_X is None else len(test_X),
            row_id_column=config.id_column,
            source_hashes=source_hashes,
        )
        return LoadedDataset(
            X=X,
            y=y,
            test_X=test_X,
            row_ids=row_ids,
            test_row_ids=test_row_ids,
            groups=groups,
            time_values=time_values,
            predefined_folds=predefined_folds,
            bundle=bundle,
        )


def load_dataset(
    train_paths: str | Path | Sequence[str | Path],
    config: AutoMLConfig,
) -> LoadedDataset:
    """Convenience wrapper around :class:`DataLoader`."""
    return DataLoader().load(train_paths, config)


def _normalize_paths(train_paths: str | Path | Sequence[str | Path]) -> list[Path]:
    raw_paths: Sequence[str | Path] = (
        [train_paths] if isinstance(train_paths, str | Path) else train_paths
    )
    if not raw_paths:
        raise DataValidationError("at least one training CSV path is required")
    paths = [_validate_csv_path(path) for path in raw_paths]
    if len(paths) != len(set(paths)):
        raise DataValidationError("training CSV paths must not contain duplicates")
    return paths


def _validate_csv_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.exists():
        raise DataValidationError(f"CSV file does not exist: {candidate}")
    if not candidate.is_file():
        raise DataValidationError(f"CSV path is not a regular file: {candidate}")
    if candidate.suffix.lower() != ".csv":
        raise DataValidationError(f"only .csv input files are supported: {candidate}")
    return candidate


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        delimiter = _detect_delimiter(path)
        frame = pd.read_csv(
            path,
            sep=delimiter,
            encoding="utf-8-sig",
            low_memory=False,
            on_bad_lines="error",
        )
    except UnicodeDecodeError as error:
        raise DataValidationError(f"CSV must be UTF-8 encoded: {path}") from error
    except (EmptyDataError, ParserError, csv.Error) as error:
        raise DataValidationError(f"CSV could not be parsed: {path}: {error}") from error
    except OSError as error:
        raise DataValidationError(f"CSV could not be read: {path}: {error}") from error

    if frame.empty:
        raise DataValidationError(f"CSV contains no data rows: {path}")
    if frame.columns.empty:
        raise DataValidationError(f"CSV contains no columns: {path}")
    columns = [str(column) for column in frame.columns]
    if any(not column.strip() for column in columns):
        raise DataValidationError(f"CSV contains a blank column name: {path}")
    if len(columns) != len(set(columns)):
        raise DataValidationError(f"CSV contains duplicate column names: {path}")
    frame.columns = columns
    return frame


def _detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(65_536)
    if not sample.strip():
        raise DataValidationError(f"CSV is empty: {path}")
    first_line = sample.splitlines()[0]
    counts = {delimiter: first_line.count(delimiter) for delimiter in _SUPPORTED_DELIMITERS}
    delimiter, count = max(counts.items(), key=lambda item: item[1])
    if count == 0:
        raise DataValidationError(f"CSV delimiter could not be detected: {path}")
    try:
        sniffed = csv.Sniffer().sniff(sample, delimiters=_SUPPORTED_DELIMITERS).delimiter
    except csv.Error:
        return delimiter
    return sniffed


def _concatenate_training_frames(frames: list[pd.DataFrame], paths: list[Path]) -> pd.DataFrame:
    expected = list(frames[0].columns)
    expected_set = set(expected)
    aligned = [frames[0]]
    for frame, path in zip(frames[1:], paths[1:], strict=True):
        actual_set = set(frame.columns)
        if actual_set != expected_set:
            missing = sorted(expected_set - actual_set)
            extra = sorted(actual_set - expected_set)
            raise DataValidationError(
                f"training CSV schema mismatch for {path}; missing={missing}, extra={extra}"
            )
        aligned.append(frame.loc[:, expected])
    return pd.concat(aligned, axis=0, ignore_index=True)


def _validate_training_schema(frame: pd.DataFrame, config: AutoMLConfig) -> None:
    if config.target not in frame.columns:
        raise DataValidationError(f"target column is absent from training data: {config.target}")
    target = _get_series(frame, config.target)
    missing_count = int(target.isna().sum())
    if missing_count:
        raise DataValidationError(
            f"target column contains {missing_count} missing values; remove or label those rows"
        )
    for role, column in (
        ("id", config.id_column),
        ("group", config.group_column),
        ("time", config.time_column),
        ("predefined fold", config.predefined_fold_column),
    ):
        if column is not None and column not in frame.columns:
            raise DataValidationError(f"configured {role} column is absent: {column}")


def _reserved_training_columns(config: AutoMLConfig) -> set[str]:
    return {
        column
        for column in (config.id_column, config.group_column, config.predefined_fold_column)
        if column is not None
    }


def _extract_optional(frame: pd.DataFrame, column: str | None) -> pd.Series | None:
    if column is None:
        return None
    return _get_series(frame, column).copy()


def _get_series(frame: pd.DataFrame, column: str) -> pd.Series:
    value = frame[column]
    if isinstance(value, pd.DataFrame):
        raise DataValidationError(f"column name is ambiguous because it is duplicated: {column}")
    return value


def _prepare_test_frame(
    test: pd.DataFrame,
    feature_columns: list[str],
    config: AutoMLConfig,
) -> tuple[pd.DataFrame, pd.Series | None]:
    if config.target in test.columns:
        raise DataValidationError("test CSV must not contain the target column")

    if config.id_column is not None and config.id_column not in test.columns:
        raise DataValidationError(
            f"configured id column is absent from test data: {config.id_column}"
        )
    test_row_ids = _extract_optional(test, config.id_column)

    missing = sorted(set(feature_columns).difference(test.columns))
    allowed_extras = {
        column
        for column in (config.id_column, config.group_column, config.predefined_fold_column)
        if column is not None
    }
    extra = sorted(set(test.columns).difference(feature_columns).difference(allowed_extras))
    if missing or extra:
        raise DataValidationError(f"test CSV schema mismatch; missing={missing}, extra={extra}")
    return test.loc[:, feature_columns].copy(), test_row_ids

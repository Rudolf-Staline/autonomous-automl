"""Fold-safe scikit-learn preprocessing components and factories."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

_MISSING_TOKEN = "__MISSING__"


class FiniteValueCleaner(TransformerMixin, BaseEstimator):
    """Convert numeric inputs to float and replace infinities with missing values."""

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: object | None = None,
    ) -> FiniteValueCleaner:
        frame = _as_frame(X)
        self.n_features_in_ = frame.shape[1]
        self.feature_names_in_ = np.asarray([str(column) for column in frame.columns], dtype=object)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self, "n_features_in_")
        frame = _as_frame(X, expected_columns=self.feature_names_in_)
        if frame.shape[1] != self.n_features_in_:
            raise ValueError("numeric feature count changed between fit and transform")
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        return np.asarray(numeric.replace([np.inf, -np.inf], np.nan), dtype=np.float64)

    def get_feature_names_out(self, input_features: object | None = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        if input_features is None:
            return self.feature_names_in_.copy()
        return np.asarray(input_features, dtype=object)


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """Encode each category with training-fold frequency; unknown values map to zero."""

    def __init__(self, *, normalize: bool = True) -> None:
        self.normalize = normalize

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: object | None = None,
    ) -> FrequencyEncoder:
        frame = _as_frame(X)
        self.n_features_in_ = frame.shape[1]
        self.feature_names_in_ = np.asarray([str(column) for column in frame.columns], dtype=object)
        self.mappings_: list[dict[object, float]] = []
        for column in frame.columns:
            normalized = _normalize_missing(_get_series(frame, column))
            counts = normalized.value_counts(dropna=False, normalize=self.normalize)
            self.mappings_.append({key: float(value) for key, value in counts.items()})
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["mappings_", "feature_names_in_"])
        frame = _as_frame(X, expected_columns=self.feature_names_in_)
        if frame.shape[1] != self.n_features_in_:
            raise ValueError("categorical feature count changed between fit and transform")
        columns = []
        for index, column in enumerate(frame.columns):
            normalized = _normalize_missing(_get_series(frame, column))
            encoded = normalized.map(self.mappings_[index]).fillna(0.0)
            columns.append(np.asarray(encoded, dtype=np.float64))
        if not columns:
            return np.empty((len(frame), 0), dtype=np.float64)
        return np.column_stack(columns)

    def get_feature_names_out(self, input_features: object | None = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        names = self.feature_names_in_ if input_features is None else np.asarray(input_features)
        return np.asarray([f"{name}__frequency" for name in names], dtype=object)


class CategoricalValueCleaner(TransformerMixin, BaseEstimator):
    """Normalize categorical columns to strings while preserving missing values."""

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: object | None = None,
    ) -> CategoricalValueCleaner:
        frame = _as_frame(X)
        self.n_features_in_ = frame.shape[1]
        self.feature_names_in_ = np.asarray([str(column) for column in frame.columns], dtype=object)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        frame = _as_frame(X, expected_columns=self.feature_names_in_)
        if frame.shape[1] != self.n_features_in_:
            raise ValueError("categorical feature count changed between fit and transform")
        normalized = frame.astype("string")
        return normalized.to_numpy(dtype=object, na_value=np.nan)

    def get_feature_names_out(self, input_features: object | None = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        if input_features is None:
            return self.feature_names_in_.copy()
        return np.asarray(input_features, dtype=object)


class DateFeatureExtractor(TransformerMixin, BaseEstimator):
    """Extract deterministic calendar features relative to an explicit epoch."""

    def __init__(self, *, reference_date: str = "1970-01-01") -> None:
        self.reference_date = reference_date

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: object | None = None,
    ) -> DateFeatureExtractor:
        frame = _as_frame(X)
        reference = pd.Timestamp(self.reference_date, tz="UTC")
        if pd.isna(reference):
            raise ValueError("reference_date must be parseable")
        self.reference_timestamp_ = reference
        self.n_features_in_ = frame.shape[1]
        self.feature_names_in_ = np.asarray([str(column) for column in frame.columns], dtype=object)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["reference_timestamp_", "feature_names_in_"])
        frame = _as_frame(X, expected_columns=self.feature_names_in_)
        if frame.shape[1] != self.n_features_in_:
            raise ValueError("datetime feature count changed between fit and transform")
        outputs: list[np.ndarray] = []
        for column in frame.columns:
            parsed = cast(
                pd.Series,
                pd.to_datetime(frame[column], errors="coerce", utc=True, format="mixed"),
            )
            outputs.extend(
                [
                    np.asarray(parsed.dt.year, dtype=np.float64),
                    np.asarray(parsed.dt.month, dtype=np.float64),
                    np.asarray(parsed.dt.day, dtype=np.float64),
                    np.asarray(parsed.dt.dayofweek, dtype=np.float64),
                    np.asarray(
                        (parsed - self.reference_timestamp_).dt.total_seconds() / 86_400.0,
                        dtype=np.float64,
                    ),
                ]
            )
        if not outputs:
            return np.empty((len(frame), 0), dtype=np.float64)
        return np.column_stack(outputs)

    def get_feature_names_out(self, input_features: object | None = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        names = self.feature_names_in_ if input_features is None else np.asarray(input_features)
        suffixes = ("year", "month", "day", "dayofweek", "days_since_1970")
        return np.asarray(
            [f"{name}__{suffix}" for name in names for suffix in suffixes],
            dtype=object,
        )


def build_numeric_transformer(imputer: str, scaler: str) -> Pipeline:
    """Build a numeric preprocessing pipeline from stable strategy names."""
    steps: list[tuple[str, object]] = [("finite", FiniteValueCleaner())]
    if imputer == "none":
        pass
    elif imputer in {"mean", "median"}:
        steps.append(
            (
                "imputer",
                SimpleImputer(strategy=imputer, keep_empty_features=True),
            )
        )
    elif imputer == "constant":
        steps.append(
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
            )
        )
    elif imputer == "median_indicator":
        steps.append(
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            )
        )
    elif imputer == "knn":
        steps.append(("imputer", KNNImputer(n_neighbors=5, keep_empty_features=True)))
    else:
        raise ValueError(f"unknown numeric imputer strategy: {imputer}")

    if scaler == "none":
        pass
    elif scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler == "robust":
        steps.append(("scaler", RobustScaler()))
    else:
        raise ValueError(f"unknown numeric scaler strategy: {scaler}")
    return Pipeline(steps)


def build_categorical_transformer(
    imputer: str,
    encoder: str,
    *,
    dense_output: bool,
) -> Pipeline:
    """Build categorical preprocessing with safe unknown-category behavior."""
    if imputer == "constant":
        categorical_imputer = SimpleImputer(
            strategy="constant",
            fill_value=_MISSING_TOKEN,
            keep_empty_features=True,
        )
    elif imputer == "most_frequent":
        categorical_imputer = SimpleImputer(strategy="most_frequent", keep_empty_features=True)
    else:
        raise ValueError(f"unknown categorical imputer strategy: {imputer}")

    if encoder == "one_hot":
        categorical_encoder: object = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=not dense_output,
        )
    elif encoder == "ordinal":
        categorical_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
        )
    elif encoder == "frequency":
        categorical_encoder = FrequencyEncoder()
    else:
        raise ValueError(f"unknown categorical encoder strategy: {encoder}")
    return Pipeline(
        [
            ("clean", CategoricalValueCleaner()),
            ("imputer", categorical_imputer),
            ("encoder", categorical_encoder),
        ]
    )


def build_datetime_transformer(strategy: str) -> Pipeline:
    """Build calendar extraction followed by fold-local missing-value handling."""
    if strategy != "calendar":
        raise ValueError(f"unknown datetime transformer strategy: {strategy}")
    return Pipeline(
        [
            ("extract", DateFeatureExtractor()),
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
        ]
    )


def _as_frame(
    X: pd.DataFrame | np.ndarray,
    *,
    expected_columns: np.ndarray | None = None,
) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy(deep=False)
    values = np.asarray(X)
    if values.ndim != 2:
        raise ValueError("transformer input must be two-dimensional")
    columns: object = expected_columns if expected_columns is not None else range(values.shape[1])
    return pd.DataFrame(values, columns=columns)


def _normalize_missing(series: pd.Series) -> pd.Series:
    normalized = series.astype(object).copy()
    return normalized.where(pd.notna(normalized), _MISSING_TOKEN)


def _get_series(frame: pd.DataFrame, column: object) -> pd.Series:
    value = frame[column]
    if isinstance(value, pd.DataFrame):
        raise ValueError(f"duplicate categorical column is ambiguous: {column}")
    return value


__all__ = [
    "CategoricalValueCleaner",
    "DateFeatureExtractor",
    "FiniteValueCleaner",
    "FrequencyEncoder",
    "build_categorical_transformer",
    "build_datetime_transformer",
    "build_numeric_transformer",
]

"""Deterministic, conservative type inference for tabular features."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import cast

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

_DATE_VALUE_PATTERN = re.compile(r"^(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})")
_DATE_NAME_PATTERN = re.compile(r"(?:^|_)(?:date|time|timestamp|datetime)(?:_|$)", re.I)


class InferredType(StrEnum):
    """Feature types understood by the V1 pipeline grammar."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    ALL_MISSING = "all_missing"

    @property
    def primary_role(self) -> InferredType:
        """Return the executable role used for an all-missing feature."""
        if self is InferredType.ALL_MISSING:
            return InferredType.CATEGORICAL
        return self


def infer_column_type(series: pd.Series, *, name: str | None = None) -> InferredType:
    """Infer one feature type without mutating or fitting on other datasets.

    String dates require recognisable date values. A date-like column name is a
    supporting signal, not enough on its own, which avoids coercing arbitrary IDs.
    Long, mostly unique strings are treated as free text and can be excluded by the
    V1 pipeline planner.
    """
    non_missing = series.dropna()
    if non_missing.empty:
        return InferredType.ALL_MISSING

    dtype = series.dtype
    if is_bool_dtype(dtype) or _contains_only_booleans(non_missing):
        return InferredType.BOOLEAN
    if is_datetime64_any_dtype(dtype):
        return InferredType.DATETIME
    if is_numeric_dtype(dtype):
        return InferredType.NUMERIC
    if isinstance(dtype, pd.CategoricalDtype):
        return InferredType.CATEGORICAL
    if is_object_dtype(dtype) or is_string_dtype(dtype):
        values = non_missing.astype("string")
        if _looks_like_datetime(values, name=name or str(series.name or "")):
            return InferredType.DATETIME
        if _looks_like_free_text(values):
            return InferredType.TEXT
        return InferredType.CATEGORICAL
    return InferredType.CATEGORICAL


def infer_dataframe_types(frame: pd.DataFrame) -> dict[str, InferredType]:
    """Infer every feature type in stable input-column order."""
    return {
        str(column): infer_column_type(cast(pd.Series, frame[column]), name=str(column))
        for column in frame.columns
    }


def _contains_only_booleans(series: pd.Series) -> bool:
    values = set(series.unique().tolist())
    return bool(values) and values.issubset({True, False})


def _looks_like_datetime(values: pd.Series, *, name: str) -> bool:
    sample = values.iloc[: min(len(values), 256)].str.strip()
    if sample.empty:
        return False
    pattern_ratio = float(sample.str.match(_DATE_VALUE_PATTERN, na=False).mean())
    if pattern_ratio < 0.8:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    parse_ratio = float(parsed.notna().mean())
    return parse_ratio >= 0.9 or (bool(_DATE_NAME_PATTERN.search(name)) and parse_ratio >= 0.8)


def _looks_like_free_text(values: pd.Series) -> bool:
    lengths = values.str.len()
    median_length = float(lengths.median())
    mean_length = float(lengths.mean())
    unique_ratio = float(values.nunique(dropna=True) / len(values))
    return median_length >= 40.0 or (mean_length >= 24.0 and unique_ratio >= 0.5)


__all__ = ["InferredType", "infer_column_type", "infer_dataframe_types"]

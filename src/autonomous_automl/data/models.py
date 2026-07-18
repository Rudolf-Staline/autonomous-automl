"""Runtime-only containers for loaded tabular data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from autonomous_automl.contracts import DatasetBundle


@dataclass(slots=True)
class LoadedDataset:
    """DataFrames used by execution services, deliberately absent from manifests."""

    X: pd.DataFrame
    y: pd.Series
    test_X: pd.DataFrame | None
    row_ids: pd.Series | None
    test_row_ids: pd.Series | None
    groups: pd.Series | None
    time_values: pd.Series | None
    predefined_folds: pd.Series | None
    bundle: DatasetBundle

    def __post_init__(self) -> None:
        if len(self.X) != len(self.y):
            raise ValueError("X and y must contain the same number of rows")
        if self.bundle.target in self.X.columns:
            raise ValueError("the target must never be present in X")
        if list(self.X.columns) != self.bundle.feature_columns:
            raise ValueError("runtime features must match DatasetBundle.feature_columns")
        if self.test_X is not None and list(self.test_X.columns) != list(self.X.columns):
            raise ValueError("test_X must use the exact training feature order")

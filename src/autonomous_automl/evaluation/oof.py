"""Out-of-fold prediction accumulation by stable row position."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from autonomous_automl.contracts import TaskType
from autonomous_automl.evaluation.metrics import align_probabilities


class OOFAccumulator:
    """Average repeated validation predictions without ever storing targets."""

    def __init__(
        self,
        n_rows: int,
        task: TaskType,
        *,
        classes: np.ndarray[Any, Any] | list[object] | None = None,
        row_ids: pd.Series | None = None,
    ) -> None:
        if n_rows <= 0:
            raise ValueError("OOF accumulation requires at least one row")
        if task is TaskType.AUTO:
            raise ValueError("OOF accumulation requires a resolved task")
        if row_ids is not None and len(row_ids) != n_rows:
            raise ValueError("row IDs must match the number of OOF rows")
        self.n_rows = n_rows
        self.task = task
        self.classes = None if classes is None else np.asarray(classes, dtype=object)
        self.row_ids = None if row_ids is None else row_ids.reset_index(drop=True).copy()
        self._counts = np.zeros(n_rows, dtype=np.int64)
        self._prediction_sums = np.zeros(n_rows, dtype=np.float64)
        self._probability_sums = (
            np.zeros((n_rows, len(self.classes)), dtype=np.float64)
            if self.classes is not None
            else None
        )
        self._probability_counts = np.zeros(n_rows, dtype=np.int64)
        self._label_votes: list[Counter[object]] = [Counter() for _ in range(n_rows)]

        if task in {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        } and (self.classes is None or len(self.classes) < 2):
            raise ValueError("classification OOF predictions require at least two classes")

    @property
    def counts(self) -> np.ndarray[Any, np.dtype[np.int64]]:
        """Return a defensive copy of validation observation counts."""
        return self._counts.copy()

    @property
    def is_complete(self) -> bool:
        """Whether every source row has at least one validation prediction."""
        return bool((self._counts > 0).all())

    def add_regression(
        self,
        positions: list[int] | np.ndarray[Any, Any],
        predictions: np.ndarray[Any, Any] | list[float],
    ) -> None:
        """Add one fold of numeric predictions."""
        if self.task is not TaskType.REGRESSION:
            raise ValueError("regression predictions cannot be added to classification OOF")
        indices = self._validated_positions(positions)
        values = np.asarray(predictions, dtype=np.float64).reshape(-1)
        if len(values) != len(indices) or not bool(np.isfinite(values).all()):
            raise ValueError("regression OOF predictions must be finite and match positions")
        self._prediction_sums[indices] += values
        self._counts[indices] += 1

    def add_classification(
        self,
        positions: list[int] | np.ndarray[Any, Any],
        predictions: np.ndarray[Any, Any] | list[object],
        probabilities: np.ndarray[Any, Any] | None = None,
        model_classes: np.ndarray[Any, Any] | list[object] | None = None,
    ) -> np.ndarray[Any, np.dtype[np.float64]] | None:
        """Add labels and, when supplied, globally aligned probabilities."""
        if self.task is TaskType.REGRESSION or self.classes is None:
            raise ValueError("classification predictions cannot be added to regression OOF")
        indices = self._validated_positions(positions)
        labels = np.asarray(predictions, dtype=object).reshape(-1)
        if len(labels) != len(indices):
            raise ValueError("classification OOF predictions must match positions")
        self._counts[indices] += 1
        for position, label in zip(indices, labels, strict=True):
            if not any(label == known_class for known_class in self.classes):
                raise ValueError("classification OOF prediction is not a known class")
            self._label_votes[int(position)][label] += 1
        if probabilities is None and model_classes is None:
            return None
        if probabilities is None or model_classes is None:
            raise ValueError("probabilities and model classes must be supplied together")
        aligned = align_probabilities(probabilities, model_classes, self.classes)
        if len(aligned) != len(indices):
            raise ValueError("classification OOF probabilities must match positions")
        assert self._probability_sums is not None
        self._probability_sums[indices] += aligned
        self._probability_counts[indices] += 1
        return aligned

    def to_frame(self) -> pd.DataFrame:
        """Materialize source-ordered OOF values without the target column."""
        frame = pd.DataFrame({"row_position": np.arange(self.n_rows, dtype=np.int64)})
        if self.row_ids is not None:
            frame.insert(1, "row_id", self.row_ids.to_numpy(copy=True))
        covered = self._counts > 0
        if self.task is TaskType.REGRESSION:
            averaged = np.full(self.n_rows, np.nan, dtype=np.float64)
            averaged[covered] = self._prediction_sums[covered] / self._counts[covered]
            frame["prediction"] = averaged
        else:
            assert self.classes is not None
            assert self._probability_sums is not None
            averaged_probabilities = np.full_like(self._probability_sums, np.nan)
            probability_covered = self._probability_counts > 0
            averaged_probabilities[probability_covered] = (
                self._probability_sums[probability_covered]
                / self._probability_counts[probability_covered, None]
            )
            predictions = np.empty(self.n_rows, dtype=object)
            predictions[:] = pd.NA
            for position in np.flatnonzero(covered):
                votes = self._label_votes[position]
                highest_vote = max(votes.values())
                predictions[position] = next(
                    label for label in self.classes if votes[label] == highest_vote
                )
            frame["prediction"] = predictions
            for class_index in range(len(self.classes)):
                frame[f"probability_{class_index}"] = averaged_probabilities[:, class_index]
        frame["oof_count"] = self._counts
        return frame

    def _validated_positions(
        self, positions: list[int] | np.ndarray[Any, Any]
    ) -> np.ndarray[Any, np.dtype[np.int64]]:
        indices = np.asarray(positions, dtype=np.int64).reshape(-1)
        if len(indices) == 0:
            raise ValueError("OOF positions cannot be empty")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("OOF positions cannot contain duplicates within a fold")
        if bool((indices < 0).any()) or bool((indices >= self.n_rows).any()):
            raise IndexError("OOF position is outside the source dataset")
        return indices


__all__ = ["OOFAccumulator"]

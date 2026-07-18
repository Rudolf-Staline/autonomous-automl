"""Explainable, training-only heuristics for likely target leakage."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from autonomous_automl.contracts import (
    DatasetProfile,
    FindingSeverity,
    LeakageFinding,
    LeakageFindingType,
    LeakageReport,
    TaskType,
)
from autonomous_automl.data import LoadedDataset

_POST_OUTCOME_PATTERN = re.compile(
    r"(?:^|_)(?:outcome|result|final|resolved|closed|after|post|approved|paid)(?:_|$)",
    re.I,
)
_TARGET_NAME_PATTERN = re.compile(
    r"(?:^|_)(?:target|label|response|prediction|ground_truth)(?:_|$)",
    re.I,
)


@dataclass(frozen=True, slots=True)
class LeakageDetector:
    """Detect high-value leakage signals without inspecting the test set."""

    near_match_threshold: float = 0.98
    correlation_threshold: float = 0.999

    def __post_init__(self) -> None:
        if not 0.9 <= self.near_match_threshold < 1.0:
            raise ValueError("near_match_threshold must be in [0.9, 1.0)")
        if not 0.9 <= self.correlation_threshold <= 1.0:
            raise ValueError("correlation_threshold must be in [0.9, 1.0]")

    def detect(
        self,
        dataset: LoadedDataset,
        profile: DatasetProfile,
    ) -> LeakageReport:
        """Return findings and explicit exclusions based on training aggregates."""
        findings: list[LeakageFinding] = []
        excluded: set[str] = set()
        target_name = _normalize_name(dataset.bundle.target)

        for column in dataset.X.columns:
            name = str(column)
            series = _get_series(dataset.X, name)
            match_ratio = _equality_ratio(series, dataset.y)
            if match_ratio == 1.0:
                findings.append(
                    LeakageFinding(
                        finding_type=LeakageFindingType.TARGET_COPY,
                        severity=FindingSeverity.CRITICAL,
                        confidence=1.0,
                        column=name,
                        evidence_summary={"matching_ratio": 1.0, "n_rows": len(series)},
                        action="exclude_before_validation",
                    )
                )
                excluded.add(name)
            elif match_ratio >= self.near_match_threshold:
                findings.append(
                    LeakageFinding(
                        finding_type=LeakageFindingType.NEAR_TARGET_COPY,
                        severity=FindingSeverity.CRITICAL,
                        confidence=match_ratio,
                        column=name,
                        evidence_summary={"matching_ratio": match_ratio, "n_rows": len(series)},
                        action="exclude_before_validation",
                    )
                )
                excluded.add(name)
            elif profile.inferred_task == TaskType.REGRESSION:
                correlation = _finite_correlation(series, dataset.y)
                if correlation is not None and abs(correlation) >= self.correlation_threshold:
                    confidence = min(1.0, abs(correlation))
                    findings.append(
                        LeakageFinding(
                            finding_type=LeakageFindingType.NEAR_TARGET_COPY,
                            severity=FindingSeverity.CRITICAL,
                            confidence=confidence,
                            column=name,
                            evidence_summary={
                                "absolute_correlation": abs(correlation),
                                "n_rows": len(series),
                            },
                            action="exclude_before_validation",
                        )
                    )
                    excluded.add(name)

            normalized = _normalize_name(name)
            if _looks_target_named(normalized, target_name):
                findings.append(
                    LeakageFinding(
                        finding_type=LeakageFindingType.SUSPICIOUS_NAME,
                        severity=FindingSeverity.WARNING,
                        confidence=0.8,
                        column=name,
                        evidence_summary={"name_signal": "target_semantics"},
                        action="review_before_search",
                    )
                )
            if _POST_OUTCOME_PATTERN.search(normalized):
                findings.append(
                    LeakageFinding(
                        finding_type=LeakageFindingType.POST_OUTCOME,
                        severity=FindingSeverity.WARNING,
                        confidence=0.75,
                        column=name,
                        evidence_summary={"name_signal": "post_outcome_semantics"},
                        action="review_before_search",
                    )
                )

        for column in profile.id_candidates:
            if column not in dataset.X.columns:
                continue
            confidence = max(0.8, profile.unique_ratios.get(column, 0.0))
            findings.append(
                LeakageFinding(
                    finding_type=LeakageFindingType.IDENTIFIER,
                    severity=FindingSeverity.WARNING,
                    confidence=min(confidence, 1.0),
                    column=column,
                    evidence_summary={
                        "unique_ratio": profile.unique_ratios.get(column, 0.0),
                        "cardinality": profile.cardinalities.get(column, 0),
                    },
                    action="exclude_identifier_from_features",
                )
            )
            excluded.add(column)

        if profile.duplicate_row_count > 0:
            findings.append(
                LeakageFinding(
                    finding_type=LeakageFindingType.DUPLICATE_ROWS,
                    severity=FindingSeverity.WARNING,
                    confidence=1.0,
                    evidence_summary={
                        "duplicate_row_count": profile.duplicate_row_count,
                        "potential_cross_fold_rows": (profile.potential_cross_fold_duplicate_count),
                    },
                    action="audit_duplicate_groups_after_fold_materialization",
                )
            )

        return LeakageReport(findings=findings, excluded_columns=sorted(excluded))


def detect_leakage(dataset: LoadedDataset, profile: DatasetProfile) -> LeakageReport:
    """Run the default deterministic leakage detector."""
    return LeakageDetector().detect(dataset, profile)


def _get_series(frame: pd.DataFrame, column: str) -> pd.Series:
    value = frame[column]
    if isinstance(value, pd.DataFrame):
        raise ValueError(f"duplicate feature name is ambiguous: {column}")
    return value


def _equality_ratio(feature: pd.Series, target: pd.Series) -> float:
    if len(feature) != len(target) or len(feature) == 0:
        return 0.0
    comparison = feature.reset_index(drop=True).eq(target.reset_index(drop=True))
    return float(comparison.fillna(False).mean())


def _finite_correlation(feature: pd.Series, target: pd.Series) -> float | None:
    if not is_numeric_dtype(feature.dtype) or not is_numeric_dtype(target.dtype):
        return None
    numeric_feature = cast(pd.Series, pd.to_numeric(feature, errors="coerce"))
    numeric_target = cast(pd.Series, pd.to_numeric(target, errors="coerce"))
    feature_values = np.asarray(numeric_feature, dtype=np.float64)
    target_values = np.asarray(numeric_target, dtype=np.float64)
    mask = np.isfinite(feature_values) & np.isfinite(target_values)
    if int(mask.sum()) < 3:
        return None
    selected_feature = feature_values[mask]
    selected_target = target_values[mask]
    if np.ptp(selected_feature) == 0 or np.ptp(selected_target) == 0:
        return None
    correlation = float(np.corrcoef(selected_feature, selected_target)[0, 1])
    return correlation if math.isfinite(correlation) else None


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")


def _looks_target_named(column_name: str, target_name: str) -> bool:
    if _TARGET_NAME_PATTERN.search(column_name):
        return True
    return bool(target_name and target_name in column_name and column_name != target_name)


__all__ = ["LeakageDetector", "detect_leakage"]

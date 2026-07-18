"""Stable string enumerations used by persisted contracts."""

from __future__ import annotations

from enum import StrEnum


class TaskType(StrEnum):
    AUTO = "auto"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


class MetricName(StrEnum):
    AUTO = "auto"
    ROC_AUC = "roc_auc"
    AVERAGE_PRECISION = "average_precision"
    LOG_LOSS = "log_loss"
    ACCURACY = "accuracy"
    BALANCED_ACCURACY = "balanced_accuracy"
    F1 = "f1"
    MACRO_F1 = "macro_f1"
    RMSE = "rmse"
    MAE = "mae"
    R2 = "r2"


class OptimizationProfile(StrEnum):
    ACCURACY = "accuracy"
    BALANCED = "balanced"
    FAST = "fast"


class TrialStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PRUNED = "pruned"
    INTERRUPTED = "interrupted"


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class FindingSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class LeakageFindingType(StrEnum):
    TARGET_COPY = "target_copy"
    NEAR_TARGET_COPY = "near_target_copy"
    SUSPICIOUS_NAME = "suspicious_name"
    IDENTIFIER = "identifier"
    DUPLICATE_ROWS = "duplicate_rows"
    GROUP_OVERLAP = "group_overlap"
    TEMPORAL_ORDER = "temporal_order"
    POST_OUTCOME = "post_outcome"


class NotebookValidationStatus(StrEnum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"

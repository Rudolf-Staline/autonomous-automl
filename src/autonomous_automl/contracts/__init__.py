"""Validated and JSON-serializable public contracts."""

from autonomous_automl.contracts.base import ContractModel
from autonomous_automl.contracts.config import AutoMLConfig
from autonomous_automl.contracts.dataset import DatasetBundle, DatasetProfile
from autonomous_automl.contracts.enums import (
    FindingSeverity,
    LeakageFindingType,
    MetricName,
    NotebookValidationStatus,
    OptimizationProfile,
    RunStatus,
    TaskType,
    TrialStatus,
)
from autonomous_automl.contracts.evaluation import FoldResult, TrialResult
from autonomous_automl.contracts.leakage import LeakageFinding, LeakageReport
from autonomous_automl.contracts.pipeline import FidelitySpec, PipelineSpec
from autonomous_automl.contracts.run import NotebookValidation, RunManifest, RunResult
from autonomous_automl.contracts.validation import FoldAssignment, ValidationAudit, ValidationPlan

__all__ = [
    "AutoMLConfig",
    "ContractModel",
    "DatasetBundle",
    "DatasetProfile",
    "FidelitySpec",
    "FindingSeverity",
    "FoldAssignment",
    "FoldResult",
    "LeakageFinding",
    "LeakageFindingType",
    "LeakageReport",
    "MetricName",
    "NotebookValidation",
    "NotebookValidationStatus",
    "OptimizationProfile",
    "PipelineSpec",
    "RunManifest",
    "RunResult",
    "RunStatus",
    "TaskType",
    "TrialResult",
    "TrialStatus",
    "ValidationAudit",
    "ValidationPlan",
]

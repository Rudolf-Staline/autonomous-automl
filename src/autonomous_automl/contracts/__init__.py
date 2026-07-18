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
from autonomous_automl.contracts.trust import (
    MetricDirection,
    ObservedTrustGap,
    PipelineSelectionExplanation,
    ResumeEvidence,
    RuntimeTelemetry,
    SearchStopReason,
    SelectionContextItem,
    SourceHashEvidence,
    TrustCertificate,
    TrustCertificateStatus,
    TrustGapProtocol,
    TrustGapStatus,
    VerificationEvidence,
    VerificationStatus,
    derive_certificate_status,
)
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
    "MetricDirection",
    "MetricName",
    "NotebookValidation",
    "NotebookValidationStatus",
    "ObservedTrustGap",
    "OptimizationProfile",
    "PipelineSelectionExplanation",
    "PipelineSpec",
    "ResumeEvidence",
    "RunManifest",
    "RunResult",
    "RunStatus",
    "RuntimeTelemetry",
    "SearchStopReason",
    "SelectionContextItem",
    "SourceHashEvidence",
    "TaskType",
    "TrialResult",
    "TrialStatus",
    "TrustCertificate",
    "TrustCertificateStatus",
    "TrustGapProtocol",
    "TrustGapStatus",
    "ValidationAudit",
    "ValidationPlan",
    "VerificationEvidence",
    "VerificationStatus",
    "derive_certificate_status",
]

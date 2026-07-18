"""Cross-validation, metrics, and out-of-fold prediction services."""

from autonomous_automl.evaluation.evaluator import EvaluationOutcome, PipelineEvaluator
from autonomous_automl.evaluation.explanation import explain_pipeline_selection
from autonomous_automl.evaluation.metrics import (
    MetricDefinition,
    MetricValue,
    align_probabilities,
    ordered_classes,
    resolve_metric,
    score_metric,
)
from autonomous_automl.evaluation.oof import OOFAccumulator
from autonomous_automl.evaluation.trust_gap import (
    TrustGapEvaluation,
    compute_observed_trust_gap,
    disabled_trust_gap,
    metric_direction,
)

__all__ = [
    "EvaluationOutcome",
    "MetricDefinition",
    "MetricValue",
    "OOFAccumulator",
    "PipelineEvaluator",
    "TrustGapEvaluation",
    "align_probabilities",
    "compute_observed_trust_gap",
    "disabled_trust_gap",
    "explain_pipeline_selection",
    "metric_direction",
    "ordered_classes",
    "resolve_metric",
    "score_metric",
]

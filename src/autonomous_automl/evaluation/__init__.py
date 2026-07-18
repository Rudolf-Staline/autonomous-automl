"""Cross-validation, metrics, and out-of-fold prediction services."""

from autonomous_automl.evaluation.evaluator import EvaluationOutcome, PipelineEvaluator
from autonomous_automl.evaluation.metrics import (
    MetricDefinition,
    MetricValue,
    align_probabilities,
    ordered_classes,
    resolve_metric,
    score_metric,
)
from autonomous_automl.evaluation.oof import OOFAccumulator

__all__ = [
    "EvaluationOutcome",
    "MetricDefinition",
    "MetricValue",
    "OOFAccumulator",
    "PipelineEvaluator",
    "align_probabilities",
    "ordered_classes",
    "resolve_metric",
    "score_metric",
]

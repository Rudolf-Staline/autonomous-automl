"""Pipeline grammar, compatibility, fingerprinting, and materialization."""

from autonomous_automl.pipelines.compatibility import (
    CompatibilityResult,
    check_pipeline_compatibility,
    compatible_specs,
)
from autonomous_automl.pipelines.factory import build_model, build_pipeline, build_preprocessor
from autonomous_automl.pipelines.fingerprint import pipeline_fingerprint
from autonomous_automl.pipelines.grammar import PipelineGrammar, generate_initial_candidates

__all__ = [
    "CompatibilityResult",
    "PipelineGrammar",
    "build_model",
    "build_pipeline",
    "build_preprocessor",
    "check_pipeline_compatibility",
    "compatible_specs",
    "generate_initial_candidates",
    "pipeline_fingerprint",
]

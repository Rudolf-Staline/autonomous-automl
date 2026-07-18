"""Stable identity for deduplication and exact-once resume semantics."""

from autonomous_automl.contracts import PipelineSpec
from autonomous_automl.utils.hashing import sha256_json


def pipeline_fingerprint(spec: PipelineSpec) -> str:
    """Hash the canonical PipelineSpec JSON representation."""
    return sha256_json(spec.to_json_value())


__all__ = ["pipeline_fingerprint"]

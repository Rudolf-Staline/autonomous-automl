"""Domain exception hierarchy for Autonomous AutoML."""

from __future__ import annotations


class AutoMLError(Exception):
    """Base class for errors that can be handled by the application."""


class DataValidationError(AutoMLError):
    """Raised when input data fails validation."""


class ConfigurationError(AutoMLError):
    """Raised when an AutoML configuration is invalid."""


class IncompatiblePipelineError(AutoMLError):
    """Raised when a pipeline specification cannot handle a dataset."""


class TrialTimeoutError(AutoMLError):
    """Raised when an individual trial exceeds its time allowance."""


class TrialMemoryError(AutoMLError):
    """Raised when an individual trial exceeds its memory allowance."""


class ModelUnavailableError(AutoMLError):
    """Raised when a requested optional model is not installed or usable."""


class MetricError(AutoMLError):
    """Raised when a metric is unknown or incompatible with a task."""


class ArtifactValidationError(AutoMLError):
    """Raised when persisted artifacts fail integrity or replay validation."""


class ResumeError(AutoMLError):
    """Raised when a persisted run cannot be resumed safely."""


class TrackingError(AutoMLError):
    """Raised when transactional run state is missing or inconsistent."""


class RunExecutionError(AutoMLError):
    """Raised when a run cannot produce a usable final pipeline."""


class PlannedInterruption(AutoMLError):
    """Raised after a requested checkpoint interruption used to demonstrate resume."""

    def __init__(self, run_directory: str) -> None:
        super().__init__(f"run intentionally interrupted; resume from {run_directory}")
        self.run_directory = run_directory

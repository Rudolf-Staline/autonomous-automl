"""Public package for the Autonomous AutoML engine."""

from importlib.metadata import PackageNotFoundError, version

from autonomous_automl.api import AutoMLRun
from autonomous_automl.contracts import AutoMLConfig

try:
    __version__ = version("autonomous-automl")
except PackageNotFoundError:  # pragma: no cover - only for an uninstalled source tree
    __version__ = "0.1.0"

__all__ = ["AutoMLConfig", "AutoMLRun", "__version__"]

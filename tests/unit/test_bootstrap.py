"""Bootstrap acceptance tests."""

from typer.testing import CliRunner

import autonomous_automl
from autonomous_automl.cli.app import app


def test_package_is_importable() -> None:
    assert autonomous_automl.__version__


def test_cli_help_lists_doctor() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "doctor" in result.stdout


def test_fit_help_lists_every_supported_task() -> None:
    result = CliRunner().invoke(app, ["fit", "--help"])

    assert result.exit_code == 0
    assert "multiclass" in result.stdout


def test_doctor_reports_runtime() -> None:
    result = CliRunner().invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "autonomous-automl" in result.stdout
    assert "python 3.12" in result.stdout

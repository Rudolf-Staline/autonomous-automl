"""Run the same deterministic scenario as ``automl demo`` from Python."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--budget", default="6s")
    arguments = parser.parse_args()
    command = ["automl", "demo", "--budget", arguments.budget]
    if arguments.output_root is not None:
        command.extend(["--output-root", str(arguments.output_root)])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)

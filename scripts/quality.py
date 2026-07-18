"""Run the repository quality gates in a stable order."""

from __future__ import annotations

import subprocess
import sys

COMMANDS: tuple[tuple[str, ...], ...] = (
    ("ruff", "check", "."),
    ("ruff", "format", "--check", "."),
    ("pyright",),
    ("pytest",),
)


def main() -> int:
    """Run every quality command, stopping on the first failure."""
    for command in COMMANDS:
        print(f"+ {' '.join(command)}", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())

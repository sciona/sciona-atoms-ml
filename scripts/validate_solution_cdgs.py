#!/usr/bin/env python3
"""Compatibility wrapper for the centralized solution CDG validator."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    validator = (
        Path(__file__).resolve().parents[2]
        / "sciona-atoms"
        / "scripts"
        / "validate_solution_cdgs.py"
    )
    if not validator.exists():
        sys.stderr.write(f"Central solution CDG validator not found: {validator}\n")
        raise SystemExit(1)
    runpy.run_path(str(validator), run_name="__main__")


if __name__ == "__main__":
    main()

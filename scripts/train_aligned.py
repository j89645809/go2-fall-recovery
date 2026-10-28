#!/usr/bin/env python3
"""Register the isolated Go2 recovery task, then invoke the normal trainer."""

from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.tasks.velocity.config.go2_recovery_aligned  # noqa: F401


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("train.py")), run_name="__main__")

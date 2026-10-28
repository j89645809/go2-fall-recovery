#!/usr/bin/env python3
"""Targeted continuation from model_13798 with 50% balanced retention replay."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["PYTHON_FRNET_TARGETED_RECOVERY"] = "1"

import src.tasks.velocity.config.go2_recovery_aligned  # noqa: E402,F401
from scripts.train import TrainConfig, launch_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--gpus", nargs="+", type=int, default=[1, 2])
    parser.add_argument(
        "--load-run",
        default="2026-09-14_16-56-16_targeted_full_joint_stairs_slope_1000",
    )
    parser.add_argument("--load-checkpoint", default="model_13798.pt")
    args = parser.parse_args()

    task = "Unitree-Go2-Recovery-FRNet-Aligned"
    cfg = TrainConfig.from_task(task)
    cfg.agent.max_iterations = args.iterations
    cfg.agent.run_name = "decoupled_allfolded_right_edge_replay_400"
    cfg.agent.resume = True
    cfg.agent.load_run = args.load_run
    cfg.agent.load_checkpoint = args.load_checkpoint
    cfg.agent.logger = "wandb"
    cfg.agent.wandb_project = "mjlab"
    cfg.agent.wandb_tags = (
        "go2-recovery",
        "all-folded",
        "right-fall",
        "terrain-edge",
        "retention-replay-50pct",
    )
    cfg = replace(cfg, gpu_ids=args.gpus)
    launch_training(task, cfg)


if __name__ == "__main__":
    main()

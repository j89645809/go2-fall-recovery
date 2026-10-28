#!/usr/bin/env python3
"""Short safety continuation from model_14196 for folded, calf and left-fall failures."""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PYTHON_FRNET_TARGETED_RECOVERY"] = "1"
os.environ["PYTHON_FRNET_TARGETED_V2"] = "1"
os.environ["FRNET_SOURCE_ITERATION_OFFSET"] = "14196"

import src.tasks.velocity.config.go2_recovery_aligned  # noqa: E402,F401
from scripts.train import TrainConfig, launch_training  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--gpus", nargs="+", type=int, default=[2])
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument(
        "--run-name", default="acceptance_v6_left_folded_calf_safety_60"
    )
    parser.add_argument("--logger", choices=("tensorboard", "wandb"), default="wandb")
    args = parser.parse_args()

    task = "Unitree-Go2-Recovery-FRNet-Aligned"
    cfg = TrainConfig.from_task(task)
    cfg.agent.max_iterations = args.iterations
    cfg.agent.save_interval = 20
    cfg.agent.run_name = args.run_name
    cfg.agent.resume = True
    cfg.agent.load_run = "2026-09-18_06-45-12_acceptance_v5_safety_finetune_100"
    cfg.agent.load_checkpoint = "model_14196.pt"
    cfg.agent.logger = args.logger
    cfg.agent.wandb_project = "mjlab"
    cfg.agent.upload_model = True
    cfg.agent.wandb_tags = (
        "safety-finetune-v6",
        "left-fall-45pct",
        "all-folded-60pct",
        "early-calf-speed",
        "strict-position-margin",
        "teacher-anchor",
    )
    cfg.env.scene.num_envs = args.num_envs
    cfg = replace(cfg, gpu_ids=args.gpus)
    launch_training(task, cfg)


if __name__ == "__main__":
    main()

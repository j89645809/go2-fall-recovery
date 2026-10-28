#!/usr/bin/env python3
"""Targeted continuation run for the expanded fallen-pose domain."""
from pathlib import Path
import os
import sys
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.tasks.velocity.config.go2_recovery_aligned  # noqa: F401
from scripts.train import TrainConfig, launch_training

if __name__ == "__main__":
    task = "Unitree-Go2-Recovery-FRNet-Aligned"
    cfg = TrainConfig.from_task(task)
    cfg.agent.max_iterations = 1000
    cfg.agent.run_name = "targeted_full_joint_stairs_slope_1000"
    cfg.agent.resume = True
    cfg.agent.load_run = "2026-09-14_12-12-05_randomized_birth_correct_contact_multigpu_300"
    cfg.agent.load_checkpoint = "model_12799.pt"
    cfg.agent.logger = "wandb"
    cfg.agent.wandb_project = "mjlab"
    cfg.agent.wandb_tags = ("go2-recovery", "full-joint-range", "stairs-slope", "targeted-finetune")
    cfg = replace(cfg, gpu_ids="all")
    launch_training(task, cfg)

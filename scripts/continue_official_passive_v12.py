#!/usr/bin/env python3
"""Adapt model 14264 to the repository's official Go2 passive dynamics."""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

os.environ["PYTHON_FRNET_TARGETED_RECOVERY"]="1"
os.environ["PYTHON_FRNET_TARGETED_V2"]="1"
os.environ["PYTHON_FRNET_STATIC_LIBRARY_TARGET_FRACTION"]="0.5"
os.environ["PYTHON_FRNET_PREDICTIVE_ALL_JOINTS"]="1"
os.environ["PYTHON_FRNET_PREDICTIVE_LATCH"]="0"
os.environ["FRNET_PREDICTIVE_GENERAL_OFFSET"]="0"
os.environ["FRNET_PREDICTIVE_CALF_OFFSET"]="0"
os.environ["PYTHON_FRNET_OFFICIAL_PASSIVE_DYNAMICS"]="1"
os.environ["FRNET_SOURCE_ITERATION_OFFSET"]="14264"

import src.tasks.velocity.config.go2_recovery_aligned  # noqa: E402,F401
from scripts.train import TrainConfig,launch_training  # noqa: E402


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--iterations",type=int,default=20)
    parser.add_argument("--gpus",nargs="+",type=int,default=[3])
    parser.add_argument("--num-envs",type=int,default=4096)
    parser.add_argument(
        "--run-name",default="acceptance_v12_official_passive_adaptation_20"
    )
    parser.add_argument("--logger",choices=("tensorboard","wandb"),default="wandb")
    args=parser.parse_args()

    task="Unitree-Go2-Recovery-FRNet-Aligned"
    cfg=TrainConfig.from_task(task)
    cfg.agent.max_iterations=args.iterations
    cfg.agent.save_interval=2
    cfg.agent.run_name=args.run_name
    cfg.agent.resume=True
    cfg.agent.load_run="2026-09-28_07-41-28_acceptance_v11_source14259_anchor_20"
    cfg.agent.load_checkpoint="model_14264.pt"
    cfg.agent.logger=args.logger
    cfg.agent.wandb_project="mjlab"
    cfg.agent.upload_model=True
    cfg.agent.algorithm.teacher_checkpoint=(
        "logs/rsl_rl/go2_recovery_frnet_aligned/"
        "2026-09-28_07-41-28_acceptance_v11_source14259_anchor_20/"
        "model_14264.pt"
    )
    cfg.agent.algorithm.teacher_pull=0.005
    cfg.agent.wandb_tags=(
        "FR-Net","Go2","official-passive-dynamics","armature-0.01",
        "joint-damping-0.1","frictionloss-0.2","zero-offset-projection",
        "source-14264-anchor","fold55-static-library","mirror-consistency",
    )
    cfg.env.scene.num_envs=args.num_envs
    cfg=replace(cfg,gpu_ids=args.gpus)
    launch_training(task,cfg)


if __name__=="__main__":
    main()

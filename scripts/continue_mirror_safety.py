#!/usr/bin/env python3
"""Short mirror-consistency adaptation from model_14196 with safety C active."""
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
os.environ["FRNET_SOURCE_ITERATION_OFFSET"]="14196"

import src.tasks.velocity.config.go2_recovery_aligned  # noqa: E402,F401
from scripts.train import TrainConfig,launch_training  # noqa: E402


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--iterations",type=int,default=40)
    parser.add_argument("--gpus",nargs="+",type=int,default=[1,3])
    parser.add_argument("--num-envs",type=int,default=4096)
    parser.add_argument("--run-name",default="acceptance_v7_mirror_safety_c_40")
    parser.add_argument("--logger",choices=("tensorboard","wandb"),default="wandb")
    args=parser.parse_args()

    task="Unitree-Go2-Recovery-FRNet-Aligned"
    cfg=TrainConfig.from_task(task)
    cfg.agent.max_iterations=args.iterations
    cfg.agent.save_interval=10
    cfg.agent.run_name=args.run_name
    cfg.agent.resume=True
    cfg.agent.load_run="2026-09-18_06-45-12_acceptance_v5_safety_finetune_100"
    cfg.agent.load_checkpoint="model_14196.pt"
    cfg.agent.logger=args.logger
    cfg.agent.wandb_project="mjlab"
    cfg.agent.upload_model=True
    cfg.agent.wandb_tags=(
        "FR-Net","Go2","mirror-consistency","safety-layer-c",
        "teacher-anchor","model-14196",
    )
    cfg.env.scene.num_envs=args.num_envs
    cfg=replace(cfg,gpu_ids=args.gpus)
    launch_training(task,cfg)


if __name__=="__main__":
    main()

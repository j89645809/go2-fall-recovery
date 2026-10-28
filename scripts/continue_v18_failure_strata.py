#!/usr/bin/env python3
"""Short safety fine-tune from model 14286 on the release static library."""
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
os.environ["PYTHON_FRNET_STATIC_LIBRARY_TARGET_FRACTION"]="0.60"
os.environ["PYTHON_FRNET_STATIC_LIBRARY_PATH"]=(
    "/home/zy/old_data/unitree_rl_mjlab/run_logs/static_library_eval/"
    "v13_fold55_plus_stairs_front_diagFRRL.json"
)
os.environ["PYTHON_FRNET_PREDICTIVE_ALL_JOINTS"]="1"
os.environ["PYTHON_FRNET_PREDICTIVE_LATCH"]="0"
os.environ["FRNET_PREDICTIVE_GENERAL_OFFSET"]="0"
os.environ["FRNET_PREDICTIVE_CALF_OFFSET"]="0"
os.environ["PYTHON_FRNET_OFFICIAL_PASSIVE_DYNAMICS"]="1"
os.environ["PYTHON_FRNET_HARD_JOINT_LIMITS"]="1"
os.environ["FRNET_JOINT_LIMIT_MARGIN"]="0.05"
os.environ["PYTHON_FRNET_ACTUATOR_POSITION_BARRIER"]="1"
os.environ["PYTHON_FRNET_ACTUATOR_POSITION_DAMPING"]="0"
os.environ["FRNET_ACTUATOR_POSITION_BRAKE_GAIN"]="0.25"
os.environ["FRNET_ACTUATOR_POSITION_DAMPING_MARGIN"]="0.12"
os.environ["FRNET_CONTACT_PRELOAD_DEPTH_M"]="0"
os.environ["FRNET_SOURCE_ITERATION_OFFSET"]="14286"

import src.tasks.velocity.config.go2_recovery_aligned  # noqa: E402,F401
from scripts.train import TrainConfig,launch_training  # noqa: E402


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--iterations",type=int,default=20)
    parser.add_argument("--gpus",nargs="+",type=int,default=[1])
    parser.add_argument("--num-envs",type=int,default=4096)
    parser.add_argument("--run-name",default="acceptance_v18_failure_strata_safety_20")
    parser.add_argument("--logger",choices=("tensorboard","wandb"),default="wandb")
    args=parser.parse_args()

    task="Unitree-Go2-Recovery-FRNet-Aligned"
    cfg=TrainConfig.from_task(task)
    cfg.agent.max_iterations=args.iterations
    cfg.agent.save_interval=2
    cfg.agent.run_name=args.run_name
    cfg.agent.resume=True
    cfg.agent.load_run="2026-09-28_09-34-37_acceptance_v13_jointlimit_damping_adaptation_20"
    cfg.agent.load_checkpoint="model_14286.pt"
    cfg.agent.logger=args.logger
    cfg.agent.wandb_project="mjlab"
    cfg.agent.upload_model=True
    cfg.agent.algorithm.teacher_checkpoint=(
        "logs/rsl_rl/go2_recovery_frnet_aligned/"
        "2026-09-28_09-34-37_acceptance_v13_jointlimit_damping_adaptation_20/"
        "model_14286.pt"
    )
    cfg.agent.algorithm.teacher_pull=0.01
    cfg.agent.wandb_tags=(
        "FR-Net","Go2","hard-joint-limits","position-damping",
        "official-passive-dynamics","zero-contact-preload",
        "release-validated-static-library","source-14286-anchor",
        "mirror-consistency",
    )
    cfg.env.scene.num_envs=args.num_envs
    cfg=replace(cfg,gpu_ids=args.gpus)
    launch_training(task,cfg)


if __name__=="__main__":
    main()

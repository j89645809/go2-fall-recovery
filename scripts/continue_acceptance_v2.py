#!/usr/bin/env python3
"""Acceptance-aligned continuation from model_13798 on GPU 1/2."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['PYTHON_FRNET_TARGETED_RECOVERY']='1'
os.environ['PYTHON_FRNET_TARGETED_V2']='1'
os.environ['FRNET_SOURCE_ITERATION_OFFSET']='13798'

import src.tasks.velocity.config.go2_recovery_aligned  # noqa:E402,F401
from scripts.train import TrainConfig,launch_training  # noqa:E402


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--iterations',type=int,default=300)
    parser.add_argument('--gpus',nargs='+',type=int,default=[1,2])
    parser.add_argument('--num-envs',type=int,default=4096)
    parser.add_argument('--run-name',default='acceptance_v2_local_contact_teacher60_300')
    parser.add_argument('--logger',choices=('tensorboard','wandb'),default='tensorboard')
    parser.add_argument('--upload-model',action='store_true')
    args=parser.parse_args()
    task='Unitree-Go2-Recovery-FRNet-Aligned'
    cfg=TrainConfig.from_task(task)
    cfg.agent.max_iterations=args.iterations
    cfg.agent.run_name=args.run_name
    cfg.agent.resume=True
    cfg.agent.load_run='2026-09-14_16-56-16_targeted_full_joint_stairs_slope_1000'
    cfg.agent.load_checkpoint='model_13798.pt'
    cfg.agent.logger=args.logger
    cfg.agent.wandb_project='mjlab'
    cfg.agent.upload_model=args.upload_model
    cfg.agent.wandb_tags=('local-contact-reset','onboard-hold','acceptance-reward',
                          'teacher-anchor','retention-60pct')
    cfg.env.scene.num_envs=args.num_envs
    cfg=replace(cfg,gpu_ids=args.gpus)
    launch_training(task,cfg)


if __name__=='__main__': main()

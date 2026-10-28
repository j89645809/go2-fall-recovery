#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['PYTHON_FRNET_TARGETED_RECOVERY']='1'
os.environ['PYTHON_FRNET_TARGETED_V2']='1'
os.environ['FRNET_SOURCE_ITERATION_OFFSET']='14097'
import src.tasks.velocity.config.go2_recovery_aligned  # noqa
from scripts.train import TrainConfig, launch_training
p=argparse.ArgumentParser()
p.add_argument('--iterations',type=int,default=100)
p.add_argument('--gpus',nargs='+',type=int,default=[5])
p.add_argument('--num-envs',type=int,default=4096)
p.add_argument('--run-name',default='acceptance_v5_safety_finetune_100')
p.add_argument('--logger',default='wandb')
a=p.parse_args(); task='Unitree-Go2-Recovery-FRNet-Aligned'
cfg=TrainConfig.from_task(task)
cfg.agent.max_iterations=a.iterations
cfg.agent.run_name=a.run_name
cfg.agent.resume=True
cfg.agent.load_run='2026-09-17_03-20-54_acceptance_v5_source_iter_reward_contact_300'
cfg.agent.load_checkpoint='model_14097.pt'
cfg.agent.logger=a.logger
cfg.agent.wandb_project='mjlab'
cfg.agent.wandb_tags=('safety-finetune','teacher-anchor','retention-60pct')
cfg.env.scene.num_envs=a.num_envs
cfg=replace(cfg,gpu_ids=a.gpus)
launch_training(task,cfg)

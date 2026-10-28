#!/usr/bin/env python3
from __future__ import annotations
import os,sys
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['PYTHON_FRNET_TARGETED_RECOVERY']='1'
os.environ['PYTHON_FRNET_TARGETED_V2']='1'
os.environ['FRNET_SOURCE_ITERATION_OFFSET']='14196'
import src.tasks.velocity.config.go2_recovery_aligned
from scripts.train import TrainConfig,launch_training
task='Unitree-Go2-Recovery-FRNet-Aligned'
cfg=TrainConfig.from_task(task)
cfg.agent.max_iterations=100
cfg.agent.run_name='acceptance_v5_slew015_finetune_100'
cfg.agent.resume=True
cfg.agent.load_run='2026-09-18_06-45-12_acceptance_v5_safety_finetune_100'
cfg.agent.load_checkpoint='model_14196.pt'
cfg.agent.logger='wandb'
cfg.agent.wandb_project='mjlab'
cfg.agent.wandb_tags=('safety-envelope-015','teacher-anchor','retention-60pct')
cfg.env.scene.num_envs=4096
cfg=replace(cfg,gpu_ids=[0])
launch_training(task,cfg)

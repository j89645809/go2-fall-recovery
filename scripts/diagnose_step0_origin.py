#!/usr/bin/env python3
"""Separate reset/contact motion from the first policy command."""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import torch
import mjlab.tasks  # noqa: F401
import src.tasks.velocity.config.go2_recovery_aligned  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from scripts.vectorized_decoupled_eval import (
    POSES, TERRAINS, assign_cases, build_cases,
)
from src.tasks.velocity.config.go2_recovery_aligned.official_go2_limits import (
    POSITION_LOWER, POSITION_UPPER, VELOCITY_LIMIT,
)


def snapshot(robot, device):
    velocity_limit=torch.tensor(VELOCITY_LIMIT,device=device)
    lower=torch.tensor(POSITION_LOWER,device=device)
    upper=torch.tensor(POSITION_UPPER,device=device)
    ratio=robot.data.joint_vel.abs()/velocity_limit
    calf=ratio[:,(2,5,8,11)].amax(dim=-1)
    excess=torch.maximum(
        lower-robot.data.joint_pos,robot.data.joint_pos-upper
    ).clamp_min(0.0).amax(dim=-1)
    return calf,excess


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--depths",nargs="+",type=float,default=[0.002])
    args=parser.parse_args()

    torch.manual_seed(20260916)
    torch.cuda.manual_seed_all(20260916)
    cfg=load_env_cfg("Unitree-Go2-Recovery-FRNet-Aligned",play=True)
    cfg.seed=20260916
    cfg.scene.num_envs=16
    cfg.viewer=None
    cfg.events.pop("settle_fallen",None)
    env=ManagerBasedRlEnv(cfg=cfg,device=args.device)
    wrapped=RslRlVecEnvWrapper(env)
    agent_cfg=load_rl_cfg("Unitree-Go2-Recovery-FRNet-Aligned")
    runner_cls=(
        load_runner_cls("Unitree-Go2-Recovery-FRNet-Aligned")
        or MjlabOnPolicyRunner
    )
    runner=runner_cls(wrapped,asdict(agent_cfg),device=args.device)
    runner.load(
        args.checkpoint,load_cfg={"actor":True},strict=True,map_location=args.device
    )
    policy=runner.get_inference_policy(device=args.device)
    cases=build_cases(list(TERRAINS),list(POSES),["all_folded"],1)
    robot=env.scene["robot"]

    for depth in args.depths:
        os.environ["FRNET_CONTACT_PRELOAD_DEPTH_M"]=str(depth)
        # Branch A: leave the reset-synchronised PD target unchanged and
        # advance the four 5 ms substeps used by one 50 Hz policy step.
        env.reset()
        contact,*_=assign_cases(env,cases)
        hold_calf=[];hold_pos=[]
        for _ in range(4):
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(dt=env.physics_dt)
            env.sim.forward();env.sim.sense()
            calf,pos=snapshot(robot,args.device)
            hold_calf.append(calf.clone());hold_pos.append(pos.clone())

        # Branch B: reinstall the identical cases and execute the first policy
        # action through the normal manager/decimation path.
        env.reset()
        policy_contact,*_=assign_cases(env,cases)
        obs=wrapped.get_observations()
        with torch.inference_mode():
            action=policy(obs)
        wrapped.step(action)
        policy_calf,policy_pos=snapshot(robot,args.device)

        hold_peak=torch.stack(hold_calf).amax(dim=0)
        hold_pos_peak=torch.stack(hold_pos).amax(dim=0)
        print({
            "depth_m":depth,
            "cases":len(cases),
            "hold_initial_contact":int(contact.sum().item()),
            "policy_initial_contact":int(policy_contact.sum().item()),
            "hold_velocity_unsafe":int((hold_peak>1.0).sum().item()),
            "hold_position_unsafe":int((hold_pos_peak>0.02).sum().item()),
            "hold_peak_velocity_ratio":float(hold_peak.max().item()),
            "hold_peak_position_excess_rad":float(hold_pos_peak.max().item()),
            "policy_velocity_unsafe":int((policy_calf>1.0).sum().item()),
            "policy_position_unsafe":int((policy_pos>0.02).sum().item()),
            "policy_peak_velocity_ratio":float(policy_calf.max().item()),
            "policy_peak_position_excess_rad":float(policy_pos.max().item()),
            "hold_substep_velocity_unsafe":[
                int((value>1.0).sum().item()) for value in hold_calf
            ],
        },flush=True)
    env.close()


if __name__=="__main__":
    main()

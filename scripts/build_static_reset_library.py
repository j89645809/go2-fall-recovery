#!/usr/bin/env python3
"""Build release-validated, zero-velocity recovery reset states offline."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch
import mjlab.tasks  # noqa: F401
import src.tasks.velocity.config.go2_recovery_aligned  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from scripts.vectorized_decoupled_eval import (
    EVALUATION_TERRAIN_SEED, MODES, POSES, TERRAINS, assign_cases, build_cases,
)
from src.tasks.velocity.config.go2_recovery_aligned.aligned_reset import _contact_found
from src.tasks.velocity.config.go2_recovery_aligned.official_go2_limits import (
    JOINT_NAMES, POSITION_LOWER, POSITION_UPPER, VELOCITY_LIMIT,
)
from src.tasks.velocity.config.go2_recovery_aligned.recovery_acceptance import STAND_JOINT_TARGET


def fall_direction_drift_rad(initial_quat, candidate_quat):
    """Angle between body-frame gravity vectors, independent of world yaw."""
    gravity=torch.zeros_like(initial_quat[..., :3])
    gravity[..., 2]=-1.0
    initial_direction=quat_apply_inverse(initial_quat,gravity)
    candidate_direction=quat_apply_inverse(candidate_quat,gravity)
    dot=(initial_direction*candidate_direction).sum(dim=-1).clamp(-1.0,1.0)
    return torch.acos(dot)


def static_birth_mode_mask(cases, joint_positions):
    """Ensure serialized states retain every fold named by their birth mode."""
    result=torch.ones(len(cases),dtype=torch.bool,device=joint_positions.device)
    q=joint_positions[:len(cases)]
    folded=(q[:,(1,4,7,10)] > 2.50) & (q[:,(2,5,8,11)] < -2.45)
    required={
        "single_FL":(0,),"single_FR":(1,),"single_RL":(2,),"single_RR":(3,),
        "diagonal_FL_RR":(0,3),"diagonal_FR_RL":(1,2),
    }
    for index,case in enumerate(cases):
        mode=case["birth_mode"]
        if mode == "all_folded":
            result[index]=folded[index].sum() >= 3
        elif mode in required:
            result[index]=folded[index,list(required[mode])].all()
    return result


def apply_fold_alpha(cases, alpha):
    """Interpolate all-folded cases from the calibrated stand pose."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("fold alpha must be in [0, 1]")
    output=copy.deepcopy(cases)
    for case in output:
        case["fold_alpha"]=float(alpha)
        if case["birth_mode"] != "all_folded":
            continue
        folded=case["joint_positions_rad"]
        if alpha == 0.0:
            case["joint_positions_rad"]=list(STAND_JOINT_TARGET)
        elif alpha == 1.0:
            case["joint_positions_rad"]=folded
        else:
            case["joint_positions_rad"]=[
                float(stand+alpha*(target-stand))
                for stand,target in zip(STAND_JOINT_TARGET,folded,strict=True)
            ]
    return output


def apply_symmetric_joint_jitter(cases, scale):
    """Search nearby poses while preserving exact left/right symmetry."""
    if scale < 0.0:
        raise ValueError("joint jitter must be non-negative")
    output=copy.deepcopy(cases)
    lower=[value+0.03 for value in POSITION_LOWER]
    upper=[value-0.03 for value in POSITION_UPPER]
    for case in output:
        case["symmetric_joint_jitter_rad"]=float(scale)
        if scale == 0.0:
            continue
        rng=random.Random(int(case["seed"])+9173)
        q=list(case["joint_positions_rad"])
        for left,right in ((0,3),(6,9)):
            hip=0.5*(q[left]-q[right])+rng.uniform(-scale,scale)
            thigh=0.5*(q[left+1]+q[right+1])+rng.uniform(-1.5*scale,1.5*scale)
            calf=0.5*(q[left+2]+q[right+2])+rng.uniform(-scale,scale)
            q[left]=hip;q[right]=-hip
            q[left+1]=thigh;q[right+1]=thigh
            q[left+2]=calf;q[right+2]=calf
        case["joint_positions_rad"]=[
            min(max(value,lo),hi)
            for value,lo,hi in zip(q,lower,upper,strict=True)
        ]
    return output


def release_validation_mask(
    initial_valid, final_contact, peak_velocity_ratio, peak_position_excess,
    final_root_speed, final_joint_speed, *, root_speed_limit=0.05,
    joint_speed_limit=0.10,
):
    """Strict acceptance gate for a reusable static reset state."""
    return (
        initial_valid.bool()
        & final_contact.bool()
        & (peak_velocity_ratio <= 1.0)
        & (peak_position_excess <= 0.02)
        & (final_root_speed <= root_speed_limit)
        & (final_joint_speed <= joint_speed_limit)
    )


def validate_release(env, control_target, steps):
    """Connect the stored PD preload and measure the ensuing free dynamics."""
    robot=env.scene["robot"]
    velocity=control_target.new_tensor(VELOCITY_LIMIT)
    lower=control_target.new_tensor(POSITION_LOWER)
    upper=control_target.new_tensor(POSITION_UPPER)
    peak_velocity=(robot.data.joint_vel.abs()/velocity).amax(dim=-1)
    peak_position=torch.maximum(
        lower-robot.data.joint_pos,robot.data.joint_pos-upper
    ).clamp_min(0).amax(dim=-1)
    zero_velocity=torch.zeros_like(control_target)
    for _ in range(steps):
        robot.set_joint_position_target(control_target)
        robot.set_joint_velocity_target(zero_velocity)
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(dt=env.physics_dt)
        speed_ratio=(robot.data.joint_vel.abs()/velocity).amax(dim=-1)
        excess=torch.maximum(
            lower-robot.data.joint_pos,robot.data.joint_pos-upper
        ).clamp_min(0).amax(dim=-1)
        peak_velocity=torch.maximum(peak_velocity,speed_ratio)
        peak_position=torch.maximum(peak_position,excess)
    env.sim.forward();env.sim.sense()
    contact=_contact_found(
        env,torch.arange(env.num_envs,device=env.device),require_nonfoot=False
    )
    root_speed=robot.data.root_link_vel_w.norm(dim=-1)
    joint_speed=robot.data.joint_vel.abs().amax(dim=-1)
    return {
        "contact":contact,
        "peak_velocity_ratio":peak_velocity,
        "peak_position_excess_rad":peak_position,
        "final_root_speed":root_speed,
        "final_joint_speed":joint_speed,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--terrains",nargs="+",choices=TERRAINS,default=list(TERRAINS))
    parser.add_argument("--poses",nargs="+",choices=POSES,default=list(POSES))
    parser.add_argument("--modes",nargs="+",choices=MODES,default=list(MODES))
    parser.add_argument("--per-stratum",type=int,default=1)
    parser.add_argument("--batch-size",type=int,default=128)
    parser.add_argument("--release-steps",type=int,default=200)
    parser.add_argument("--fold-alpha",type=float,default=1.0)
    parser.add_argument("--symmetric-joint-jitter",type=float,default=0.0)
    parser.add_argument("--max-pose-drift-deg",type=float,default=40.0)
    parser.add_argument("--max-xy-drift",type=float,default=0.30)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()

    cases=apply_symmetric_joint_jitter(
        apply_fold_alpha(
            build_cases(args.terrains,args.poses,args.modes,args.per_stratum),
            args.fold_alpha,
        ),
        args.symmetric_joint_jitter,
    )
    torch.manual_seed(20260916);torch.cuda.manual_seed_all(20260916)
    cfg=load_env_cfg("Unitree-Go2-Recovery-FRNet-Aligned",play=True)
    cfg.seed=20260916;cfg.scene.num_envs=args.batch_size;cfg.viewer=None
    cfg.scene.terrain.terrain_generator.seed=EVALUATION_TERRAIN_SEED
    cfg.events.pop("settle_fallen",None)
    env=ManagerBasedRlEnv(cfg=cfg,device=args.device)
    robot=env.scene["robot"]
    if tuple(robot.joint_names) != JOINT_NAMES:
        raise ValueError("Static library joint ordering mismatch")
    action=env.action_manager.get_term("recovery")
    rows=[];attempted=0
    gate_counts=Counter()
    rejected_diagnostics=[]
    for start in range(0,len(cases),args.batch_size):
        group=cases[start:start+args.batch_size];n=len(group)
        env.reset()
        initial=assign_cases(env,group)
        initial_valid=initial[0]
        root_pose=robot.data.root_link_pose_w.clone()
        joint_pos=robot.data.joint_pos.clone()
        control_target=action.target.clone()

        first=validate_release(env,control_target,args.release_steps)
        first_mask=release_validation_mask(
            initial_valid,
            first["contact"][:n],first["peak_velocity_ratio"][:n],
            first["peak_position_excess_rad"][:n],first["final_root_speed"][:n],
            first["final_joint_speed"][:n],
        )

        # Re-zero the candidate after its first relaxation and verify that the
        # exact state to be serialized survives a second independent release.
        candidate_pose=robot.data.root_link_pose_w.clone()
        candidate_q=robot.data.joint_pos.clone()
        candidate_root=torch.cat(
            (candidate_pose,torch.zeros((env.num_envs,6),device=env.device,dtype=candidate_pose.dtype)),dim=-1
        )
        robot.write_joint_state_to_sim(candidate_q,torch.zeros_like(candidate_q))
        robot.write_root_state_to_sim(candidate_root)
        env.scene.write_data_to_sim();env.sim.forward();env.sim.sense()
        candidate_contact=_contact_found(
            env,torch.arange(env.num_envs,device=env.device),require_nonfoot=False
        )[:n]
        second=validate_release(env,control_target,args.release_steps)
        second_mask=release_validation_mask(
            candidate_contact,
            second["contact"][:n],second["peak_velocity_ratio"][:n],
            second["peak_position_excess_rad"][:n],second["final_root_speed"][:n],
            second["final_joint_speed"][:n],
        )
        # Re-run the exact serialized candidate, rather than accepting a state
        # that happened to survive only once.
        robot.write_joint_state_to_sim(candidate_q,torch.zeros_like(candidate_q))
        robot.write_root_state_to_sim(candidate_root)
        env.scene.write_data_to_sim();env.sim.forward();env.sim.sense()
        third=validate_release(env,control_target,args.release_steps)
        third_mask=release_validation_mask(
            candidate_contact,
            third["contact"][:n],third["peak_velocity_ratio"][:n],
            third["peak_position_excess_rad"][:n],third["final_root_speed"][:n],
            third["final_joint_speed"][:n],
        )
        quat_dot=(candidate_pose[:n,3:7]*root_pose[:n,3:7]).sum(dim=-1).abs().clamp(0.0,1.0)
        full_quaternion_drift=2.0*torch.acos(quat_dot)
        fall_direction_drift=fall_direction_drift_rad(
            root_pose[:n,3:7],candidate_pose[:n,3:7]
        )
        xy_drift=(candidate_pose[:n,:2]-root_pose[:n,:2]).norm(dim=-1)
        pose_preserved=(fall_direction_drift <= math.radians(args.max_pose_drift_deg)) & (
            xy_drift <= args.max_xy_drift
        )
        birth_mode_preserved=static_birth_mode_mask(group,candidate_q)
        accepted=second_mask & third_mask & pose_preserved & birth_mode_preserved
        checks={
            "initial_valid":initial_valid,
            "first_contact":first["contact"][:n],
            "first_velocity_safe":first["peak_velocity_ratio"][:n] <= 1.0,
            "first_position_safe":first["peak_position_excess_rad"][:n] <= 0.02,
            "first_root_still":first["final_root_speed"][:n] <= 0.05,
            "first_joint_still":first["final_joint_speed"][:n] <= 0.10,
            "first_all":first_mask,
            "second_contact":second["contact"][:n],
            "second_velocity_safe":second["peak_velocity_ratio"][:n] <= 1.0,
            "second_position_safe":second["peak_position_excess_rad"][:n] <= 0.02,
            "second_root_still":second["final_root_speed"][:n] <= 0.05,
            "second_joint_still":second["final_joint_speed"][:n] <= 0.10,
            "second_all":second_mask,
            "third_contact":third["contact"][:n],
            "third_velocity_safe":third["peak_velocity_ratio"][:n] <= 1.0,
            "third_position_safe":third["peak_position_excess_rad"][:n] <= 0.02,
            "third_root_still":third["final_root_speed"][:n] <= 0.05,
            "third_joint_still":third["final_joint_speed"][:n] <= 0.10,
            "third_all":third_mask,
            "pose_preserved":pose_preserved,
            "birth_mode_preserved":birth_mode_preserved,
            "accepted":accepted,
        }
        for name,mask in checks.items():
            gate_counts[name]+=int(mask.sum().item())
        origins=env.scene.terrain.env_origins
        for i,case in enumerate(group):
            attempted+=1
            if not bool(accepted[i]):
                if len(rejected_diagnostics) < 64:
                    rejected_diagnostics.append({
                        "terrain_family":case["terrain_family"],
                        "fall_pose":case["fall_pose"],
                        "birth_mode":case["birth_mode"],
                        "sample":case["sample"],
                        "first_peak_velocity_ratio":float(first["peak_velocity_ratio"][i]),
                        "first_peak_position_excess_rad":float(first["peak_position_excess_rad"][i]),
                        "first_final_root_speed":float(first["final_root_speed"][i]),
                        "first_final_joint_speed":float(first["final_joint_speed"][i]),
                        "second_peak_velocity_ratio":float(second["peak_velocity_ratio"][i]),
                        "second_peak_position_excess_rad":float(second["peak_position_excess_rad"][i]),
                        "second_final_root_speed":float(second["final_root_speed"][i]),
                        "second_final_joint_speed":float(second["final_joint_speed"][i]),
                        "third_peak_velocity_ratio":float(third["peak_velocity_ratio"][i]),
                        "third_peak_position_excess_rad":float(third["peak_position_excess_rad"][i]),
                        "third_final_root_speed":float(third["final_root_speed"][i]),
                        "third_final_joint_speed":float(third["final_joint_speed"][i]),
                        "pose_drift_rad":float(full_quaternion_drift[i]),
                        "full_quaternion_drift_rad":float(full_quaternion_drift[i]),
                        "fall_direction_drift_rad":float(fall_direction_drift[i]),
                        "xy_drift_m":float(xy_drift[i]),
                        "birth_mode_preserved":bool(birth_mode_preserved[i]),
                    })
                continue
            relative_pose=candidate_pose[i].clone()
            relative_pose[:3]-=origins[i]
            rows.append({
                **case,
                "relative_root_pose":relative_pose.tolist(),
                "joint_positions_rad":candidate_q[i].tolist(),
                "control_target_rad":control_target[i].tolist(),
                "root_velocity": [0.0]*6,
                "joint_velocity": [0.0]*12,
                "release_validation":{
                    "first_peak_velocity_ratio":float(first["peak_velocity_ratio"][i]),
                    "first_peak_position_excess_rad":float(first["peak_position_excess_rad"][i]),
                    "second_peak_velocity_ratio":float(second["peak_velocity_ratio"][i]),
                    "second_peak_position_excess_rad":float(second["peak_position_excess_rad"][i]),
                    "second_final_root_speed":float(second["final_root_speed"][i]),
                    "second_final_joint_speed":float(second["final_joint_speed"][i]),
                    "third_peak_velocity_ratio":float(third["peak_velocity_ratio"][i]),
                    "third_peak_position_excess_rad":float(third["peak_position_excess_rad"][i]),
                    "third_final_root_speed":float(third["final_root_speed"][i]),
                    "third_final_joint_speed":float(third["final_joint_speed"][i]),
                    "pose_drift_rad":float(full_quaternion_drift[i]),
                    "full_quaternion_drift_rad":float(full_quaternion_drift[i]),
                    "fall_direction_drift_rad":float(fall_direction_drift[i]),
                    "xy_drift_m":float(xy_drift[i]),
                    "birth_mode_preserved":bool(birth_mode_preserved[i]),
                },
            })
        print(json.dumps({"attempted":attempted,"accepted":len(rows)}),flush=True)
    by_mode=Counter(row["birth_mode"] for row in rows)
    by_pose=Counter(row["fall_pose"] for row in rows)
    by_terrain=Counter(row["terrain_family"] for row in rows)
    report={
        "schema_version":2,
        "joint_names":list(JOINT_NAMES),
        "release_steps":args.release_steps,
        "fold_alpha":args.fold_alpha,
        "symmetric_joint_jitter_rad":args.symmetric_joint_jitter,
        "max_pose_drift_deg":args.max_pose_drift_deg,
        "pose_preservation_metric":"body_frame_gravity_direction",
        "max_fall_direction_drift_deg":args.max_pose_drift_deg,
        "max_xy_drift_m":args.max_xy_drift,
        "physics_dt":env.physics_dt,
        "attempted":len(cases),
        "accepted":len(rows),
        "by_mode":dict(by_mode),
        "by_pose":dict(by_pose),
        "by_terrain":dict(by_terrain),
        "gate_counts":dict(gate_counts),
        "rejected_diagnostics":rejected_diagnostics,
        "rows":rows,
    }
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n")
    env.close()


if __name__ == "__main__":
    main()

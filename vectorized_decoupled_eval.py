#!/usr/bin/env python3
"""Batched native-terrain recovery evaluation with per-case safety records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path

import torch
import mjlab.tasks  # noqa: F401
import src.tasks.velocity.config.go2_recovery_aligned  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_from_euler_xyz
from src.tasks.velocity.config.go2_recovery_aligned.aligned_reset import _place_static_in_contact
from src.tasks.velocity.config.go2_recovery_aligned.official_go2_limits import (
    FRNET_EFFORT_LIMIT, JOINT_NAMES, POSITION_LOWER, POSITION_UPPER, VELOCITY_LIMIT,
)
from src.tasks.velocity.config.go2_recovery_aligned.recovery_acceptance import (
    STAND_JOINT_TARGET, foot_contact_mask, local_relative_height, physical_acceptance,
)

TERRAINS={"flat":(0,1),"stairs":(2,3,4),"slope":(5,6),"rough":(7,)}
POSES={"front":(0.,-math.pi/2),"back":(0.,math.pi/2),
       "left":(-math.pi/2,0.),"right":(math.pi/2,0.)}
MODES=("independent_random","single_FL","single_FR","single_RL","single_RR",
       "diagonal_FL_RR","diagonal_FR_RL","all_folded")
FOLDED=((0.82,2.9,-2.68),(-0.82,2.9,-2.68),
        (0.82,3.5,-2.68),(-0.82,3.5,-2.68))
EVALUATION_TERRAIN_SEED=20260916

# These variables change the simulated robot, action safety layer, or reset
# behavior. Store their effective values with every report so two files cannot
# look comparable while silently using different safety settings.
RUNTIME_ENV_DEFAULTS={
    "PYTHON_FRNET_TARGETED_RECOVERY":"0",
    "PYTHON_FRNET_TARGETED_V2":"0",
    "PYTHON_FRNET_PREDICTIVE_ALL_JOINTS":"0",
    "PYTHON_FRNET_PREDICTIVE_LATCH":"1",
    "FRNET_PREDICTIVE_GENERAL_TRIGGER":"0.50",
    "FRNET_PREDICTIVE_CALF_TRIGGER":"0.30",
    "FRNET_PREDICTIVE_GENERAL_OFFSET":"0.015",
    "FRNET_PREDICTIVE_CALF_OFFSET":"0.030",
    "FRNET_PREDICTIVE_BRAKE_RELEASE_RATIO":"0.20",
    "FRNET_SAFETY_GENERAL_MARGIN":"0.020",
    "FRNET_SAFETY_CALF_MARGIN":"0.040",
    "FRNET_INITIAL_TARGET_HOLD_STEPS":"0",
    "FRNET_RESET_TARGET_RAMP_STEPS":"20",
    "PYTHON_FRNET_OFFICIAL_PASSIVE_DYNAMICS":"0",
    "PYTHON_FRNET_HARD_JOINT_LIMITS":"0",
    "FRNET_JOINT_LIMIT_MARGIN":"0.02",
    "FRNET_JOINT_LIMIT_TIMECONST":"0.01",
    "FRNET_JOINT_LIMIT_DAMPRATIO":"1.0",
    "FRNET_JOINT_LIMIT_IMPEDANCE_MIN":"0.95",
    "FRNET_JOINT_LIMIT_IMPEDANCE_MAX":"0.99",
    "PYTHON_FRNET_ACTUATOR_VELOCITY_BRAKE":"0",
    "FRNET_ACTUATOR_GENERAL_BRAKE_TRIGGER":"0.50",
    "FRNET_ACTUATOR_CALF_BRAKE_TRIGGER":"0.30",
    "PYTHON_FRNET_ACTUATOR_POSITION_BARRIER":"0",
    "FRNET_ACTUATOR_POSITION_MARGIN":"0.05",
    "FRNET_ACTUATOR_POSITION_BRAKE_GAIN":"0.25",
    "PYTHON_FRNET_ACTUATOR_POSITION_DAMPING":"0",
    "FRNET_ACTUATOR_POSITION_DAMPING_MARGIN":"0.12",
    "FRNET_ACTUATOR_POSITION_DAMPING_GAIN":"2.0",
    "PYTHON_FRNET_FOLDED_CALF_PRELOAD":"0",
    "FRNET_FOLDED_CALF_PRELOAD_NM":"0.0",
    "FRNET_FOLDED_CALF_PRELOAD_STEPS":"1",
    "FRNET_CONTACT_PRELOAD_DEPTH_M":"0.002",
    "PYTHON_FRNET_REMAINING_BRAKE":"0",
    "FRNET_REMAINING_BRAKE_INERTIA":"0.10",
    "FRNET_REMAINING_BRAKE_SAFETY_FACTOR":"0.10",
    "FRNET_REMAINING_BRAKE_DT":"0.02",
    "FRNET_REMAINING_BRAKE_MAX_TARGET_STEP":"0.015",
    "FRNET_REMAINING_BRAKE_TORQUE_GAIN":"1.0",
    "FRNET_REMAINING_BRAKE_TORQUE_BASE_GAIN":"0.25",
    "FRNET_GLOBAL_IMPACT_CAUSAL_MODE":"0",
    "FRNET_GLOBAL_IMPACT_TRANSFER_GUARD":"0",
    "FRNET_GLOBAL_IMPACT_TRANSFER_REQUIRE_PREVIOUS":"1",
    "FRNET_GLOBAL_IMPACT_TRANSFER_INCLUDE_HIP":"0",
    "FRNET_GLOBAL_IMPACT_TRANSFER_COORDINATE":"1",
    "FRNET_GLOBAL_IMPACT_TRANSFER_NEIGHBOR_MODE":"lateral",
    "FRNET_GLOBAL_IMPACT_TRANSFER_EXTRA_PAIRS":"",
    "FRNET_GLOBAL_IMPACT_TRANSFER_SPEED_RATIO":"0.0",
    "FRNET_GLOBAL_IMPACT_TRANSFER_ACCEL_THRESHOLD":"0.40",
    "FRNET_GLOBAL_IMPACT_TRANSFER_SCALE":"0.50",
    "FRNET_GLOBAL_IMPACT_TRANSFER_USE_SOURCE":"0",
    "FRNET_GLOBAL_IMPACT_PRESERVE_HIP":"0",
    "FRNET_GLOBAL_IMPACT_GUARD_RELEASE_RATIO":"0.0",
    "FRNET_GLOBAL_IMPACT_PREDICTIVE_STOP":"0",
    "FRNET_GLOBAL_IMPACT_PREDICTIVE_DT":"0.02",
    "FRNET_GLOBAL_IMPACT_PREDICTIVE_ACCEL":"1.5",
    "FRNET_GLOBAL_IMPACT_PREDICTIVE_CALF_ONLY":"0",
    "FRNET_IMPACT_SOURCE_STRICT":"0",
    "FRNET_TARGET_SLEW_MAX_RAD":"0.0",
    "FRNET_TARGET_SLEW_CALF_MAX_RAD":"0.0",
    "FRNET_PREDICTIVE_TARGET_BRAKE_GAIN":"0.0",
    "FRNET_PREDICTIVE_TARGET_BRAKE_CALF_GAIN":"0.0",
    "FRNET_PREDICTIVE_TARGET_BRAKE_TRIGGER":"0.30",
    "FRNET_SOURCE_JOINT_COORDINATION":"0",
    "FRNET_SOURCE_JOINT_CALF_SCALE":"0.85",
    "FRNET_SOURCE_JOINT_THIGH_SCALE":"0.95",
    "FRNET_SOURCE_JOINT_HIP_SCALE":"1.0",
    "FRNET_SOURCE_JOINT_COORDINATION_MAX_STEPS":"4",
    "FRNET_SOURCE_JOINT_REQUIRE_NONHIP":"0",
    "FRNET_SLOPE_MAX_DEG":"5.0",
}


def evaluation_runtime_config():
    """Return the effective, non-secret environment settings for this run."""
    return {key:os.environ.get(key,default)
            for key,default in RUNTIME_ENV_DEFAULTS.items()}


def stable_seed(terrain,pose,mode,sample):
    return (20260916+list(TERRAINS).index(terrain)*100_000
            +list(POSES).index(pose)*10_000+MODES.index(mode)*100+sample)


def case_for(terrain,pose,mode,sample):
    seed=stable_seed(terrain,pose,mode,sample)
    rng=random.Random(seed)
    column=rng.choice(TERRAINS[terrain]); row=rng.randrange(10)
    xy=(rng.uniform(-2.7,2.7),rng.uniform(-2.7,2.7))
    yaw=rng.uniform(-math.pi,math.pi)
    q=[rng.uniform(lo+0.03,hi-0.03) for lo,hi in zip(POSITION_LOWER,POSITION_UPPER)]
    action_delay_steps=rng.randrange(6)
    if mode=="all_folded": legs=(0,1,2,3)
    elif mode.startswith("single_"): legs=(("FL","FR","RL","RR").index(mode[7:]),)
    elif mode=="diagonal_FL_RR": legs=(0,3)
    elif mode=="diagonal_FR_RL": legs=(1,2)
    else: legs=()
    for leg in legs:
        for j,value in enumerate(FOLDED[leg]):
            index=leg*3+j
            q[index]=min(max(value,POSITION_LOWER[index]+0.03),POSITION_UPPER[index]-0.03)
    return dict(terrain_family=terrain,terrain_row=row,terrain_column=column,
                fall_pose=pose,birth_mode=mode,sample=sample,seed=seed,
                spawn_offset_xy_m=xy,yaw_rad=yaw,joint_positions_rad=q,
                action_delay_steps=action_delay_steps,
                terrain_seed=EVALUATION_TERRAIN_SEED)


def build_cases(terrains,poses,modes,per_stratum):
    return [case_for(t,p,m,s) for t in terrains for p in poses for m in modes
            for s in range(per_stratum)]


def select_case_episodes(cases, episodes):
    """Return exact cases by their pre-selection episode indices."""
    if episodes is None:
        return cases
    selected=[]
    for episode in episodes:
        if episode < 0 or episode >= len(cases):
            raise ValueError(
                f"case episode {episode} is outside [0, {len(cases)-1}]"
            )
        row=dict(cases[episode])
        row["source_episode"]=int(episode)
        selected.append(row)
    return selected


def static_library_row_matches_mode(row):
    """Reject a library row whose relaxed pose no longer matches its label."""
    mode=row.get("birth_mode")
    if mode == "independent_random":
        return True
    q=row.get("joint_positions_rad",())
    if len(q) != 12:
        return False
    folded=tuple(q[index] > 2.50 and q[index+1] < -2.45
                 for index in (1,4,7,10))
    if mode == "all_folded":
        return sum(folded) >= 3
    required={
        "single_FL":(0,),"single_FR":(1,),"single_RL":(2,),"single_RR":(3,),
        "diagonal_FL_RR":(0,3),"diagonal_FR_RL":(1,2),
    }
    return mode in required and all(folded[index] for index in required[mode])


def replace_all_folded_with_static_library(cases,rows):
    """Use deterministic, release-validated moderate folds for curriculum eval."""
    by_stratum={}
    for row in rows:
        if row.get("birth_mode") != "all_folded" or not static_library_row_matches_mode(row):
            continue
        by_stratum.setdefault(
            (row["terrain_family"],row["fall_pose"]),[]
        ).append(row)
    output=[]
    fields=(
        "terrain_row","terrain_column","spawn_offset_xy_m","yaw_rad",
        "joint_positions_rad","relative_root_pose","control_target_rad",
        "fold_alpha",
    )
    for case in cases:
        if case["birth_mode"] != "all_folded":
            output.append(case)
            continue
        key=(case["terrain_family"],case["fall_pose"])
        candidates=by_stratum.get(key,())
        if not candidates:
            raise ValueError(f"Static reset library lacks stratum {key}")
        source=candidates[int(case["sample"]) % len(candidates)]
        merged=dict(case)
        for field in fields:
            if field in source:
                merged[field]=source[field]
        merged["static_library_state"]=True
        output.append(merged)
    return output


def replace_matching_cases_with_static_library(cases, rows):
    """Replace only strata represented by a release-validated static library."""
    by_stratum={}
    for row in rows:
        if not static_library_row_matches_mode(row):
            continue
        by_stratum.setdefault(
            (row["terrain_family"],row["fall_pose"],row["birth_mode"]),[]
        ).append(row)
    output=[]
    fields=(
        "terrain_row","terrain_column","spawn_offset_xy_m","yaw_rad",
        "joint_positions_rad","relative_root_pose","control_target_rad",
        "fold_alpha",
    )
    for case in cases:
        key=(case["terrain_family"],case["fall_pose"],case["birth_mode"])
        candidates=by_stratum.get(key)
        if not candidates:
            output.append(case)
            continue
        source=candidates[int(case["sample"]) % len(candidates)]
        merged=dict(case)
        for field in fields:
            if field in source:
                merged[field]=source[field]
        merged["static_library_state"]=True
        merged["static_library_source_sample"]=int(source.get("sample",0))
        output.append(merged)
    return output


def assign_cases(env,cases):
    """Install independent static fallen states in an already-reset vector env."""
    n=len(cases); ids=torch.arange(env.num_envs,device=env.device)
    if n<env.num_envs:
        cases=cases+[cases[-1]]*(env.num_envs-n)
    terrain=env.scene.terrain
    rows=torch.tensor([c['terrain_row'] for c in cases],device=env.device,dtype=torch.long)
    columns=torch.tensor([c['terrain_column'] for c in cases],device=env.device,dtype=torch.long)
    terrain.terrain_levels[ids]=rows;terrain.terrain_types[ids]=columns
    terrain.env_origins[ids]=terrain.terrain_origins[rows,columns]
    q=torch.tensor([c['joint_positions_rad'] for c in cases],device=env.device)
    root=env.scene['robot'].data.default_root_state.clone()
    root[:,:3]=terrain.env_origins[ids]
    root[:,:2]+=torch.tensor([c['spawn_offset_xy_m'] for c in cases],device=env.device)
    angles=torch.tensor([POSES[c['fall_pose']] for c in cases],device=env.device)
    yaw=torch.tensor([c['yaw_rad'] for c in cases],device=env.device)
    root[:,3:7]=quat_from_euler_xyz(angles[:,0],angles[:,1],yaw)
    root[:,7:]=0.0
    library_mask=torch.tensor(
        [bool(c.get("static_library_state",False)) for c in cases],
        device=env.device,dtype=torch.bool,
    )
    library_ids=library_mask.nonzero(as_tuple=False).squeeze(-1)
    if len(library_ids):
        relative=torch.tensor(
            [cases[int(i)]["relative_root_pose"] for i in library_ids.tolist()],
            device=env.device,dtype=root.dtype,
        )
        root[library_ids,:3]=terrain.env_origins[library_ids]+relative[:,:3]
        root[library_ids,3:7]=relative[:,3:7]
    control_target=q.clone()
    bracket_found=library_mask.clone()
    regular_ids=(~library_mask).nonzero(as_tuple=False).squeeze(-1)
    if len(regular_ids):
        placed_root,placed_q,placed_target,placed_found=_place_static_in_contact(
            env,regular_ids,q[regular_ids],root[regular_ids],require_nonfoot=False
        )
        root[regular_ids]=placed_root
        q[regular_ids]=placed_q
        control_target[regular_ids]=placed_target
        bracket_found[regular_ids]=placed_found
    if len(library_ids):
        control_target[library_ids]=torch.tensor(
            [cases[int(i)]["control_target_rad"] for i in library_ids.tolist()],
            device=env.device,dtype=q.dtype,
        )
    robot=env.scene['robot']
    robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids)
    robot.write_root_state_to_sim(root,env_ids=ids)
    action=env.action_manager.get_term('recovery')
    action.sync_target_to_joint_state(ids,q,reference_target=control_target)
    action.delay.delays[ids]=torch.tensor(
        [c['action_delay_steps'] for c in cases],
        device=env.device,dtype=action.delay.delays.dtype,
    )
    env.scene.write_data_to_sim();env.sim.forward();env.sim.sense()
    contact=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    initial_contact_force=torch.zeros(env.num_envs,device=env.device)
    force_by_sensor={name:torch.zeros(env.num_envs,device=env.device) for name in ('nonfoot_ground_touch','base_ground_contact','feet_ground_contact')}
    for name in ('nonfoot_ground_touch','base_ground_contact','feet_ground_contact'):
        sensor=env.scene[name];sensor.update(0.0)
        contact|=sensor.data.found.reshape(env.num_envs,-1).any(dim=-1)
        force=getattr(sensor.data, "force", None)
        if force is not None:
            force_norm=force.reshape(env.num_envs,-1,3).norm(dim=-1)
            contact |= force_norm.gt(1.0).any(dim=-1)
            sensor_total=force_norm.sum(dim=-1)
            force_by_sensor[name]=sensor_total
            initial_contact_force += sensor_total
    initial_unique_contact_force=force_by_sensor['nonfoot_ground_touch']+force_by_sensor['feet_ground_contact']
    root_speed=robot.data.root_link_vel_w.norm(dim=-1)
    joint_speed=robot.data.joint_vel.abs().amax(dim=-1)
    env.obs_buf=env.observation_manager.compute(update_history=True)
    return (bracket_found[:n]&contact[:n],root_speed[:n],joint_speed[:n],initial_contact_force[:n],initial_unique_contact_force[:n],{name:value[:n] for name,value in force_by_sensor.items()})


def merge_actuator_barrier_diagnostics(num_envs, num_joints, records):
    """Place per-actuator diagnostic tensors into canonical joint order."""
    if not records:
        return torch.zeros((num_envs,num_joints),dtype=torch.int32),torch.zeros((num_envs,num_joints))
    device=records[0][1].device
    counts=torch.zeros((num_envs,num_joints),dtype=torch.int32,device=device)
    peaks=torch.zeros((num_envs,num_joints),dtype=records[0][2].dtype,device=device)
    for target_ids,actuator_counts,actuator_peaks in records:
        counts[:,target_ids]=actuator_counts
        peaks[:,target_ids]=actuator_peaks
    return counts,peaks


def reset_actuator_barrier_diagnostics(robot, env_ids=None):
    for actuator in robot.actuators:
        reset=getattr(actuator,"reset_barrier_diagnostics",None)
        if reset is not None:
            reset(env_ids)


def collect_actuator_barrier_diagnostics(robot):
    records=[]
    for actuator in robot.actuators:
        counts=getattr(actuator,"barrier_active_counts",None)
        peaks=getattr(actuator,"barrier_max_torque_delta",None)
        if counts is not None:
            records.append((actuator.target_ids,counts,peaks))
    if not records:
        shape=robot.data.joint_pos.shape
        return (
            torch.zeros(shape,dtype=torch.int32,device=robot.data.joint_pos.device),
            torch.zeros_like(robot.data.joint_pos),
        )
    return merge_actuator_barrier_diagnostics(
        robot.data.joint_pos.shape[0],robot.data.joint_pos.shape[1],records
    )


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--terrains',nargs='+',choices=TERRAINS,default=list(TERRAINS))
    parser.add_argument('--poses',nargs='+',choices=POSES,default=list(POSES))
    parser.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    parser.add_argument('--per-stratum',type=int,default=20)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--horizon',type=int,default=600)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--case-episodes',nargs='+',type=int)
    parser.add_argument('--static-library-all-folded',action='store_true')
    parser.add_argument('--static-library-matching',action='store_true')
    parser.add_argument('--static-library-path',default=str(
        Path(__file__).resolve().parents[1]
        /'src/tasks/velocity/config/go2_recovery_aligned/static_reset_library_fold55_v1.json'
    ))
    args=parser.parse_args()
    if args.horizon<500: raise ValueError('Acceptance horizon must be >=500 steps')
    cases=build_cases(args.terrains,args.poses,args.modes,args.per_stratum)
    if args.static_library_all_folded and args.static_library_matching:
        raise ValueError('select only one static-library replacement mode')
    if args.static_library_all_folded:
        library=json.loads(Path(args.static_library_path).read_text())
        cases=replace_all_folded_with_static_library(cases,library['rows'])
    elif args.static_library_matching:
        library=json.loads(Path(args.static_library_path).read_text())
        cases=replace_matching_cases_with_static_library(cases,library['rows'])
    cases=select_case_episodes(cases,args.case_episodes)
    case_set_hash=hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest()
    torch.manual_seed(20260916)
    torch.cuda.manual_seed_all(20260916)
    cfg=load_env_cfg('Unitree-Go2-Recovery-FRNet-Aligned',play=True)
    cfg.seed=20260916
    cfg.scene.terrain.terrain_generator.seed=EVALUATION_TERRAIN_SEED
    cfg.scene.num_envs=args.batch_size;cfg.viewer=None
    cfg.events.pop('settle_fallen',None)
    env=ManagerBasedRlEnv(cfg=cfg,device=args.device)
    wrapped=RslRlVecEnvWrapper(env)
    agent_cfg=load_rl_cfg('Unitree-Go2-Recovery-FRNet-Aligned')
    runner_cls=load_runner_cls('Unitree-Go2-Recovery-FRNet-Aligned') or MjlabOnPolicyRunner
    runner=runner_cls(wrapped,asdict(agent_cfg),device=args.device)
    runner.load(args.checkpoint,load_cfg={'actor':True},strict=True,map_location=args.device)
    policy=runner.get_inference_policy(device=args.device)
    robot=env.scene['robot']
    action_term=env.action_manager.get_term('recovery')
    if tuple(robot.joint_names)!=JOINT_NAMES or set(robot.actuator_names)!=set(JOINT_NAMES):
        raise ValueError('Unexpected Go2 joint or actuator order')
    effort_by_name=dict(zip(JOINT_NAMES,FRNET_EFFORT_LIMIT,strict=True))
    effort=torch.tensor([effort_by_name[name] for name in robot.actuator_names],device=args.device)
    velocity=torch.tensor(VELOCITY_LIMIT,device=args.device)
    lower=torch.tensor(POSITION_LOWER,device=args.device)
    upper=torch.tensor(POSITION_UPPER,device=args.device)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for start in range(0,len(cases),args.batch_size):
        group=cases[start:start+args.batch_size];n=len(group)
        env.reset();contact,root_speed,joint_speed,initial_contact_force,initial_unique_contact_force,force_by_sensor=assign_cases(env,group)
        reset_actuator_barrier_diagnostics(robot)
        obs=wrapped.get_observations()
        active=torch.arange(env.num_envs,device=env.device)<n
        hold=torch.zeros(env.num_envs,dtype=torch.int32,device=env.device)
        success_step=torch.full((env.num_envs,),-1,dtype=torch.int32,device=env.device)
        reset_detected=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
        peak_torque=torch.zeros(env.num_envs,device=env.device)
        peak_torque_ratio=torch.zeros_like(peak_torque)
        peak_velocity_ratio=torch.zeros_like(peak_torque)
        peak_velocity_joint=torch.zeros(env.num_envs,dtype=torch.long,device=env.device)
        peak_velocity_step=torch.zeros(env.num_envs,dtype=torch.int32,device=env.device)
        peak_velocity_signed=torch.zeros_like(peak_torque)
        peak_velocity_position=torch.zeros_like(peak_torque)
        peak_velocity_target=torch.zeros_like(peak_torque)
        startup_calf_velocity_ratio=torch.zeros_like(peak_torque)
        safety_modified_steps=torch.zeros(env.num_envs,dtype=torch.int32,device=env.device)
        safety_max_target_delta=torch.zeros_like(peak_torque)
        safety_max_brake_distance=torch.zeros_like(peak_torque)
        barrier_active_counts=torch.zeros((env.num_envs,12),dtype=torch.int32,device=env.device)
        barrier_max_torque_delta=torch.zeros((env.num_envs,12),device=env.device)
        peak_position_excess=torch.zeros_like(peak_torque)
        peak_position_joint=torch.zeros(env.num_envs,dtype=torch.long,device=env.device)
        peak_position_step=torch.zeros(env.num_envs,dtype=torch.int32,device=env.device)
        diagnostic_names=("height","upright","angular","linear","four_feet","joint_ready","acceptance")
        diagnostic_counts={name:torch.zeros(env.num_envs,dtype=torch.int32,device=env.device) for name in diagnostic_names}
        diagnostic_runs={name:torch.zeros(env.num_envs,dtype=torch.int32,device=env.device) for name in diagnostic_names}
        diagnostic_max_runs={name:torch.zeros(env.num_envs,dtype=torch.int32,device=env.device) for name in diagnostic_names}
        engagement_joint_pos=torch.full((env.num_envs,12),float('nan'),device=env.device)
        engagement_height=torch.full((env.num_envs,),float('nan'),device=env.device)
        final_height=torch.full_like(peak_torque,float('nan'))
        steps_run=torch.zeros(env.num_envs,dtype=torch.int32,device=env.device)
        impact_trigger_steps=torch.zeros(env.num_envs,dtype=torch.int32,device=env.device)
        impact_trigger_by_leg=torch.zeros((env.num_envs,4),dtype=torch.int32,device=env.device)
        first_impact_leg=torch.full((env.num_envs,),-1,dtype=torch.int32,device=env.device)
        first_impact_step=torch.full((env.num_envs,),-1,dtype=torch.int32,device=env.device)
        first_source_leg=torch.full((env.num_envs,),-1,dtype=torch.int32,device=env.device)
        first_source_joint=torch.full((env.num_envs,),-1,dtype=torch.int32,device=env.device)
        impact_source_trigger_steps=torch.zeros(env.num_envs,dtype=torch.int32,device=env.device)
        impact_source_by_leg=torch.zeros((env.num_envs,4),dtype=torch.int32,device=env.device)
        impact_source_max_score=torch.zeros(env.num_envs,device=env.device)
        first_source_step=torch.full((env.num_envs,),-1,dtype=torch.int32,device=env.device)
        previous_eval_velocity=torch.zeros_like(robot.data.joint_vel)
        peak_acceleration_ratio=torch.zeros_like(peak_torque)
        peak_acceleration_joint=torch.zeros(env.num_envs,dtype=torch.long,device=env.device)
        for step in range(args.horizon):
            with torch.inference_mode(): action=policy(obs)
            obs,_,done,_=wrapped.step(action)
            torque=robot.data.actuator_force.abs()
            speed=robot.data.joint_vel.abs()
            eval_acceleration=(robot.data.joint_vel-previous_eval_velocity)/max(env.step_dt,1e-6)
            acceleration_ratio=(eval_acceleration.abs()/velocity.clamp_min(1e-6)).amax(dim=-1)
            acc_new=active & (acceleration_ratio>peak_acceleration_ratio)
            peak_acceleration_ratio=torch.where(acc_new,acceleration_ratio,peak_acceleration_ratio)
            _,acc_idx=eval_acceleration.abs().max(dim=-1)
            peak_acceleration_joint=torch.where(acc_new,acc_idx,peak_acceleration_joint)
            previous_eval_velocity.copy_(robot.data.joint_vel)
            excess=torch.maximum(lower-robot.data.joint_pos,robot.data.joint_pos-upper).clamp_min(0)
            peak_torque=torch.where(active,torch.maximum(peak_torque,torque.amax(-1)),peak_torque)
            peak_torque_ratio=torch.where(active,torch.maximum(peak_torque_ratio,(torque/effort).amax(-1)),peak_torque_ratio)
            velocity_ratio=speed/velocity
            velocity_value,velocity_index=velocity_ratio.max(dim=-1)
            velocity_new=active & (velocity_value>peak_velocity_ratio)
            peak_velocity_ratio=torch.where(velocity_new,velocity_value,peak_velocity_ratio)
            peak_velocity_joint=torch.where(velocity_new,velocity_index,peak_velocity_joint)
            peak_velocity_step=torch.where(velocity_new,torch.full_like(peak_velocity_step,step),peak_velocity_step)
            selected_velocity=robot.data.joint_vel.gather(1,velocity_index.unsqueeze(-1)).squeeze(-1)
            selected_position=robot.data.joint_pos.gather(1,velocity_index.unsqueeze(-1)).squeeze(-1)
            selected_target=action_term.target.gather(1,velocity_index.unsqueeze(-1)).squeeze(-1)
            peak_velocity_signed=torch.where(velocity_new,selected_velocity,peak_velocity_signed)
            peak_velocity_position=torch.where(velocity_new,selected_position,peak_velocity_position)
            peak_velocity_target=torch.where(velocity_new,selected_target,peak_velocity_target)
            if step < 50:
                calf_ratio=velocity_ratio[:,(2,5,8,11)].amax(dim=-1)
                startup_calf_velocity_ratio=torch.where(
                    active,torch.maximum(startup_calf_velocity_ratio,calf_ratio),startup_calf_velocity_ratio
                )
            guard_trigger=getattr(robot,'_frnet_global_impact_guard_trigger',None)
            if guard_trigger is not None:
                impact_trigger_steps += active & guard_trigger.any(dim=-1)
                impact_trigger_by_leg += guard_trigger.to(torch.int32)
                trigger_any=guard_trigger.any(dim=-1)
                trigger_leg=guard_trigger.to(torch.int32).argmax(dim=-1)
                first_impact_leg=torch.where((first_impact_leg<0)&trigger_any,trigger_leg,first_impact_leg)
                first_impact_step=torch.where(
                    (first_impact_step < 0) & trigger_any,
                    torch.full_like(first_impact_step, step),
                    first_impact_step,
                )
            source_leg=getattr(robot,"_frnet_impact_source_leg",None)
            source_joint=getattr(robot,"_frnet_impact_source_joint",None)
            source_trigger=getattr(robot,"_frnet_impact_source_trigger",None)
            source_score=getattr(robot,"_frnet_impact_source_score",None)
            if source_leg is not None and source_trigger is not None:
                source_active=active & source_trigger
                impact_source_trigger_steps += source_active
                first_source_leg=torch.where(
                    (first_source_leg < 0) & source_active,
                    source_leg,
                    first_source_leg,
                )
                if source_joint is not None:
                    first_source_joint=torch.where(
                        (first_source_joint < 0) & source_active,
                        source_joint,
                        first_source_joint,
                    )
                first_source_step=torch.where(
                    (first_source_step < 0) & source_active,
                    torch.full_like(first_source_step, step),
                    first_source_step,
                )
                valid_source=source_active & (source_leg >= 0)
                source_one_hot=torch.nn.functional.one_hot(
                    source_leg.clamp_min(0).to(torch.long), num_classes=4
                ).to(torch.int32)
                impact_source_by_leg += source_one_hot * valid_source.unsqueeze(-1).to(torch.int32)
            if source_score is not None:
                impact_source_max_score=torch.where(
                    active, torch.maximum(impact_source_max_score, source_score), impact_source_max_score
                )
            safety_modified=getattr(robot,'_frnet_safety_modified',torch.zeros_like(active))
            safety_modified_steps+=active & safety_modified
            safety_delta=getattr(robot,'_frnet_safety_target_delta',None)
            if safety_delta is not None:
                safety_max_target_delta=torch.where(
                    active,torch.maximum(safety_max_target_delta,safety_delta.amax(dim=-1)),safety_max_target_delta
                )
            safety_brake=getattr(robot,'_frnet_safety_brake_distance',None)
            if safety_brake is not None:
                safety_max_brake_distance=torch.where(
                    active,torch.maximum(safety_max_brake_distance,safety_brake.amax(dim=-1)),safety_max_brake_distance
                )
            current_barrier_counts,current_barrier_peaks=collect_actuator_barrier_diagnostics(robot)
            barrier_active_counts=torch.where(
                active.unsqueeze(-1),current_barrier_counts,barrier_active_counts
            )
            barrier_max_torque_delta=torch.where(
                active.unsqueeze(-1),current_barrier_peaks,barrier_max_torque_delta
            )
            position_value,position_index=excess.max(dim=-1)
            position_new=active & (position_value>peak_position_excess)
            peak_position_excess=torch.where(position_new,position_value,peak_position_excess)
            peak_position_joint=torch.where(position_new,position_index,peak_position_joint)
            peak_position_step=torch.where(position_new,torch.full_like(peak_position_step,step),peak_position_step)
            steps_run[active]=step+1
            height=local_relative_height(env)
            final_height=torch.where(active,height,final_height)
            good=physical_acceptance(env)
            engagement=(hold==9)&good&torch.isnan(engagement_height)
            engagement_joint_pos[engagement]=robot.data.joint_pos[engagement]
            engagement_height[engagement]=height[engagement]
            target=robot.data.joint_pos.new_tensor(STAND_JOINT_TARGET)
            components={
                "height":height > 0.30,
                "upright":robot.data.projected_gravity_b[:,2] < -0.90,
                "angular":robot.data.root_link_ang_vel_b.norm(dim=-1) < 0.35,
                "linear":robot.data.root_link_lin_vel_b.norm(dim=-1) < 0.30,
                "four_feet":foot_contact_mask(env).all(dim=-1),
                "joint_ready":(robot.data.joint_pos-target).abs().amax(dim=-1) < 0.55,
                "acceptance":good,
            }
            for name,value in components.items():
                diagnostic_counts[name]+=active & value
                diagnostic_runs[name]=torch.where(active & value,diagnostic_runs[name]+1,torch.zeros_like(diagnostic_runs[name]))
                diagnostic_max_runs[name]=torch.maximum(diagnostic_max_runs[name],diagnostic_runs[name])
            hold=torch.where(active&good,hold+1,torch.where(active,torch.zeros_like(hold),hold))
            won=active&(hold>=50)
            success_step[won]=step-49
            done=done.bool()
            early_done=active&done&(step<args.horizon-1)
            reset_detected|=early_done
            active&=~won&~done
            if not active[:n].any(): break
        for i,case in enumerate(group):
            row=dict(case)
            row.update(episode=start+i,initial_contact=bool(contact[i].item()),
                       initial_root_speed=float(root_speed[i].item()),
                       initial_joint_speed=float(joint_speed[i].item()),
                       initial_contact_force_n=float(initial_contact_force[i].item()),
                       initial_unique_contact_force_n=float(initial_unique_contact_force[i].item()),
                       initial_contact_force_by_sensor_n={name:float(value[i].item()) for name,value in force_by_sensor.items()},
                       initial_state_verified=bool(contact[i].item() and root_speed[i].item()<=1e-8 and joint_speed[i].item()<=1e-8),
                       success=bool(success_step[i].item()>=0 and not reset_detected[i].item()),
                       recovery_time_s=None if success_step[i].item()<0 else round(float(success_step[i].item()*env.step_dt),4),
                       horizon_s=round(args.horizon*env.step_dt,4),steps_run=int(steps_run[i].item()),
                       reset_detected=bool(reset_detected[i].item()),
                       final_relative_height_m=float(final_height[i].item()),
                       peak_torque_nm=float(peak_torque[i].item()),
                       peak_torque_ratio=float(peak_torque_ratio[i].item()),
                       peak_velocity_ratio=float(peak_velocity_ratio[i].item()),
                       peak_velocity_joint=robot.joint_names[int(peak_velocity_joint[i].item())],
                       peak_velocity_step=int(peak_velocity_step[i].item()),
                       peak_velocity_rad_s=float(peak_velocity_signed[i].item()),
                       peak_velocity_position_rad=float(peak_velocity_position[i].item()),
                       peak_velocity_target_rad=float(peak_velocity_target[i].item()),
                       first_impact_guard_leg=None if first_impact_leg[i].item()<0 else ("FL","FR","RL","RR")[int(first_impact_leg[i].item())],
                       first_impact_guard_step=int(first_impact_step[i].item()),
                       impact_guard_trigger_steps=int(impact_trigger_steps[i].item()),
                       impact_guard_trigger_steps_by_leg={leg:int(v) for leg,v in zip(("FL","FR","RL","RR"),impact_trigger_by_leg[i].tolist(),strict=True)},
                       first_impact_source_leg=None if first_source_leg[i].item()<0 else ("FL","FR","RL","RR")[int(first_source_leg[i].item())],
                       first_impact_source_joint=None if first_source_joint[i].item()<0 else robot.joint_names[int(first_source_joint[i].item())],
                       first_impact_source_step=int(first_source_step[i].item()),
                       impact_source_trigger_steps=int(impact_source_trigger_steps[i].item()),
                       impact_source_trigger_steps_by_leg={leg:int(v) for leg,v in zip(("FL","FR","RL","RR"),impact_source_by_leg[i].tolist(),strict=True)},
                       impact_source_max_score=float(impact_source_max_score[i].item()),
                       peak_acceleration_ratio=float(peak_acceleration_ratio[i].item()),
                       peak_acceleration_joint=robot.joint_names[int(peak_acceleration_joint[i].item())],
                       first_50_calf_peak_velocity_ratio=float(startup_calf_velocity_ratio[i].item()),
                       safety_modified_steps=int(safety_modified_steps[i].item()),
                       safety_modified_fraction=float(safety_modified_steps[i].item()/max(1,steps_run[i].item())),
                       safety_max_target_delta_rad=float(safety_max_target_delta[i].item()),
                       safety_max_brake_distance_rad=float(safety_max_brake_distance[i].item()),
                       actuator_barrier_active_joint_substeps=int(barrier_active_counts[i].sum().item()),
                       actuator_barrier_active_joint_count=int((barrier_active_counts[i]>0).sum().item()),
                       actuator_barrier_active_substeps_by_joint={
                           name:int(value) for name,value in zip(
                               JOINT_NAMES,barrier_active_counts[i].tolist(),strict=True
                           )
                       },
                       actuator_barrier_max_torque_delta_nm=float(barrier_max_torque_delta[i].amax().item()),
                       peak_position_excess_rad=float(peak_position_excess[i].item()),
                       peak_position_joint=robot.joint_names[int(peak_position_joint[i].item())],
                       peak_position_step=int(peak_position_step[i].item()),
                       hardware_safety_pass=bool(peak_torque_ratio[i].item()<=1.00001 and peak_velocity_ratio[i].item()<=1.0 and peak_position_excess[i].item()<=0.02),
                       acceptance_diagnostics={name:{
                           'fraction':float(diagnostic_counts[name][i].item()/max(1,steps_run[i].item())),
                           'max_consecutive_steps':int(diagnostic_max_runs[name][i].item()),
                       } for name in diagnostic_names},
                       engagement_height_m=None if torch.isnan(engagement_height[i]) else float(engagement_height[i].item()),
                       engagement_joint_positions_rad=None if torch.isnan(engagement_height[i]) else [float(x) for x in engagement_joint_pos[i].tolist()])
            rows.append(row)
        print(json.dumps({'completed':len(rows),'total':len(cases),
                          'success':sum(r['success'] for r in rows),
                          'initial_valid':sum(r['initial_state_verified'] for r in rows),
                          'mid_resets':sum(r['reset_detected'] for r in rows)}),flush=True)
        output.with_suffix(output.suffix+'.partial').write_text(json.dumps(rows))
    report={'checkpoint':args.checkpoint,'suite':'native_terrain_decoupled_batched',
            'episodes':len(rows),'per_stratum':args.per_stratum,'horizon_s':args.horizon*env.step_dt,
            'case_set_sha256':case_set_hash,'environment_seed':cfg.seed,
            'runtime_config':evaluation_runtime_config(),
            'results':rows}
    output.write_text(json.dumps(report,indent=2)+'\n')
    env.close()


if __name__=='__main__': main()

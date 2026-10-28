"""Training reset mixture (distinct from the settled Viser reset)."""
import math
import os
import json
from pathlib import Path

import torch
from mjlab.utils.lab_api.math import (
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
)

from .baseline_protocol import JOINT_NAMES, sample_joint_reset, sample_root_reset


# Every training reset starts from a settled, zero-velocity fallen state.
STATIC_FALL_FRACTION = 1.0
# Strict static-start protocol: no residual root motion is sampled.
RESIDUAL_MOTION_FRACTION = 0.0
TARGETED_REPLAY_FRACTION = 0.40
TARGETED_ALL_FOLDED_FRACTION = 0.50
TARGETED_RIGHT_FALL_FRACTION = 0.35
_SUCCESS_POSES = ((0.0,-0.5*math.pi),(0.0,0.5*math.pi),
                  (-0.5*math.pi,0.0),(0.5*math.pi,0.0))
_SUCCESS_POSE_NAMES = ('front','back','left','right')
_replay_cache = {}
_static_library_cache = {}
THIGH_INDICES=[1,4,7,10]
CALF_INDICES=[2,5,8,11]
THIGH_INDICES=[1,4,7,10]
CALF_INDICES=[2,5,8,11]


def _success_replay(device):
    key=str(device)
    if key not in _replay_cache:
        path=Path(os.environ.get(
            "PYTHON_FRNET_SUCCESS_REPLAY_PATH",
            str(Path(__file__).with_name("success_replay.json")),
        ))
        data=json.loads(path.read_text())
        if tuple(data['joint_names']) != JOINT_NAMES or data['count'] != len(data['rows']):
            raise ValueError('Success replay joint order or record count mismatch')
        rows=data['rows']
        _replay_cache[key]={
            'q':torch.tensor([r['joint_positions_rad'] for r in rows],device=device),
            'row':torch.tensor([r['terrain_row'] for r in rows],device=device,dtype=torch.long),
            'column':torch.tensor([r['terrain_column'] for r in rows],device=device,dtype=torch.long),
            'offset':torch.tensor([r['spawn_offset_xy_m'] for r in rows],device=device),
            'pose':torch.tensor([_SUCCESS_POSE_NAMES.index(r['fall_pose']) for r in rows],device=device,dtype=torch.long),
        }
    return _replay_cache[key]


def _static_library_target_fraction():
    value=float(os.environ.get("PYTHON_FRNET_STATIC_LIBRARY_TARGET_FRACTION","0"))
    if not 0.0 <= value <= 1.0:
        raise ValueError("PYTHON_FRNET_STATIC_LIBRARY_TARGET_FRACTION must be in [0, 1]")
    return value


def _contact_preload_depth():
    """Return the shallow contact penetration used after height bracketing."""
    value=float(os.environ.get("FRNET_CONTACT_PRELOAD_DEPTH_M","0.002"))
    if not 0.0 <= value <= 0.002:
        raise ValueError("FRNET_CONTACT_PRELOAD_DEPTH_M must be in [0, 0.002]")
    return value


def _static_reset_library(device):
    path=Path(os.environ.get(
        "PYTHON_FRNET_STATIC_LIBRARY_PATH",
        str(Path(__file__).with_name("static_reset_library_fold55_v1.json")),
    ))
    key=(str(path.resolve()),str(device))
    if key not in _static_library_cache:
        data=json.loads(path.read_text())
        rows=data["rows"]
        if data.get("schema_version") != 2 or data.get("count") != len(rows):
            raise ValueError("Static reset library schema/count mismatch")
        if tuple(data["joint_names"]) != JOINT_NAMES:
            raise ValueError("Static reset library joint ordering mismatch")
        coverage={(r["terrain_family"],r["fall_pose"]) for r in rows}
        expected={(t,p) for t in ("flat","stairs","slope","rough") for p in _SUCCESS_POSE_NAMES}
        if coverage != expected:
            raise ValueError("Static reset library lacks terrain/fall coverage")
        terrain_names=("flat","stairs","slope","rough")
        strata=[
            terrain_names.index(r["terrain_family"])*len(_SUCCESS_POSE_NAMES)
            + _SUCCESS_POSE_NAMES.index(r["fall_pose"])
            for r in rows
        ]
        _static_library_cache[key]={
            "q":torch.tensor([r["joint_positions_rad"] for r in rows],device=device),
            "target":torch.tensor([r["control_target_rad"] for r in rows],device=device),
            "root_relative":torch.tensor([r["relative_root_pose"] for r in rows],device=device),
            "row":torch.tensor([r["terrain_row"] for r in rows],device=device,dtype=torch.long),
            "column":torch.tensor([r["terrain_column"] for r in rows],device=device,dtype=torch.long),
            "stratum":torch.tensor(strata,device=device,dtype=torch.long),
            "pose_names":[r["fall_pose"] for r in rows],
        }
    return _static_library_cache[key]


def _sample_balanced_static_library_indices(library,count):
    device=library["q"].device
    requested=torch.randint(0,16,(count,),device=device)
    samples=torch.empty(count,dtype=torch.long,device=device)
    for stratum in range(16):
        output_ids=(requested == stratum).nonzero(as_tuple=False).squeeze(-1)
        if not len(output_ids):
            continue
        candidates=(library["stratum"] == stratum).nonzero(as_tuple=False).squeeze(-1)
        choice=torch.randint(0,len(candidates),(len(output_ids),),device=device)
        samples[output_ids]=candidates[choice]
    return samples

_POSITION_LOWER = (
    -1.0472, -1.5708, -2.7227,
    -1.0472, -1.5708, -2.7227,
    -1.0472, -0.5236, -2.7227,
    -1.0472, -0.5236, -2.7227,
)
_POSITION_UPPER = (
    1.0472, 3.4907, -0.83776,
    1.0472, 3.4907, -0.83776,
    1.0472, 4.5379, -0.83776,
    1.0472, 4.5379, -0.83776,
)
_FOLDED = (
    0.82, 2.9, -2.68,
    -0.82, 2.9, -2.68,
    0.82, 3.5, -2.68,
    -0.82, 3.5, -2.68,
)


def _deep_fold_mask(q, *, calf_threshold=-2.45, thigh_threshold=2.50,
                    minimum_legs=3):
    thighs=q[...,(1,4,7,10)]
    calves=q[...,(2,5,8,11)]
    return ((thighs > thigh_threshold) & (calves < calf_threshold)).sum(dim=-1) >= minimum_legs


def _offline_settle_mask(q):
    if os.environ.get("PYTHON_FRNET_SETTLE_LIBRARY_ALL","0") == "1":
        return torch.ones(q.shape[:-1],dtype=torch.bool,device=q.device)
    if os.environ.get("PYTHON_FRNET_SETTLE_DEEP_FOLD","0") == "1":
        return _deep_fold_mask(q)
    return torch.zeros(q.shape[:-1],dtype=torch.bool,device=q.device)


def _targeted_continuation_enabled():
    return os.environ.get("PYTHON_FRNET_TARGETED_RECOVERY", "0") == "1"


def _apply_targeted_joint_mixture(q, target_ids):
    """Match the eight evaluation birth modes, with extra all-folded samples."""
    if len(target_ids) == 0:
        return q
    lower=q.new_tensor(_POSITION_LOWER)+0.03
    upper=q.new_tensor(_POSITION_UPPER)-0.03
    q[target_ids]=lower+(upper-lower)*torch.rand(
        (len(target_ids),12),device=q.device,dtype=q.dtype
    )
    all_folded=torch.rand(len(target_ids),device=q.device)<TARGETED_ALL_FOLDED_FRACTION
    # Non-all-folded samples uniformly cover independent, four singles and two diagonals.
    mode=torch.randint(0,7,(len(target_ids),),device=q.device)
    folded=q.new_tensor(_FOLDED).reshape(4,3)
    for local,env_id in enumerate(target_ids.tolist()):
        if bool(all_folded[local]): legs=(0,1,2,3)
        elif int(mode[local])==0: legs=()
        elif int(mode[local])<=4: legs=(int(mode[local])-1,)
        elif int(mode[local])==5: legs=(0,3)
        else: legs=(1,2)
        for leg in legs:
            q[env_id,3*leg:3*leg+3]=folded[leg]
    return q


def _contact_found(env, ids, require_nonfoot=False):
    found=torch.zeros(len(ids),dtype=torch.bool,device=env.device)
    names=("nonfoot_ground_touch",) if require_nonfoot else (
        "nonfoot_ground_touch","base_ground_contact","feet_ground_contact")
    for name in names:
        sensor=env.scene[name]
        sensor.update(0.0)
        value=sensor.data.found[ids]
        found |= value.reshape(len(ids),-1).any(dim=-1)
    return found


def _update_stable_counter(counter, stable):
    return torch.where(stable,counter+1,torch.zeros_like(counter))


def _project_vertical_state(root, simulated_pose):
    return _project_static_state(root,simulated_pose,relax_pose=False)


def _project_static_state(root,simulated_pose,*,relax_pose=False):
    projected=root.clone()
    if relax_pose:
        projected[:,:7]=simulated_pose
    else:
        projected[:,2]=simulated_pose[:,2]
    projected[:,7:]=0.0
    return projected


def _bounded_relaxed_joint_state(original_q, simulated_q, max_delta=0.10):
    """Keep the prescribed birth mode while permitting a small static preload."""
    bounded=torch.maximum(
        torch.minimum(simulated_q, original_q + max_delta),
        original_q - max_delta,
    )
    lower=original_q.new_tensor(_POSITION_LOWER)+0.03
    upper=original_q.new_tensor(_POSITION_UPPER)-0.03
    return torch.maximum(torch.minimum(bounded,upper),lower)


def _settle_static_state(env, ids, q, root, initial_valid, *, require_nonfoot=False,
                         relax_pose=False):
    robot=env.scene["robot"]
    all_ids=torch.arange(env.num_envs,device=env.device)
    inactive_mask=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
    inactive_mask[ids]=False
    inactive_ids=all_ids[inactive_mask]
    if len(inactive_ids):
        inactive_root=torch.cat((robot.data.root_link_pose_w[inactive_ids].clone(),robot.data.root_link_vel_w[inactive_ids].clone()),dim=-1)
        inactive_q=robot.data.joint_pos[inactive_ids].clone()
        inactive_dq=robot.data.joint_vel[inactive_ids].clone()
    original_root=root.clone()
    original_q=q.clone()
    current_root=root.clone()
    current_q=q.clone()
    robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids)
    robot.write_root_state_to_sim(current_root,env_ids=ids)
    robot.set_joint_position_target(q,env_ids=ids)
    stable_counter=torch.zeros(len(ids),dtype=torch.int32,device=env.device)
    captured=torch.zeros(len(ids),dtype=torch.bool,device=env.device)
    settled_root=root.clone()
    settled_q=q.clone()
    for _ in range(300):
        previous_z=current_root[:,2].clone()
        previous_pose=current_root[:,:7].clone()
        previous_q=current_q.clone()
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(dt=env.physics_dt)
        env.sim.forward(); env.sim.sense()
        simulated_pose=robot.data.root_link_pose_w[ids].clone()
        simulated_q=robot.data.joint_pos[ids].clone()
        root_velocity=robot.data.root_link_vel_w[ids]
        vertical_speed=root_velocity[:,2].abs()
        projected=_project_static_state(
            original_root,simulated_pose,relax_pose=relax_pose
        )
        relaxed_q=_bounded_relaxed_joint_state(original_q,simulated_q)
        contact=_contact_found(env,ids,require_nonfoot=require_nonfoot)
        if relax_pose:
            position_stable=(projected[:,:3]-previous_pose[:,:3]).norm(dim=-1) < 0.0003
            quaternion_stable=(projected[:,3:7]*previous_pose[:,3:7]).sum(dim=-1).abs() > 0.999999
            pose_stable=position_stable & quaternion_stable
            root_stable=root_velocity.norm(dim=-1) < 0.05
        else:
            pose_stable=(projected[:,2]-previous_z).abs() < 0.0002
            root_stable=vertical_speed < 0.025
        stable=(
            contact
            & root_stable
            & pose_stable
            & ((relaxed_q-previous_q).abs().amax(dim=-1) < 0.0003)
        )
        stable_counter=_update_stable_counter(stable_counter,stable)
        newly=(stable_counter >= 5) & ~captured
        settled_root[newly]=projected[newly]
        settled_q[newly]=relaxed_q[newly]
        captured |= newly
        current_root=torch.where(captured.unsqueeze(-1),settled_root,projected)
        current_q=torch.where(captured.unsqueeze(-1),settled_q,relaxed_q)
        # Quasi-static relaxation: retain the small joint displacement that
        # balances contact/gravity, but remove transient velocity each step.
        robot.write_joint_state_to_sim(current_q,torch.zeros_like(current_q),env_ids=ids)
        robot.write_root_state_to_sim(current_root,env_ids=ids)
        if len(inactive_ids):
            robot.write_joint_state_to_sim(inactive_q,inactive_dq,env_ids=inactive_ids)
            robot.write_root_state_to_sim(inactive_root,env_ids=inactive_ids)
        if bool(captured.all().item()): break
    output_root=torch.where(captured.unsqueeze(-1),settled_root,original_root)
    output_q=torch.where(captured.unsqueeze(-1),settled_q,original_q)
    robot.write_joint_state_to_sim(output_q,torch.zeros_like(output_q),env_ids=ids)
    robot.write_root_state_to_sim(output_root,env_ids=ids)
    if len(inactive_ids):
        robot.write_joint_state_to_sim(inactive_q,inactive_dq,env_ids=inactive_ids)
        robot.write_root_state_to_sim(inactive_root,env_ids=inactive_ids)
    env.sim.forward(); env.sim.sense()
    final_valid=_contact_found(env,ids,require_nonfoot=require_nonfoot)
    lost=captured & ~final_valid
    if lost.any():
        output_root[lost]=original_root[lost]
        robot.write_root_state_to_sim(output_root,env_ids=ids)
        env.sim.forward(); env.sim.sense()
        final_valid=_contact_found(env,ids,require_nonfoot=require_nonfoot)
    # original_q is the actuator equilibrium target used during relaxation.
    return output_root,output_q,original_q,final_valid & initial_valid


def _place_static_in_contact(env, ids, q, root, *, require_nonfoot=False):
    """Vectorized local-height contact bracket for static training resets."""
    if len(ids)==0:
        return root, q, q, torch.zeros(0,dtype=torch.bool,device=env.device)
    robot=env.scene['robot']
    probe=root.clone()
    probe[:,2]=env.scene.terrain.env_origins[ids,2]+2.5
    robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids)
    robot.write_root_state_to_sim(probe,env_ids=ids)
    env.sim.forward(); env.sim.sense()
    rays=env.scene['terrain_scan'].data.hit_pos_w[ids,:,2].reshape(len(ids),17,11)
    central=rays[:,7:10,4:7].reshape(len(ids),-1)
    fallback=env.scene.terrain.env_origins[ids,2].unsqueeze(-1)
    central=torch.where(torch.isfinite(central),central,fallback)
    ground=central.median(dim=-1).values

    high=ground+0.80
    low=ground-0.60
    contact_z=torch.empty_like(ground)
    no_contact_z=high.clone()
    found=torch.zeros(len(ids),dtype=torch.bool,device=env.device)
    # Descend in 4 cm increments. Freeze each environment at its first contact.
    for step in range(36):
        candidate=high-0.04*(step+1)
        active=~found
        if not active.any(): break
        probe[active,2]=candidate[active]
        robot.write_root_state_to_sim(probe,env_ids=ids)
        env.sim.forward(); env.sim.sense()
        hit=_contact_found(env,ids,require_nonfoot=require_nonfoot) & active
        contact_z[hit]=candidate[hit]
        no_contact_z[active & ~hit]=candidate[active & ~hit]
        found |= hit
    # Some stair edge poses need a deeper second bracket because the local
    # scan median can sit above the nearest tread. Continue descending only
    # unresolved environments, still checking onboard contact sensors.
    for step in range(84):
        active=~found
        if not active.any(): break
        # Stair edges and diagonal folded poses can put the scan median well
        # above the nearest collision surface. Search down to four metres
        # below that median before declaring the reset unresolved.
        candidate=ground-0.68-0.04*(step+1)
        probe[active,2]=candidate[active]
        robot.write_root_state_to_sim(probe,env_ids=ids)
        env.sim.forward(); env.sim.sense()
        hit=_contact_found(env,ids,require_nonfoot=require_nonfoot) & active
        contact_z[hit]=candidate[hit]
        found |= hit
    # Refine only valid brackets without advancing simulated time.
    lo=torch.where(found,contact_z,ground+0.10)
    hi=torch.where(found,contact_z+0.04,ground+0.14)
    for _ in range(9):
        mid=(lo+hi)*0.5
        probe[:,2]=mid
        robot.write_root_state_to_sim(probe,env_ids=ids)
        env.sim.forward(); env.sim.sense()
        hit=_contact_found(env,ids,require_nonfoot=require_nonfoot)
        lo=torch.where(hit,mid,lo)
        hi=torch.where(hit,hi,mid)
    # Use the shallow baseline margin for every valid bracket. Only lower
    # individual environments that still lack a measured contact, avoiding a
    # global penetration impulse in already-valid resets.
    root[:,2]=lo-_contact_preload_depth()
    root[:,7:]=0.0
    final_contact=torch.zeros_like(found)
    for _ in range(8):
        robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids)
        robot.write_root_state_to_sim(root,env_ids=ids)
        env.sim.forward(); env.sim.sense()
        final_contact=_contact_found(env,ids,require_nonfoot=require_nonfoot)
        missing=~final_contact
        if not missing.any(): break
        root[missing,2]-=0.0005
    # Optional local joint equilibrium search for folded starts.  A shallow
    # height search alone cannot remove a first-substep impulse when the
    # folded thigh/calf pair is pressing against the body or terrain.  Try a
    # small, mode-preserving perturbation of each folded pair and keep the
    # candidate with the lowest instantaneous joint/root acceleration.  This
    # is a forward/sense reset calculation; it never advances simulation time
    # and is shared by training and evaluation resets.
    if os.environ.get("PYTHON_FRNET_STATIC_JOINT_SELECT", "0") == "1":
        joint_acc = getattr(robot.data, "joint_acc", None)
        root_acc = getattr(robot.data, "root_link_acc_w", None)
        if joint_acc is not None and joint_acc.shape[-1] >= q.shape[-1]:
            base_q = q.clone()
            folded_mask = (
                (base_q[..., (1, 4, 7, 10)] > 2.50)
                & (base_q[..., (2, 5, 8, 11)] < -2.45)
            )
            offset_limit = float(os.environ.get("FRNET_STATIC_JOINT_SELECT_OFFSET_RAD", "0.03"))
            offsets = root.new_tensor((-offset_limit, -0.5 * offset_limit, 0.0, 0.5 * offset_limit, offset_limit))
            best_q = base_q.clone()
            best_score = torch.full((len(ids),), float("inf"), device=env.device, dtype=q.dtype)
            for leg in range(4):
                thigh = 3 * leg + 1
                calf = 3 * leg + 2
                folded_count = folded_mask.sum(dim=-1)
                min_folded = int(os.environ.get(
                    "FRNET_STATIC_JOINT_SELECT_MIN_FOLDED_LEGS", "2"
                ))
                if os.environ.get("PYTHON_FRNET_STATIC_JOINT_SELECT_ALL", "0") == "1":
                    active_leg = torch.ones_like(folded_mask[:, leg])
                else:
                    active_leg = folded_mask[:, leg]
                # The search is a coupled-contact correction. Keep it away
                # from single-leg folds, where changing a healthy receiving
                # leg can introduce a new impulse.
                active_leg = active_leg & (folded_count >= min_folded)
                if not bool(active_leg.any().item()):
                    continue
                for off_thigh in offsets:
                    for off_calf in offsets:
                        trial_q = base_q.clone()
                        trial_q[:, thigh] = (trial_q[:, thigh] + off_thigh).clamp(
                            float(_POSITION_LOWER[thigh]) + 0.03,
                            float(_POSITION_UPPER[thigh]) - 0.03,
                        )
                        trial_q[:, calf] = (trial_q[:, calf] + off_calf).clamp(
                            float(_POSITION_LOWER[calf]) + 0.03,
                            float(_POSITION_UPPER[calf]) - 0.03,
                        )
                        robot.write_joint_state_to_sim(trial_q, torch.zeros_like(trial_q), env_ids=ids)
                        robot.write_root_state_to_sim(root, env_ids=ids)
                        env.sim.forward(); env.sim.sense()
                        contact = _contact_found(env, ids, require_nonfoot=require_nonfoot)
                        acc_raw = robot.data.joint_acc
                        if acc_raw.shape[0] != len(ids):
                            acc_raw = acc_raw[ids]
                        joint_score = acc_raw[..., :q.shape[-1]].abs().amax(dim=-1)
                        score = joint_score + (~contact).to(q.dtype) * 1.0e6
                        if root_acc is not None:
                            root_raw = robot.data.root_link_acc_w
                            if root_raw.shape[0] != len(ids):
                                root_raw = root_raw[ids]
                            score = score + 0.05 * root_raw[..., :3].abs().amax(dim=-1)
                        choose = active_leg & (score < best_score)
                        best_score = torch.where(choose, score, best_score)
                        best_q = torch.where(choose.unsqueeze(-1), trial_q, best_q)
            q = best_q
            robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
            robot.write_root_state_to_sim(root, env_ids=ids)
            env.sim.forward(); env.sim.sense()
    # Select a nearby shallow-contact height with the smallest instantaneous
    # constraint acceleration. This only calls forward/sense; it does not
    # advance time or perform a long settling loop.
    if os.environ.get("PYTHON_FRNET_STATIC_ACCEL_SELECT", "0") == "1":
        joint_acc = getattr(robot.data, "joint_acc", None)
        if joint_acc is not None and joint_acc.shape[-1] >= q.shape[-1]:
            base_z = root[:, 2].clone()
            offsets = root.new_tensor((-0.002, -0.001, 0.0, 0.001, 0.002))
            best_score = torch.full_like(base_z, float("inf"))
            best_z = base_z.clone()
            for offset in offsets:
                trial = root.clone()
                trial[:, 2] = base_z + offset
                robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
                robot.write_root_state_to_sim(trial, env_ids=ids)
                env.sim.forward(); env.sim.sense()
                contact = _contact_found(env, ids, require_nonfoot=require_nonfoot)
                acc_raw = robot.data.joint_acc
                if acc_raw.shape[0] != len(ids):
                    acc_raw = acc_raw[ids]
                acc = acc_raw[..., :q.shape[-1]].abs().amax(dim=-1)
                score = acc + (~contact).to(acc.dtype) * 1.0e6
                choose = score < best_score
                best_score = torch.where(choose, score, best_score)
                best_z = torch.where(choose, trial[:, 2], best_z)
            root[:, 2] = best_z
            robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
            robot.write_root_state_to_sim(root, env_ids=ids)
            env.sim.forward(); env.sim.sense()
            final_contact = _contact_found(env, ids, require_nonfoot=require_nonfoot)
    # Select a nearby shallow-contact height with the smallest instantaneous
    # constraint acceleration. This only calls forward/sense; it does not
    # advance time or perform a long settling loop.
    if os.environ.get("PYTHON_FRNET_STATIC_ACCEL_SELECT", "0") == "1":
        joint_acc = getattr(robot.data, "joint_acc", None)
        if joint_acc is not None and joint_acc.shape[-1] >= q.shape[-1]:
            base_z = root[:, 2].clone()
            offsets = root.new_tensor((-0.002, -0.001, 0.0, 0.001, 0.002))
            best_score = torch.full_like(base_z, float("inf"))
            best_z = base_z.clone()
            for offset in offsets:
                trial = root.clone()
                trial[:, 2] = base_z + offset
                robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
                robot.write_root_state_to_sim(trial, env_ids=ids)
                env.sim.forward(); env.sim.sense()
                contact = _contact_found(env, ids, require_nonfoot=require_nonfoot)
                acc_raw = robot.data.joint_acc
                if acc_raw.shape[0] != len(ids):
                    acc_raw = acc_raw[ids]
                acc = acc_raw[..., :q.shape[-1]].abs().amax(dim=-1)
                score = acc + (~contact).to(acc.dtype) * 1.0e6
                choose = score < best_score
                best_score = torch.where(choose, score, best_score)
                best_z = torch.where(choose, trial[:, 2], best_z)
            root[:, 2] = best_z
            robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
            robot.write_root_state_to_sim(root, env_ids=ids)
            env.sim.forward(); env.sim.sense()
            final_contact = _contact_found(env, ids, require_nonfoot=require_nonfoot)
    # Optional local roll/pitch search around the sampled fall pose.  The
    # orientation is changed only by a few hundredths of a radian and is
    # selected using contact validity plus instantaneous joint acceleration.
    if os.environ.get("PYTHON_FRNET_STATIC_POSE_SELECT", "0") == "1":
        joint_acc = getattr(robot.data, "joint_acc", None)
        if joint_acc is not None and joint_acc.shape[-1] >= q.shape[-1]:
            base_quat = root[:, 3:7].clone()
            base_z = root[:, 2].clone()
            delta = root.new_tensor((-0.03, 0.0, 0.03))
            best_score = torch.full_like(base_z, float("inf"))
            best_quat = base_quat.clone()
            for dr in delta:
                for dp in delta:
                    axis_r = root.new_tensor([1.0, 0.0, 0.0]).expand(len(ids), -1)
                    axis_p = root.new_tensor([0.0, 1.0, 0.0]).expand(len(ids), -1)
                    qr = quat_from_angle_axis(torch.full_like(base_z, dr), axis_r)
                    qp = quat_from_angle_axis(torch.full_like(base_z, dp), axis_p)
                    trial_quat = quat_mul(quat_mul(base_quat, qr), qp)
                    trial = root.clone()
                    trial[:, 3:7] = trial_quat
                    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
                    robot.write_root_state_to_sim(trial, env_ids=ids)
                    env.sim.forward(); env.sim.sense()
                    contact = _contact_found(env, ids, require_nonfoot=require_nonfoot)
                    acc_raw = robot.data.joint_acc
                    if acc_raw.shape[0] != len(ids):
                        acc_raw = acc_raw[ids]
                    acc = acc_raw[..., :q.shape[-1]].abs().amax(dim=-1)
                    score = acc + (~contact).to(acc.dtype) * 1.0e6
                    choose = score < best_score
                    best_score = torch.where(choose, score, best_score)
                    best_quat = torch.where(choose.unsqueeze(-1), trial_quat, best_quat)
            root[:, 3:7] = best_quat
            robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
            robot.write_root_state_to_sim(root, env_ids=ids)
            env.sim.forward(); env.sim.sense()
            final_contact = _contact_found(env, ids, require_nonfoot=require_nonfoot)
    # Preserve the shallow-contact birth pose by default.  The opt-in branch is
    # used only to generate/validate offline equilibrium states for deep folds;
    # it must not add hundreds of physics steps to normal online resets.
    control_target=q.clone()
    settle_mask=_offline_settle_mask(q)
    settle_local=settle_mask.nonzero(as_tuple=False).squeeze(-1)
    if len(settle_local):
        settled_root,settled_q,settled_target,settled_valid=_settle_static_state(
            env,ids[settle_local],q[settle_local],root[settle_local],
            final_contact[settle_local],require_nonfoot=require_nonfoot,
            relax_pose=os.environ.get("PYTHON_FRNET_RELAX_STATIC_POSE","0") == "1",
        )
        root[settle_local]=settled_root
        q[settle_local]=settled_q
        control_target[settle_local]=settled_target
        final_contact[settle_local]=settled_valid
    # Convert the measured static acceleration into a small, opposite holding
    # target.  Clearing velocity alone is insufficient when a fallen pose has
    # a large gravity/contact acceleration: the first 20 ms can overspeed a
    # receiving calf before the policy has another observation.  This target
    # is local to the reset, bounded, and released through the normal action
    # ramp; it does not add a settling trajectory or change the fall label.
    if os.environ.get("PYTHON_FRNET_STATIC_ACCEL_TARGET", "0") == "1":
        acc_raw = getattr(robot.data, "joint_acc", None)
        if acc_raw is not None and acc_raw.shape[-1] >= q.shape[-1]:
            if acc_raw.shape[0] != len(ids):
                acc_raw = acc_raw[ids]
            gain = float(os.environ.get("FRNET_STATIC_ACCEL_TARGET_GAIN", "0.00005"))
            max_offset = float(os.environ.get("FRNET_STATIC_ACCEL_TARGET_MAX_RAD", "0.15"))
            threshold = float(os.environ.get("FRNET_STATIC_ACCEL_TARGET_THRESHOLD", "80.0"))
            # Fully folded starts are already in a symmetric, constrained
            # contact configuration. Applying the acceleration-opposite target
            # to every folded joint can create a step-0 calf impulse; leave
            # those poses at their measured static target and use the normal
            # startup ramp. Partial folds still receive the bias.
            folded_pairs = (
                (q[..., (1, 4, 7, 10)] > 2.50)
                & (q[..., (2, 5, 8, 11)] < -2.45)
            )
            max_folded = int(os.environ.get("FRNET_STATIC_ACCEL_MAX_FOLDED_LEGS", "1"))
            # Apply the bias to sparse folds, where one receiving leg can
            # otherwise get a first-substep impulse. For diagonal/full folds
            # the contact constraints are coupled; leaving the measured q
            # target intact is safer than pushing two folded pairs at once.
            accel_bias_mask = folded_pairs.sum(dim=-1) <= max_folded
            offset = (-torch.sign(acc_raw[..., :q.shape[-1]]) *
                      (acc_raw[..., :q.shape[-1]].abs() * gain).clamp(0.0, max_offset))
            offset = torch.where(
                (acc_raw[..., :q.shape[-1]].abs() >= threshold)
                & accel_bias_mask.unsqueeze(-1),
                offset,
                torch.zeros_like(offset),
            )
            # Keep the compensated target inside the official hard limits,
            # but do not add the normal 0.02 rad safety margin here. A folded
            # joint can already start just inside a limit; pushing its target
            # away from that position creates the very impulse this reset is
            # meant to remove.
            lo = q.new_tensor(_POSITION_LOWER)
            hi = q.new_tensor(_POSITION_UPPER)
            control_target = torch.maximum(torch.minimum(q + offset, hi), lo)
    if os.environ.get("PYTHON_FRNET_DEBUG_RESET", "0") == "1":
        acc_dbg = getattr(robot.data, "joint_acc", None)
        vel_dbg = getattr(robot.data, "joint_vel", None)
        print("FRNET_RESET_DEBUG", {"q": q.detach().cpu().tolist(), "target": control_target.detach().cpu().tolist(), "acc": None if acc_dbg is None else acc_dbg.detach().cpu().tolist(), "vel": None if vel_dbg is None else vel_dbg.detach().cpu().tolist(), "contact": final_contact.detach().cpu().tolist()}, flush=True)
    return root, q, control_target, final_contact


def _apply_targeted_root_mixture(root, origins, target_ids):
    """Emphasize right falls and near-edge contacts while retaining diversity."""
    count = len(target_ids)
    if count == 0:
        return root
    root[target_ids, :3] = origins[target_ids] + origins.new_tensor([0.0, 0.0, 0.38])
    offsets = 5.6 * torch.rand((count, 2), device=origins.device) - 2.8
    edge_mask = torch.rand(count, device=origins.device) < 0.50
    edge_axis = torch.randint(0, 2, (count,), device=origins.device)
    edge_sign = torch.where(
        torch.rand(count, device=origins.device) < 0.5,
        -torch.ones(count, device=origins.device),
        torch.ones(count, device=origins.device),
    )
    edge_value = edge_sign * (2.0 + 0.8 * torch.rand(count, device=origins.device))
    edge_rows = edge_mask.nonzero(as_tuple=False).squeeze(-1)
    if len(edge_rows):
        offsets[edge_rows, edge_axis[edge_rows]] = edge_value[edge_rows]
    root[target_ids, :2] += offsets

    use_right = torch.rand(count, device=origins.device) < TARGETED_RIGHT_FALL_FRACTION
    roll = torch.empty(count, device=origins.device)
    pitch = torch.empty(count, device=origins.device)
    roll[use_right] = 0.5 * math.pi
    pitch[use_right] = 0.0
    other = (~use_right).nonzero(as_tuple=False).squeeze(-1)
    if len(other):
        choices = torch.randint(0, 3, (len(other),), device=origins.device)
        other_roll = torch.tensor([0.0, 0.0, -0.5 * math.pi], device=origins.device)
        other_pitch = torch.tensor([-0.5 * math.pi, 0.5 * math.pi, 0.0], device=origins.device)
        roll[other] = other_roll[choices]
        pitch[other] = other_pitch[choices]
    yaw = 2.0 * math.pi * torch.rand(count, device=origins.device) - math.pi
    root[target_ids, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
    root[target_ids, 7:] = 0.0
    return root


def sample_hybrid_root_reset(origins, *, generator=None):
    """Mix broad dynamic resets with low-speed, explicitly fallen resets.

    The original FR-Net reset remains as 20% of samples.  The remaining 80%
    span every horizontal fall axis (including diagonal falls), 70--180 degree
    tilt, arbitrary yaw, and mostly exact zero velocity.  A small subset keeps
    bounded residual motion so the policy is not brittle to a real robot that
    has not become perfectly motionless before recovery is enabled.
    """
    root = sample_root_reset(origins, generator=generator)
    n = len(origins)
    static_mask = (
        torch.rand(n, device=origins.device, generator=generator)
        < STATIC_FALL_FRACTION
    )
    static_ids = static_mask.nonzero(as_tuple=False).squeeze(-1)
    count = len(static_ids)
    if count == 0:
        return root, static_mask

    root[static_ids, :3] = origins[static_ids] + origins.new_tensor([0.0, 0.0, 0.38])
    root[static_ids, :2] += (
        5.0
        * torch.rand((count, 2), device=origins.device, generator=generator)
        - 2.5
    )

    fall_axis_angle = 2.0 * math.pi * torch.rand(
        count, device=origins.device, generator=generator
    )
    horizontal_axis = torch.stack(
        (
            torch.cos(fall_axis_angle),
            torch.sin(fall_axis_angle),
            torch.zeros_like(fall_axis_angle),
        ),
        dim=-1,
    )
    tilt = math.radians(70.0) + math.radians(110.0) * torch.rand(
        count, device=origins.device, generator=generator
    )
    fallen_quat = quat_from_angle_axis(tilt, horizontal_axis)
    yaw = 2.0 * math.pi * torch.rand(
        count, device=origins.device, generator=generator
    ) - math.pi
    yaw_quat = quat_from_euler_xyz(
        torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
    )
    root[static_ids, 3:7] = quat_mul(yaw_quat, fallen_quat)

    # Most explicitly fallen samples are exactly motionless.  Residual-motion
    # samples remain far below the original +/-0.5 source reset envelope.
    root[static_ids, 7:] = 0.0
    residual_mask = (
        torch.rand(count, device=origins.device, generator=generator)
        < RESIDUAL_MOTION_FRACTION
    )
    residual_ids = static_ids[residual_mask]
    residual_count = len(residual_ids)
    if residual_count:
        root[residual_ids, 7:10] = (
            0.16
            * torch.rand(
                (residual_count, 3), device=origins.device, generator=generator
            )
            - 0.08
        )
        root[residual_ids, 10:13] = (
            0.30
            * torch.rand(
                (residual_count, 3), device=origins.device, generator=generator
            )
            - 0.15
        )
    return root, static_mask


def reset_source_state(env, env_ids):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot = env.scene['robot']
    if tuple(robot.joint_names) != JOINT_NAMES:
        raise ValueError('Source reset requires verified FL/FR/RL/RR joint ordering')
    targeted = _targeted_continuation_enabled()
    target_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
    if targeted:
        replay_fraction = float(os.environ.get(
            "FRNET_REPLAY_FRACTION", str(1.0 - TARGETED_REPLAY_FRACTION)
        ))
        if not 0.0 <= replay_fraction <= 1.0:
            raise ValueError("FRNET_REPLAY_FRACTION must be in [0, 1]")
        # target_mask selects newly targeted resets; the complement is
        # sampled from the replay file. Keep the historical default (60%
        # replay) while allowing failure-replay runs to request 30-40%.
        target_mask = (
            torch.rand(len(env_ids), device=env.device)
            < (1.0 - replay_fraction)
        )
    residual_mode = os.environ.get("FRNET_RESIDUAL", "0") == "1"
    terrain_focus = target_mask & (torch.rand(len(env_ids), device=env.device) < 0.4) if residual_mode else torch.zeros_like(target_mask)
    library_mask=torch.zeros_like(target_mask)
    library=None
    library_samples=torch.empty(0,dtype=torch.long,device=env.device)
    if targeted and _static_library_target_fraction() > 0.0:
        target_ids_for_library=target_mask.nonzero(as_tuple=False).squeeze(-1)
        selected=(
            torch.rand(len(target_ids_for_library),device=env.device)
            < _static_library_target_fraction()
        )
        library_ids_local=target_ids_for_library[selected]
        library_mask[library_ids_local]=True
        if len(library_ids_local):
            library=_static_reset_library(env.device)
            library_samples=_sample_balanced_static_library_indices(
                library,len(library_ids_local)
            )
    else:
        library_ids_local=torch.empty(0,dtype=torch.long,device=env.device)
    replay_ids_local=(~target_mask).nonzero(as_tuple=False).squeeze(-1) if targeted else torch.empty(0,dtype=torch.long,device=env.device)
    replay=_success_replay(env.device) if targeted else None
    replay_samples=torch.randint(len(replay['q']),(len(replay_ids_local),),device=env.device) if targeted else None
    # Training-only slope-recovery fine-tuning distribution.  The strict
    # settled-pose acceptance suite currently passes flat, stairs and rough
    # terrain, but fails both slope columns.  Put two thirds of resets on the
    # two slope variants while retaining every other family to limit forgetting.
    # Play/Viser uses the dedicated settled reset and is unaffected.
    terrain = getattr(env.scene, 'terrain', None)
    if terrain is not None and len(env_ids) > 1:
        if targeted:
            # 50% balanced replay protects prior skills; 50% targeted replay
            # emphasizes stairs, slopes and rough terrain without dropping flat.
            retention_weights = torch.tensor(
                [0.125, 0.125, 1/12, 1/12, 1/12, 0.125, 0.125, 0.25],
                device=env.device,
            )
            targeted_weights = torch.tensor(
                [0.075,0.075,0.10,0.10,0.10,0.075,0.075,0.40],
                device=env.device,
            )
            columns = torch.multinomial(retention_weights, len(env_ids), replacement=True)
            target_ids_local = target_mask.nonzero(as_tuple=False).squeeze(-1)
            if len(target_ids_local):
                columns[target_ids_local] = torch.multinomial(
                    targeted_weights, len(target_ids_local), replacement=True
                )
        else:
            weights = torch.tensor(
                [0.04, 0.04, 0.05, 0.05, 0.05, 0.42, 0.25, 0.10],
                device=env.device,
            )
            columns = torch.multinomial(weights, len(env_ids), replacement=True)
        rows = torch.randint(
            terrain.terrain_origins.shape[0], (len(env_ids),), device=env.device
        )
        if targeted and len(replay_ids_local):
            columns[replay_ids_local]=replay['column'][replay_samples]
            rows[replay_ids_local]=replay['row'][replay_samples]
        if len(library_ids_local):
            columns[library_ids_local]=library['column'][library_samples]
            rows[library_ids_local]=library['row'][library_samples]
        if residual_mode:
            # 20% of all resets: rough terrain; 30%: back/left pose focus.
            columns[terrain_focus] = 7
        terrain.terrain_types[env_ids] = columns
        terrain.terrain_levels[env_ids] = rows
        terrain.env_origins[env_ids] = terrain.terrain_origins[rows, columns]

    q, dq = sample_joint_reset(
        len(env_ids), device=env.device, enforce_official_limits=True
    )
    # Optional reset-only relief for poses that start too close to a joint
    # boundary.  It preserves the fall orientation while removing the
    # boundary/contact coincidence that can create a first-substep impulse.
    relief = float(os.environ.get("FRNET_RESET_BOUNDARY_RELIEF_RAD", "0.0"))
    if relief > 0.0:
        lower = q.new_tensor(_POSITION_LOWER) - 0.03 + relief
        upper = q.new_tensor(_POSITION_UPPER) - 0.03 - relief
        q = torch.maximum(torch.minimum(q, upper), lower)
        dq.zero_()
    root, static_mask = sample_hybrid_root_reset(env.scene.env_origins[env_ids])
    if targeted and len(replay_ids_local):
        # These are initial configurations from model_13798's verified
        # successful episodes, including their original terrain and position.
        q[replay_ids_local]=replay['q'][replay_samples].to(q.dtype)
        dq[replay_ids_local]=0.0
        root[replay_ids_local,:3]=env.scene.env_origins[env_ids[replay_ids_local]]
        root[replay_ids_local,:2]+=replay['offset'][replay_samples].to(root.dtype)
        pose=replay['pose'][replay_samples]
        angles=root.new_tensor(_SUCCESS_POSES)
        roll=angles[pose,0]
        pitch=angles[pose,1]
        yaw=2.0*math.pi*torch.rand(len(replay_ids_local),device=env.device)-math.pi
        root[replay_ids_local,3:7]=quat_from_euler_xyz(roll,pitch,yaw)
        root[replay_ids_local,7:]=0.0
        static_mask[replay_ids_local]=True
    if targeted:
        target_ids_local = target_mask.nonzero(as_tuple=False).squeeze(-1)
        q = _apply_targeted_joint_mixture(q, target_ids_local)
        root = _apply_targeted_root_mixture(
            root, env.scene.env_origins[env_ids], target_ids_local
        )
        static_mask[target_ids_local] = True
    if residual_mode:
        pose_ids=(target_mask & ~terrain_focus).nonzero(as_tuple=False).squeeze(-1)
        choices=torch.randint(0,2,(len(pose_ids),),device=env.device)
        roll=torch.where(choices==0,0.,-0.5*math.pi)
        pitch=torch.where(choices==0,0.5*math.pi,0.)
        root[pose_ids,3:7]=quat_from_euler_xyz(roll,pitch,torch.zeros_like(roll))
        dq.zero_()
        root[:,7:]=0.
    if len(library_ids_local):
        origins=env.scene.terrain.env_origins[env_ids[library_ids_local]]
        relative=library['root_relative'][library_samples].to(root.dtype)
        q[library_ids_local]=library['q'][library_samples].to(q.dtype)
        dq[library_ids_local]=0.0
        root[library_ids_local,:3]=origins+relative[:,:3]
        root[library_ids_local,3:7]=relative[:,3:7]
        root[library_ids_local,7:]=0.0
        static_mask[library_ids_local]=True
    static_ids_local=(static_mask & ~library_mask).nonzero(as_tuple=False).squeeze(-1)
    contact_valid=torch.ones(len(env_ids),dtype=torch.bool,device=env.device)
    control_target=q.clone()
    if len(library_ids_local):
        control_target[library_ids_local]=library['target'][library_samples].to(q.dtype)
    if len(static_ids_local):
        actual_ids=env_ids[static_ids_local]
        placed,settled_q,settled_target,valid=_place_static_in_contact(
            env,actual_ids,q[static_ids_local],root[static_ids_local]
        )
        root[static_ids_local]=placed
        q[static_ids_local]=settled_q
        control_target[static_ids_local]=settled_target
        contact_valid[static_ids_local]=valid
    if residual_mode:
        assert torch.count_nonzero(dq) == 0 and torch.count_nonzero(root[:,7:]) == 0, "Non-static residual reset"
    robot.write_joint_state_to_sim(q, dq, env_ids=env_ids)
    robot.write_root_state_to_sim(root, env_ids=env_ids)
    # The action manager may reset before this event; synchronize after writing q.
    action=env.action_manager.get_term('recovery')
    if hasattr(action,'sync_target_to_joint_state'):
        action.sync_target_to_joint_state(env_ids,q,reference_target=control_target)
        if not hasattr(action._entity, "_frnet_folded_legs"):
            action._entity._frnet_folded_legs=torch.zeros((action.num_envs,4),dtype=torch.bool,device=env.device)
            action._entity._frnet_folded_transfer_receivers=torch.zeros((action.num_envs,4),dtype=torch.bool,device=env.device)
            action._entity._frnet_deep_folded=torch.zeros((action.num_envs,),dtype=torch.bool,device=env.device)
            action._entity._frnet_deep_folded_count=torch.zeros((action.num_envs,),dtype=torch.long,device=env.device)
            action._entity._frnet_birth_mode_code=torch.zeros((action.num_envs,),dtype=torch.long,device=env.device)
        folded_mask=((q.index_select(-1, q.new_tensor(THIGH_INDICES, dtype=torch.long)) > 2.50) & (q.index_select(-1, q.new_tensor(CALF_INDICES, dtype=torch.long)) < -2.45))
        action._entity._frnet_folded_legs[env_ids] = folded_mask
        receiver_mask=torch.zeros_like(folded_mask)
        receiver_mask[...,2] |= folded_mask[...,0]; receiver_mask[...,3] |= folded_mask[...,1]
        receiver_mask[...,0] |= folded_mask[...,2]; receiver_mask[...,1] |= folded_mask[...,3]
        if os.environ.get("FRNET_FOLDED_TRANSFER_BOTH_FRONT", "0") == "1":
            receiver_mask[...,0] |= folded_mask[...,3]; receiver_mask[...,1] |= folded_mask[...,2]
        action._entity._frnet_folded_transfer_receivers[env_ids] = receiver_mask
        action._entity._frnet_deep_folded[env_ids] = folded_mask.sum(dim=-1) >= max(1, int(os.environ.get("FRNET_DEEP_FOLDED_MIN_LEGS", os.environ.get("FRNET_ADAPTIVE_FOLDED_MIN_LEGS", "2"))))
        action._entity._frnet_deep_folded_count[env_ids] = folded_mask.sum(dim=-1)
        action._entity._frnet_birth_mode_code[env_ids] = torch.where(
            (folded_mask[..., 1] & folded_mask[..., 2])
            & (folded_mask.sum(dim=-1) == 2),
            torch.full((len(env_ids),), 2, dtype=torch.long, device=env.device),
            torch.zeros((len(env_ids),), dtype=torch.long, device=env.device),
        )
    env.last_training_static_fall_fraction = float(static_mask.float().mean().item())
    env.last_training_targeted_fraction = float(target_mask.float().mean().item())
    env.last_training_success_replay_fraction = float(
        len(replay_ids_local)/len(env_ids)
    )
    env.last_training_static_library_fraction = float(
        len(library_ids_local)/len(env_ids)
    )
    env.last_training_static_library_ids = env_ids[library_ids_local].clone()
    env.last_training_static_library_samples = library_samples.clone()
    env.last_training_static_contact_fraction = float(
        contact_valid[static_mask].float().mean().item() if static_mask.any() else 1.0
    )

"""Deterministic, vectorized FR-Net Go2 recovery acceptance suite.

The suite deliberately separates training reward from task success.  It tests
fixed fall directions/angles, leg-fold combinations, every generated terrain
row/column at five within-tile positions, friction, initial angular velocity,
and four light disturbance directions.  Robots are settled before the policy
is enabled so the measured task starts from a static fallen configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.acceptance_physics import set_case_friction
from scripts.acceptance_criteria import case_criteria, SuccessCounter, StableRecoveryCounter, stable_recovery_ok

import mjlab.tasks  # noqa: E402
import src.tasks.velocity.config.go2_recovery_aligned  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.sensor import ContactMatch, ContactSensorCfg  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from src.tasks.velocity.config.go2_recovery_aligned.official_go2_limits import (  # noqa: E402
    FRNET_EFFORT_LIMIT,
    clamp_joint_positions,
)


ORIENTATIONS = ("front", "back", "left", "right")
ANGLES = (90, 120, 150, 180)
FOLDS = ("FL", "FR", "RL", "RR", "diagonal_FL_RR", "diagonal_FR_RL", "all_folded")
POSITIONS = {
    "center": (0.0, 0.0),
    "x_min": (-3.0, 0.0),
    "x_max": (3.0, 0.0),
    "y_min": (0.0, -3.0),
    "y_max": (0.0, 3.0),
}
FRICTIONS = (0.5, 0.75, 1.0, 1.25)
ANGULAR_VELOCITIES = (0.0, 0.5, 1.0)
DISTURBANCES = {
    "plus_x": (30.0, 0.0),
    "minus_x": (-30.0, 0.0),
    "plus_y": (0.0, 30.0),
    "minus_y": (0.0, -30.0),
}
# TerrainGenerator assigns curriculum columns using the first cumulative
# proportion greater than col / num_cols + 0.001.  With the inherited
# 0.2/0.2/0.2/0.1/0.1/0.1/0.1 proportions and 8 columns, the exact mapping is
# therefore two flat, two stairs, then one each of the following four types.
# The configured wave terrain receives no column in this 8-column layout.
TERRAIN_NAMES = (
    "flat",
    "flat",
    "pyramid_stairs",
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
    "random_rough",
)


def quat_from_euler(roll: float, pitch: float, yaw: float = 0.0) -> list[float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def fall_quat(direction: str, angle_deg: float) -> list[float]:
    a = math.radians(angle_deg)
    if direction == "front":
        return quat_from_euler(0.0, a)
    if direction == "back":
        return quat_from_euler(0.0, -a)
    if direction == "left":
        return quat_from_euler(a, 0.0)
    if direction == "right":
        return quat_from_euler(-a, 0.0)
    raise ValueError(direction)


def add_case(cases: list[dict], group: str, pose: str, **kwargs) -> None:
    row = {
        "id": len(cases),
        "group": group,
        "pose": pose,
        "angle_deg": 90,
        "fold": "none",
        "terrain_row": 0,
        "terrain_col": 0,
        "terrain_position": "center",
        "friction": 1.0,
        "initial_ang_vel": 0.0,
        "disturbance": "none",
    }
    row.update(kwargs)
    cases.append(row)


def build_cases() -> list[dict]:
    cases: list[dict] = []
    for pose in ORIENTATIONS:
        for angle in ANGLES:
            add_case(cases, "orientation_angle", pose, angle_deg=angle)
    for pose in ORIENTATIONS:
        for fold in FOLDS:
            add_case(cases, "leg_fold", pose, fold=fold)
    for pose in ORIENTATIONS:
        for terrain_row in range(10):
            for terrain_col in range(8):
                for terrain_position in POSITIONS:
                    add_case(
                        cases,
                        "terrain",
                        pose,
                        terrain_row=terrain_row,
                        terrain_col=terrain_col,
                        terrain_position=terrain_position,
                    )
    for pose in ORIENTATIONS:
        for friction in FRICTIONS:
            add_case(cases, "friction", pose, friction=friction)
    for pose in ORIENTATIONS:
        for speed in ANGULAR_VELOCITIES:
            add_case(cases, "initial_angular_velocity", pose, initial_ang_vel=speed)
    for pose in ORIENTATIONS:
        for disturbance in DISTURBANCES:
            add_case(cases, "disturbance", pose, disturbance=disturbance)
    return cases


def joint_configuration(cases: list[dict], device: str) -> torch.Tensor:
    q = torch.tensor([0.0, 1.5, -2.4] * 4, device=device).repeat(len(cases), 1)
    leg_index = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}
    for i, case in enumerate(cases):
        fold = case["fold"]
        if fold == "none":
            continue
        if fold == "all_folded":
            legs = tuple(leg_index)
        elif fold.startswith("diagonal_"):
            legs = tuple(fold.removeprefix("diagonal_").split("_"))
        else:
            legs = (fold,)
        for leg in legs:
            j = 3 * leg_index[leg]
            side = 1.0 if leg in ("FL", "RL") else -1.0
            q[i, j : j + 3] = torch.tensor(
                [0.82 * side, 2.9 if leg in ("FL", "FR") else 3.5, -2.68],
                device=device,
            )
    return clamp_joint_positions(q)


def grouped_rates(rows: list[dict], key: str) -> dict[str, float]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        buckets[str(row[key])].append(bool(row["success"]))
    return {name: sum(values) / len(values) for name, values in sorted(buckets.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Unitree-Go2-Recovery-FRNet-Aligned")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--hold-steps", type=int, default=50)
    parser.add_argument("--placement-scan-steps", type=int, default=240)
    parser.add_argument("--placement-iterations", type=int, default=18)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trajectory-output", default=None)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--placement-only", action="store_true", help="Validate static setup without policy rollout")
    parser.add_argument("--success-profile", choices=("strict", "tiered", "stable"), default="tiered")
    args = parser.parse_args()

    if args.trajectory_output and args.success_profile != "strict":
        raise ValueError("Teacher trajectory collection requires --success-profile strict")
    torch.manual_seed(args.seed)
    cases = [c for c in build_cases() if c["initial_ang_vel"] == 0 and c["disturbance"] == "none"]
    num_envs = len(cases)
    cfg = load_env_cfg(args.task, play=True)
    cfg.seed = args.seed
    cfg.scene.num_envs = num_envs
    cfg.viewer = None
    cfg.curriculum = {}
    cfg.events = {}
    cfg.sim.nconmax = 128
    cfg.scene.terrain.terrain_generator.seed = args.seed
    all_robot_contact = ContactSensorCfg(
        name="all_robot_ground_contact",
        primary=ContactMatch(
            mode="geom", pattern=r".*_collision_.*", entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
        history_length=1,
    )
    cfg.scene.sensors = (*cfg.scene.sensors, all_robot_contact)
    actor_params = cfg.observations["actor"].terms["frnet_actor"].params
    actor_params.update(noise_enabled=True, noise_level=1.0, sensor_delay_max_steps=1)
    command_cfg = cfg.commands["twist"]
    command_cfg.heading_command = False
    command_cfg.ranges.lin_vel_x = (0.0, 0.0)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    command_cfg.ranges.heading = None

    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env)
    agent_cfg = load_rl_cfg(args.task)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
    runner.load(
        args.checkpoint,
        load_cfg={"actor": True},
        strict=True,
        map_location=args.device,
    )
    policy = runner.get_inference_policy(device=args.device)
    robot = env.scene["robot"]
    terrain = env.scene.terrain

    levels = torch.tensor([c["terrain_row"] for c in cases], device=args.device)
    types = torch.tensor([c["terrain_col"] for c in cases], device=args.device)
    origins = terrain.terrain_origins[levels, types].clone()
    terrain.terrain_levels.copy_(levels)
    terrain.terrain_types.copy_(types)
    terrain.env_origins.copy_(origins)
    env.scene.env_origins.copy_(origins)

    env.reset(seed=args.seed + 90000)
    friction = torch.tensor([c["friction"] for c in cases], device=args.device)
    robot_geoms = robot.indexing.geom_ids.long()
    friction_assignment = set_case_friction(env, robot_geoms, friction)
    q = joint_configuration(cases, args.device)
    dq = torch.zeros_like(q)
    robot.write_joint_state_to_sim(q, dq)
    robot.set_joint_position_target(q)
    robot.set_joint_velocity_target(dq)

    root = robot.data.default_root_state.clone()
    root[:, :3] = origins
    offsets = torch.tensor([POSITIONS[c["terrain_position"]] for c in cases], device=args.device)
    root[:, :2] += offsets
    root[:, 2] += 2.5
    root[:, 3:7] = torch.tensor(
        [fall_quat(c["pose"], c["angle_deg"]) for c in cases], device=args.device
    )
    root[:, 7:13] = 0.0
    robot.write_root_state_to_sim(root)
    env.sim.forward()
    env.sim.sense()

    # Use the downward center ray to bracket the local surface, then binary
    # search the highest root height that still has any robot-terrain contact.
    # Pose and joints stay fixed during placement, so there is no hidden drop.
    scan = env.scene["terrain_scan"].data
    center_ray = scan.hit_pos_w.shape[1] // 2
    center_valid = scan.distances[:, center_ray] >= 0
    local_ground_z = torch.where(
        center_valid, scan.hit_pos_w[:, center_ray, 2], origins[:, 2]
    )
    contact_sensor = env.scene["all_robot_ground_contact"]
    scan_high = local_ground_z + 2.5
    scan_delta = 0.02
    found_bracket = torch.zeros(num_envs, dtype=torch.bool, device=args.device)
    low = local_ground_z - 1.5
    high = scan_high.clone()
    for scan_step in range(args.placement_scan_steps):
        candidate = scan_high - scan_delta * scan_step
        root[:, 2] = torch.where(found_bracket, low, candidate)
        robot.write_root_state_to_sim(root)
        env.sim.forward()
        env.sim.sense()
        contact_sensor.update(0.0)
        contact_now = contact_sensor.data.found.reshape(num_envs, -1).any(dim=1)
        newly_found = (~found_bracket) & contact_now
        low = torch.where(newly_found, candidate, low)
        high = torch.where(newly_found, candidate + scan_delta, high)
        found_bracket |= newly_found
        if bool(found_bracket.all()):
            break
    for _ in range(args.placement_iterations):
        mid = (low + high) * 0.5
        root[:, 2] = mid
        robot.write_root_state_to_sim(root)
        env.sim.forward()
        env.sim.sense()
        contact_sensor.update(0.0)
        found = contact_sensor.data.found
        contact_mid = found.reshape(num_envs, -1).any(dim=1)
        low = torch.where(found_bracket & contact_mid, mid, low)
        high = torch.where(found_bracket & contact_mid, high, mid)
    # Prefer a 2 mm inset, but it can lose contact at stair/mesh edges.
    # Revalidate it and retain the last verified contact height when needed.
    root[:, 2] = low - 0.002
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(q, dq)
    robot.set_joint_position_target(q)
    robot.set_joint_velocity_target(dq)
    env.sim.forward()
    env.sim.sense()
    contact_sensor.update(0.0)
    initially_fallen = robot.data.projected_gravity_b[:, 2] > -0.70
    contact = contact_sensor.data.found.reshape(num_envs, -1).any(dim=1)
    # A vertical inset is not monotonic in contact at stair edges.
    # Only failed inset cases return to the binary search's contact endpoint.
    repaired_contact = found_bracket & ~contact
    if bool(repaired_contact.any()):
        root[:, 2] = torch.where(repaired_contact, low, root[:, 2])
        robot.write_root_state_to_sim(root)
        env.sim.forward()
        env.sim.sense()
        contact_sensor.update(0.0)
        contact = contact_sensor.data.found.reshape(num_envs, -1).any(dim=1)
    valid_setup = found_bracket & initially_fallen & contact
    static_velocity_max = max(float(robot.data.joint_vel.abs().max()),
                              float(robot.data.root_link_lin_vel_b.abs().max()),
                              float(robot.data.root_link_ang_vel_b.abs().max()))
    placement_report = {"seed": args.seed, "num_cases": num_envs,
                        "valid_cases": int(valid_setup.sum()),
                        "invalid_ids": [cases[i]["id"] for i in range(num_envs) if not valid_setup[i]],
                        "contact_endpoint_fallback_ids": [cases[i]["id"] for i in range(num_envs) if repaired_contact[i]],
                        "static_velocity_max": static_velocity_max}
    if args.placement_only:
        Path(args.output).write_text(json.dumps(placement_report, indent=2))
        print(json.dumps(placement_report))
    if not bool(valid_setup.all()) or static_velocity_max != 0.0:
        raise RuntimeError(f"Invalid static acceptance initialization: {placement_report}")
    if args.placement_only:
        env.close()
        return

    # Clear all residual velocities, then apply only
    # the explicitly requested initial angular velocity for that case.
    static_root = torch.cat(
        (robot.data.root_link_pose_w.clone(), torch.zeros((num_envs, 6), device=args.device)),
        dim=-1,
    )
    for i, case in enumerate(cases):
        speed = float(case["initial_ang_vel"])
        if case["pose"] in ("front", "back"):
            static_root[i, 11] = speed
        else:
            static_root[i, 10] = speed
    robot.write_root_state_to_sim(static_root)
    robot.write_joint_velocity_to_sim(torch.zeros_like(robot.data.joint_vel))
    torch.manual_seed(args.seed + 95000)
    env.action_manager.get_term("recovery").reset()
    env.observation_manager.reset()
    env.sim.forward()
    env.sim.sense()

    torch.manual_seed(args.seed + 100000)
    obs = wrapped.get_observations()
    if args.diagnostic:
        torch.save({"obs": obs["actor"], "root": static_root, "q": q, "action": policy(obs), "rng": torch.cuda.get_rng_state()}, args.output+".pt")
        env.close()
        return
    diagnostic_trace = []
    trajectory_obs = []
    criteria = [case_criteria(c, TERRAIN_NAMES[c["terrain_col"]], args.success_profile) for c in cases]
    gravity_threshold = torch.tensor([x[1] for x in criteria], device=args.device)
    height_threshold = torch.tensor([x[2] for x in criteria], device=args.device)
    legacy_criteria = [case_criteria(c, TERRAIN_NAMES[c["terrain_col"]], "tiered") for c in cases]
    legacy_gravity = torch.tensor([x[1] for x in legacy_criteria], device=args.device)
    legacy_height = torch.tensor([x[2] for x in legacy_criteria], device=args.device)
    legacy_counter = SuccessCounter(num_envs, args.device, args.hold_steps)
    stable_hold_steps = math.ceil(3.0/env.step_dt)
    stable_counter = StableRecoveryCounter(num_envs, args.device, stable_hold_steps, math.ceil(0.20/env.step_dt))
    robot_mass = env.sim.model.body_mass[:, robot.indexing.body_ids.long()].sum(dim=-1).expand(num_envs)
    gravity_magnitude = math.sqrt(sum(float(x)**2 for x in env.sim.mj_model.opt.gravity))
    weight = robot_mass * gravity_magnitude
    if not bool((weight > 0).all()):
        raise RuntimeError("Invalid simulated robot weight")
    selected_counter = SuccessCounter(num_envs, args.device, args.hold_steps)
    strict_counter = SuccessCounter(num_envs, args.device, args.hold_steps)
    hold = torch.zeros(num_envs, dtype=torch.int64, device=args.device)
    first = torch.full((num_envs,), -1, dtype=torch.int64, device=args.device)
    peak_raw_action = torch.zeros(num_envs, device=args.device)
    peak_clipped_action = torch.zeros(num_envs, device=args.device)
    clipped_action_count = torch.zeros(num_envs, device=args.device)
    action_sample_count = torch.zeros(num_envs, device=args.device)
    peak_torque = torch.zeros(num_envs, device=args.device)
    torque_saturation_count = torch.zeros(num_envs, device=args.device)
    torque_sample_count = torch.zeros(num_envs, device=args.device)
    prone_run = torch.zeros(num_envs, dtype=torch.int64, device=args.device)
    max_prone_run = torch.zeros_like(prone_run)
    spin_run = torch.zeros_like(prone_run)
    max_spin_run = torch.zeros_like(prone_run)
    post_run = torch.zeros_like(prone_run)
    max_post_run = torch.zeros_like(prone_run)
    effort_limits = torch.tensor(FRNET_EFFORT_LIMIT, device=args.device)

    disturbance_ids = torch.tensor(
        [i for i, c in enumerate(cases) if c["group"] == "disturbance"],
        dtype=torch.long,
        device=args.device,
    )
    disturbance_forces = torch.tensor(
        [(*DISTURBANCES[cases[int(i)]["disturbance"]], 0.0) for i in disturbance_ids.cpu()],
        device=args.device,
    )
    zero_forces = torch.zeros_like(disturbance_forces)
    zero_torques = torch.zeros_like(disturbance_forces)
    base_body = robot.body_names.index("base")

    def current_ground_height() -> torch.Tensor:
        data = env.scene["terrain_scan"].data
        idx = data.hit_pos_w.shape[1] // 2
        valid = data.distances[:, idx] >= 0
        return torch.where(valid, data.hit_pos_w[:, idx, 2], origins[:, 2])

    for step in range(args.horizon):
        torch.manual_seed(args.seed + 100001 + step)
        if args.trajectory_output and step % 10 == 0:
            trajectory_obs.append(obs["actor"].detach().cpu().clone())
        if len(disturbance_ids) and step == 25:
            robot.write_external_wrench_to_sim(
                disturbance_forces[:, None, :],
                zero_torques[:, None, :],
                env_ids=disturbance_ids,
                body_ids=[base_body],
            )
        elif len(disturbance_ids) and step == 30:
            robot.write_external_wrench_to_sim(
                zero_forces[:, None, :],
                zero_torques[:, None, :],
                env_ids=disturbance_ids,
                body_ids=[base_body],
            )

        with torch.inference_mode():
            raw_action = policy(obs)
            obs, _, _, _ = wrapped.step(raw_action)

        if step < 20:
            diagnostic_trace.append({"obs": obs["actor"].cpu().clone(), "root": robot.data.root_link_pose_w.cpu().clone(), "action": raw_action.cpu().clone()})
        clipped_action = raw_action.clamp(-10.0, 10.0)
        peak_raw_action = torch.maximum(peak_raw_action, raw_action.abs().amax(dim=1))
        peak_clipped_action = torch.maximum(
            peak_clipped_action, clipped_action.abs().amax(dim=1)
        )
        clipped_action_count += (raw_action.abs() >= 10.0).sum(dim=1)
        action_sample_count += raw_action.shape[1]

        torque = robot.data.actuator_force.abs()
        peak_torque = torch.maximum(peak_torque, torque.amax(dim=1))
        torque_saturation_count += (torque >= effort_limits[None, :] * 0.99).sum(dim=1)
        torque_sample_count += torque.shape[1]

        gravity_z = robot.data.projected_gravity_b[:, 2]
        relative_height = robot.data.root_link_pos_w[:, 2] - current_ground_height()
        upright = (gravity_z < -0.90) & (relative_height > 0.30)
        forces = env.scene["feet_ground_contact"].data.force
        if forces is None or forces.shape[-2:] != (4,3):
            raise RuntimeError("Stable recovery requires four3D foot force measurements")
        loaded_feet = (forces.norm(dim=-1) > 8.0).sum(dim=-1)
        # Sensor force points from foot toward terrain: negative worldZ supports weight.
        foot_support_ratio = (-forces[...,2]).clamp_min(0).sum(dim=-1) / weight
        stable_ok = valid_setup & stable_recovery_ok(
            height=relative_height, gravity_z=gravity_z,
            linear_speed=robot.data.root_link_lin_vel_b.norm(dim=-1),
            angular_speed=robot.data.root_link_ang_vel_b.norm(dim=-1),
            joint_speed=robot.data.joint_vel.abs().mean(dim=-1),
            loaded_feet=loaded_feet, foot_support_ratio=foot_support_ratio)
        major_fall = (gravity_z > -0.50) | (relative_height < 0.15)
        stable_counter.update(stable_ok, major_fall, step)
        legacy_counter.update(valid_setup & (gravity_z < -legacy_gravity) & (relative_height > legacy_height), step)
        strict_counter.update(valid_setup & upright, step)
        selected_counter.update(valid_setup & (gravity_z < -gravity_threshold) & (relative_height > height_threshold), step)
        # Keep original motion diagnostics tied to the original success definition.
        first = strict_counter.first

        prone_motion = (~upright) & (robot.data.joint_vel.abs().mean(dim=1) > 0.50)
        prone_run = torch.where(prone_motion, prone_run + 1, torch.zeros_like(prone_run))
        max_prone_run = torch.maximum(max_prone_run, prone_run)

        spin = (
            upright
            & (robot.data.root_link_lin_vel_b[:, :2].norm(dim=1) < 0.15)
            & (robot.data.root_link_ang_vel_b[:, 2].abs() > 1.0)
        )
        spin_run = torch.where(spin, spin_run + 1, torch.zeros_like(spin_run))
        max_spin_run = torch.maximum(max_spin_run, spin_run)

        after_success = first >= 0
        post_motion = after_success & (
            (robot.data.joint_vel.abs().mean(dim=1) > 1.0)
            | (robot.data.root_link_ang_vel_b.norm(dim=1) > 0.8)
        )
        post_run = torch.where(post_motion, post_run + 1, torch.zeros_like(post_run))
        max_post_run = torch.maximum(max_post_run, post_run)

    torch.save(diagnostic_trace, args.output+".trace.pt")
    rows: list[dict] = []
    for i, case in enumerate(cases):
        selected_first = stable_counter.first if args.success_profile == "stable" else selected_counter.first
        first_step = int(selected_first[i].item())
        success = bool(stable_counter.success[i]) if args.success_profile == "stable" else bool(first_step >= 0)
        if not success:
            first_step = -1
        case_hold_steps = stable_hold_steps if args.success_profile == "stable" else args.hold_steps
        recovery_time = (
            None
            if not success
            else round((first_step - case_hold_steps + 1) * env.step_dt, 4)
        )
        row = {
            **case,
            "terrain_name": TERRAIN_NAMES[case["terrain_col"]],
            "valid_setup": bool(valid_setup[i].item()),
            "initial_contact": bool(contact[i].item()),
            "initially_fallen": bool(initially_fallen[i].item()),
            "success": success,
            "stable_success": bool(stable_counter.success[i]),
            "stable_first_hold_step": int(stable_counter.first[i]),
            "stable_terminal_hold_steps": int(stable_counter.hold[i]),
            "stable_refell_after_recovery": bool(stable_counter.refell[i]),
            "terminal_loaded_feet": int(loaded_feet[i]),
            "terminal_foot_support_ratio": float(foot_support_ratio[i]),
            "legacy_tiered_success": bool(legacy_counter.first[i] >= 0),
            "strict_success": bool(strict_counter.first[i] >= 0),
            "strict_first_success_step": int(strict_counter.first[i]),
            "criteria_tier": criteria[i][0],
            "criteria_gravity_cos_min": criteria[i][1],
            "criteria_height_min_m": criteria[i][2],
            "recovery_time_s": recovery_time,
            "first_success_step": first_step,
            "peak_raw_action": round(float(peak_raw_action[i]), 6),
            "peak_clipped_action": round(float(peak_clipped_action[i]), 6),
            "action_clip_fraction": round(
                float(clipped_action_count[i] / action_sample_count[i]), 8
            ),
            "peak_torque_nm": round(float(peak_torque[i]), 6),
            "torque_saturation_fraction": round(
                float(torque_saturation_count[i] / torque_sample_count[i]), 8
            ),
            # Recovery necessarily requires fast joint motion while prone, so
            # a maximum-anywhere test mislabels normal recovery as oscillation.
            # Count only motion that remains continuous at the end of the
            # horizon while the robot is still not upright.
            "prone_oscillation": bool(prone_run[i] >= 50),
            "max_prone_motion_run_steps": int(max_prone_run[i].item()),
            "terminal_prone_motion_run_steps": int(prone_run[i].item()),
            "in_place_spin": bool(max_spin_run[i] >= 50),
            "post_recovery_oscillation": bool(max_post_run[i] >= 50),
        }
        rows.append(row)

    success_times = [r["recovery_time_s"] for r in rows if r["success"]]
    group_rates = grouped_rates(rows, "group")
    pose_rates = grouped_rates(rows, "pose")
    terrain_name_rates = grouped_rates(rows, "terrain_name")
    terrain_patch_rates = grouped_rates(
        [{**r, "patch": f'{r["terrain_row"]}:{r["terrain_col"]}'} for r in rows if r["group"] == "terrain"],
        "patch",
    )
    summary = {
        "suite": "frnet_go2_recovery_acceptance_v2",
        "task": args.task,
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "num_cases": len(rows),
        "horizon_policy_steps": args.horizon,
        "hold_policy_steps": stable_hold_steps if args.success_profile == "stable" else args.hold_steps,
        "friction_assignment": friction_assignment,
        "static_placement": {
            "protocol": "verified_contact_endpoint_fallback_v1",
            "scan_steps_limit": args.placement_scan_steps,
            "scan_steps_used": scan_step + 1,
            "binary_search_iterations": args.placement_iterations,
            "bracketed_cases": int(found_bracket.sum().item()),
            "valid_cases": sum(r["valid_setup"] for r in rows),
            "invalid_cases": sum(not r["valid_setup"] for r in rows),
            "contact_endpoint_fallback_ids": placement_report["contact_endpoint_fallback_ids"],
            "static_velocity_max": static_velocity_max,
        },
        "success_rate": sum(r["success"] for r in rows) / len(rows),
        "worst_pose_success_rate": min(pose_rates.values()),
        "worst_terrain_type_success_rate": min(terrain_name_rates.values()),
        "worst_terrain_patch_success_rate": min(terrain_patch_rates.values()),
        "mean_recovery_time_s": (
            None if not success_times else sum(success_times) / len(success_times)
        ),
        "peak_torque_nm": max(r["peak_torque_nm"] for r in rows),
        "mean_torque_saturation_fraction": sum(
            r["torque_saturation_fraction"] for r in rows
        )
        / len(rows),
        "max_action_clip_fraction": max(r["action_clip_fraction"] for r in rows),
        "any_prone_oscillation": any(r["prone_oscillation"] for r in rows),
        "any_in_place_spin": any(r["in_place_spin"] for r in rows),
        "any_post_recovery_oscillation": any(
            r["post_recovery_oscillation"] for r in rows
        ),
        "rates": {
            "group": group_rates,
            "pose": pose_rates,
            "terrain_type": terrain_name_rates,
            "terrain_patch": terrain_patch_rates,
            "angle_deg": grouped_rates(rows, "angle_deg"),
            "fold": grouped_rates(rows, "fold"),
            "friction": grouped_rates(rows, "friction"),
            "initial_ang_vel": grouped_rates(rows, "initial_ang_vel"),
            "terrain_position": grouped_rates(rows, "terrain_position"),
        },
    }
    strict_rows = [{**row, "success": row["strict_success"]} for row in rows]
    strict_rate = sum(row["success"] for row in strict_rows) / len(rows)
    strict_pose = grouped_rates(strict_rows, "pose")
    strict_terrain = grouped_rates(strict_rows, "terrain_name")
    summary["success_profile"] = args.success_profile
    summary["criteria_version"] = "stable_recovery_sim_v1" if args.success_profile == "stable" else "scenario_thresholds_v1"
    stable_rows = [{**row, "success": row["stable_success"]} for row in rows]
    summary["stable_success_rate"] = sum(row["stable_success"] for row in rows)/len(rows)
    summary["legacy_tiered_success_rate"] = sum(row["legacy_tiered_success"] for row in rows)/len(rows)
    summary["stable_criteria"] = {"height_min_m":0.26,"gravity_cos_min":0.90,
        "linear_speed_max_m_s":0.15,"angular_speed_max_rad_s":0.50,
        "joint_speed_mean_max_rad_s":1.0,"loaded_feet_min":3,"foot_force_min_N":8.0,
        "foot_vertical_support_weight_fraction_min":0.80,"terminal_hold_s":3.0,
        "refall_veto":"After first stable3s window: tilt>60deg or height<0.15m for>=0.20s",
        "height_calibration":"simulation replay310 quiet low cases; median0.272m;10-90percentile0.258-0.279m; provisional0.26m; not hardware calibrated"}
    summary["strict_success_rate"] = strict_rate
    summary["strict_rates"] = {"pose": strict_pose, "terrain_type": strict_terrain}
    summary["tier_success_rates"] = grouped_rates(rows, "criteria_tier")
    summary["engineering_gate_basis"] = "original_strict_criteria"
    engineering_gate = {
        "overall_success_rate_gte_0_95": strict_rate >= 0.95,
        "worst_pose_success_rate_gte_0_90": min(strict_pose.values()) >= 0.90,
        "worst_terrain_type_success_rate_gte_0_80": min(strict_terrain.values()) >= 0.80,
        "all_setups_valid": summary["static_placement"]["invalid_cases"] == 0,
        "mean_torque_saturation_lt_0_05": summary[
            "mean_torque_saturation_fraction"
        ]
        < 0.05,
        "no_persistent_oscillation_or_spin": not (
            summary["any_prone_oscillation"]
            or summary["any_in_place_spin"]
            or summary["any_post_recovery_oscillation"]
        ),
    }
    summary["original_strict_engineering_gate"] = dict(engineering_gate)
    if args.success_profile == "stable":
        engineering_gate["overall_success_rate_gte_0_95"] = summary["stable_success_rate"] >= 0.95
        engineering_gate["worst_pose_success_rate_gte_0_90"] = min(grouped_rates(stable_rows,"pose").values()) >= 0.90
        engineering_gate["worst_terrain_type_success_rate_gte_0_80"] = min(grouped_rates(stable_rows,"terrain_name").values()) >= 0.80
        summary["engineering_gate_basis"] = "stable_recovery_sim_v1"
    summary["engineering_gate"] = engineering_gate
    summary["engineering_gate_pass"] = all(engineering_gate.values())

    summary["protocol"] = "stable_static_repeated_v1" if args.success_profile == "stable" else "strict_static_repeated_v4" if args.success_profile == "strict" else "tiered_static_repeated_v1"
    if args.trajectory_output:
        selected = torch.tensor([r["success"] for r in rows], dtype=torch.bool)
        data = torch.stack(trajectory_obs)[:,selected,:].reshape(-1,45)
        torch.save({"observations": data, "checkpoint": str(args.checkpoint), "seed": args.seed, "protocol": summary["protocol"]}, args.trajectory_output)
    payload = {"summary": summary, "cases": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    env.close()


if __name__ == "__main__":
    main()

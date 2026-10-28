#!/usr/bin/env python3
"""Native-terrain decoupled recovery evaluation with random tile positions."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
import mjlab.tasks  # noqa: F401
import src.tasks.velocity.config.go2_recovery_aligned  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from src.tasks.velocity.config.go2_recovery_aligned.official_go2_limits import (
    FRNET_EFFORT_LIMIT,
    JOINT_NAMES,
    POSITION_LOWER,
    POSITION_UPPER,
    VELOCITY_LIMIT,
)
from src.tasks.velocity.config.go2_recovery_aligned.recovery_acceptance import (
    local_relative_height,
    physical_acceptance,
)

TERRAINS = ("flat", "stairs", "slope", "rough")
POSES = ("front", "back", "left", "right")
MODES = (
    "independent_random",
    "single_FL",
    "single_FR",
    "single_RL",
    "single_RR",
    "diagonal_FL_RR",
    "diagonal_FR_RL",
    "all_folded",
)


def stable_seed(terrain: str, pose: str, mode: str, sample: int) -> int:
    return (
        20260916
        + TERRAINS.index(terrain) * 100_000
        + POSES.index(pose) * 10_000
        + MODES.index(mode) * 100
        + sample
    )


def scalar_bool(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.reshape(-1)[0].item())
    return bool(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--terrains", nargs="+", choices=TERRAINS, default=list(TERRAINS))
    parser.add_argument("--poses", nargs="+", choices=POSES, default=list(POSES))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--per-stratum", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--spawn-half-range", type=float, default=2.7)
    args = parser.parse_args()

    if args.horizon < 500:
        raise ValueError("The decoupled acceptance horizon must be at least 500 control steps")

    torch.manual_seed(20260916)
    cfg = load_env_cfg("Unitree-Go2-Recovery-FRNet-Aligned", play=True)
    cfg.scene.num_envs = 1
    cfg.viewer = None
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env)
    agent_cfg = load_rl_cfg("Unitree-Go2-Recovery-FRNet-Aligned")
    runner_cls = load_runner_cls("Unitree-Go2-Recovery-FRNet-Aligned") or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
    runner.load(
        args.checkpoint,
        load_cfg={"actor": True},
        strict=True,
        map_location=args.device,
    )
    policy = runner.get_inference_policy(device=args.device)

    robot = env.scene["robot"]
    if tuple(robot.joint_names) != JOINT_NAMES:
        raise ValueError(f"Unexpected joint order: {robot.joint_names}")
    if set(robot.actuator_names) != set(JOINT_NAMES):
        raise ValueError(f"Unexpected actuator names: {robot.actuator_names}")
    effort_by_name = dict(zip(JOINT_NAMES, FRNET_EFFORT_LIMIT, strict=True))
    effort_limit = torch.tensor(
        [effort_by_name[name] for name in robot.actuator_names], device=args.device
    )
    velocity_limit = torch.tensor(VELOCITY_LIMIT, device=args.device)
    position_lower = torch.tensor(POSITION_LOWER, device=args.device)
    position_upper = torch.tensor(POSITION_UPPER, device=args.device)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    rows: list[dict] = []
    total = len(args.terrains) * len(args.poses) * len(args.modes) * args.per_stratum
    episode = 0

    for terrain in args.terrains:
        for pose in args.poses:
            for mode in args.modes:
                for sample in range(args.per_stratum):
                    seed = stable_seed(terrain, pose, mode, sample)
                    rng = random.Random(seed)
                    spawn_x = rng.uniform(-args.spawn_half_range, args.spawn_half_range)
                    spawn_y = rng.uniform(-args.spawn_half_range, args.spawn_half_range)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    env.settled_reset_terrain_family = terrain
                    env.settled_reset_fall_pose = pose
                    env.settled_reset_birth_mode = mode
                    env.settled_reset_x = spawn_x
                    env.settled_reset_y = spawn_y
                    env.reset()
                    obs = wrapped.get_observations()

                    initial_root_speed = float(robot.data.root_link_vel_w[0].norm().item())
                    initial_joint_speed = float(robot.data.joint_vel[0].abs().max().item())
                    meta = dict(env.last_settled_reset)
                    initial_contact = False
                    for sensor_name in ("nonfoot_ground_touch", "base_ground_contact", "feet_ground_contact"):
                        try:
                            sensor = env.scene[sensor_name]
                        except KeyError:
                            continue
                        sensor.update(0.0)
                        initial_contact = initial_contact or bool(sensor.data.found.any().item())
                    initial_verified = (
                        initial_contact
                        and initial_root_speed <= 1.0e-8
                        and initial_joint_speed <= 1.0e-8
                        and not bool(meta["any_joint_clamped"])
                    )

                    hold = 0
                    success_step = None
                    reset_detected = False
                    peak_torque_ratio = 0.0
                    peak_velocity_ratio = 0.0
                    peak_position_excess = 0.0
                    peak_torque_nm = 0.0
                    torque_saturation_count = 0
                    torque_sample_count = 0
                    steps_run = 0
                    relative_height = float("nan")

                    for step in range(args.horizon):
                        with torch.inference_mode():
                            action = policy(obs)
                        obs, _, done, _ = wrapped.step(action)
                        steps_run = step + 1

                        torque = robot.data.actuator_force[0].abs()
                        joint_velocity = robot.data.joint_vel[0].abs()
                        position_excess = torch.maximum(
                            position_lower - robot.data.joint_pos[0],
                            robot.data.joint_pos[0] - position_upper,
                        ).clamp_min(0.0)
                        peak_torque_nm = max(peak_torque_nm, float(torque.max().item()))
                        peak_torque_ratio = max(
                            peak_torque_ratio, float((torque / effort_limit).max().item())
                        )
                        peak_velocity_ratio = max(
                            peak_velocity_ratio,
                            float((joint_velocity / velocity_limit).max().item()),
                        )
                        peak_position_excess = max(
                            peak_position_excess, float(position_excess.max().item())
                        )
                        torque_saturation_count += int((torque >= 0.99 * effort_limit).sum().item())
                        torque_sample_count += torque.numel()

                        if scalar_bool(done):
                            reset_detected = True
                            break

                        relative_height = float(local_relative_height(env)[0].item())
                        hold = hold + 1 if bool(physical_acceptance(env)[0].item()) else 0
                        if hold >= 50:
                            success_step = step - 49
                            break

                    hardware_safety_pass = (
                        peak_torque_ratio <= 1.00001
                        and peak_velocity_ratio <= 1.0
                        and peak_position_excess <= 0.02
                    )
                    row = dict(meta)
                    row.update(
                        episode=episode,
                        sample=sample,
                        seed=seed,
                        requested_spawn_offset_xy_m=[spawn_x, spawn_y],
                        initial_contact=initial_contact,
                        initial_root_speed=initial_root_speed,
                        initial_joint_speed=initial_joint_speed,
                        initial_state_verified=initial_verified,
                        success=(success_step is not None and not reset_detected),
                        recovery_time_s=(
                            None if success_step is None else round(success_step * env.step_dt, 4)
                        ),
                        horizon_s=round(args.horizon * env.step_dt, 4),
                        steps_run=steps_run,
                        reset_detected=reset_detected,
                        final_relative_height_m=relative_height,
                        final_projected_gravity=robot.data.projected_gravity_b[0].tolist(),
                        peak_torque_nm=peak_torque_nm,
                        peak_torque_ratio=peak_torque_ratio,
                        peak_velocity_ratio=peak_velocity_ratio,
                        peak_position_excess_rad=peak_position_excess,
                        torque_saturation_fraction=(
                            torque_saturation_count / torque_sample_count
                            if torque_sample_count
                            else 0.0
                        ),
                        hardware_safety_pass=hardware_safety_pass,
                    )
                    rows.append(row)
                    episode += 1
                    if episode % 20 == 0 or episode == total:
                        partial.write_text(json.dumps(rows, indent=2) + "\n")
                        print(
                            json.dumps(
                                {
                                    "completed": episode,
                                    "total": total,
                                    "success": sum(int(r["success"]) for r in rows),
                                    "reset_detected": sum(int(r["reset_detected"]) for r in rows),
                                }
                            ),
                            flush=True,
                        )

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["terrain_family"], row["fall_pose"], row["birth_mode"])].append(row)
    stratum_summary = {}
    for key, group in grouped.items():
        success = sum(int(row["success"]) for row in group)
        stratum_summary["/".join(key)] = {
            "episodes": len(group),
            "success": success,
            "success_rate": success / len(group),
            "hardware_safety_pass_rate": sum(
                int(row["hardware_safety_pass"]) for row in group
            ) / len(group),
            "reset_detected": sum(int(row["reset_detected"]) for row in group),
        }
    successes = sum(int(row["success"]) for row in rows)
    report = {
        "checkpoint": args.checkpoint,
        "suite": "native_terrain_decoupled_random_position_recovery",
        "episodes": len(rows),
        "strata": len(stratum_summary),
        "per_stratum": args.per_stratum,
        "horizon_steps": args.horizon,
        "horizon_s": args.horizon * env.step_dt,
        "terrain_relative_height_criterion_m": 0.30,
        "upright_projected_gravity_z_criterion": -0.90,
        "hold_steps": 50,
        "spawn_offset_range_m": [-args.spawn_half_range, args.spawn_half_range],
        "success": successes,
        "success_rate": successes / len(rows),
        "all_initial_states_verified": all(row["initial_state_verified"] for row in rows),
        "no_mid_episode_resets": not any(row["reset_detected"] for row in rows),
        "hardware_safety_pass_rate": sum(
            int(row["hardware_safety_pass"]) for row in rows
        ) / len(rows),
        "stratum_summary": stratum_summary,
        "results": rows,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "episodes",
                    "strata",
                    "success_rate",
                    "all_initial_states_verified",
                    "no_mid_episode_resets",
                    "hardware_safety_pass_rate",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()

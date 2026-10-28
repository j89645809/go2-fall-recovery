"""Deterministic settled-pose recovery and Go2 hardware-boundary evaluation."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import mjlab.tasks
import src.tasks.velocity.config.go2_recovery_aligned
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from src.tasks.velocity.config.go2_recovery_aligned.official_go2_limits import (
    POSITION_LOWER,
    POSITION_UPPER,
    VELOCITY_LIMIT,
)


TERRAINS = {
    "flat": (0, 2),
    "stairs": (3, 5),
    "slope": (6, 5),
    "rough": (7, 8),
}
POSES = ("front", "back", "left", "right")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pose", choices=POSES, required=True)
    parser.add_argument("--horizon", type=int, default=1000)
    args = parser.parse_args()

    torch.manual_seed(20260912)
    cfg = load_env_cfg("Unitree-Go2-Recovery-FRNet-Aligned", play=True)
    cfg.scene.num_envs = 1
    cfg.viewer = None
    env = ManagerBasedRlEnv(cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env)
    agent_cfg = load_rl_cfg("Unitree-Go2-Recovery-FRNet-Aligned")
    runner_cls = (
        load_runner_cls("Unitree-Go2-Recovery-FRNet-Aligned")
        or MjlabOnPolicyRunner
    )
    runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
    runner.load(
        args.checkpoint,
        load_cfg={"actor": True},
        strict=True,
        map_location=args.device,
    )
    policy = runner.get_inference_policy(device=args.device)
    robot = env.scene["robot"]

    joint_names = tuple(robot.joint_names)
    position_lower = torch.tensor(POSITION_LOWER, device=args.device)
    position_upper = torch.tensor(POSITION_UPPER, device=args.device)
    velocity_limits = torch.tensor(VELOCITY_LIMIT, device=args.device)
    assert joint_names == tuple(
        f"{leg}_{joint}_joint"
        for leg in ("FL", "FR", "RL", "RR")
        for joint in ("hip", "thigh", "calf")
    )
    actuator_names = tuple(
        name for actuator in robot.actuators for name in actuator.target_names
    )
    torque_limits = torch.tensor(
        [35.55 if "calf" in name else 23.7 for name in actuator_names],
        device=args.device,
    )

    results = []
    for family, (column, row) in TERRAINS.items():
        env.settled_reset_fall_pose = args.pose
        env.settled_reset_terrain_family = family
        env.settled_reset_terrain_column = column
        env.settled_reset_terrain_row = row
        observation, _ = env.reset()
        observation = wrapped.get_observations()

        initial_root_speed = float(robot.data.root_link_vel_w.abs().max())
        initial_joint_speed = float(robot.data.joint_vel.abs().max())
        initial_gravity_z = float(robot.data.projected_gravity_b[0, 2])
        initial_nonfoot_contact = bool(
            env.scene["nonfoot_ground_touch"].data.found.any()
        )
        hold = 0
        recovery_step = -1
        peak_torque_ratio = 0.0
        peak_velocity_ratio = 0.0
        peak_position_excess = 0.0
        done_count = 0
        saturation_samples = 0
        total_samples = 0

        for step in range(args.horizon):
            with torch.inference_mode():
                action = policy(observation)
            # Environment state tensors must remain normal tensors because the
            # next reset updates sensor histories in place.
            observation, _, done, _ = wrapped.step(action)
            done_count += int(done.any())
            upright = bool(
                (robot.data.projected_gravity_b[0, 2] < -0.90)
                and (robot.data.root_link_pos_w[0, 2] > 0.30)
            )
            hold = hold + 1 if upright else 0
            if recovery_step < 0 and hold >= 50:
                recovery_step = step - 49

            torque = robot.data.actuator_force[0].abs()
            velocity = robot.data.joint_vel[0].abs()
            position_excess = torch.maximum(
                position_lower - robot.data.joint_pos[0],
                robot.data.joint_pos[0] - position_upper,
            ).clamp_min(0)
            peak_torque_ratio = max(
                peak_torque_ratio, float((torque / torque_limits).max())
            )
            peak_velocity_ratio = max(
                peak_velocity_ratio, float((velocity / velocity_limits).max())
            )
            peak_position_excess = max(
                peak_position_excess, float(position_excess.max())
            )
            saturation_samples += int((torque >= torque_limits * 0.99).sum())
            total_samples += torque.numel()

        safety_pass = (
            peak_torque_ratio <= 1.00001
            and peak_velocity_ratio <= 1.0
            and peak_position_excess <= 0.02
            and done_count == 0
        )
        results.append(
            {
                "pose": args.pose,
                "terrain": family,
                "terrain_column": column,
                "terrain_row": row,
                "settled_attempts": env.last_settled_reset["attempts"],
                "initial_root_speed": initial_root_speed,
                "initial_joint_speed": initial_joint_speed,
                "initial_gravity_z": initial_gravity_z,
                "initial_nonfoot_contact": initial_nonfoot_contact,
                "success": recovery_step >= 0,
                "recovery_time_s": (
                    None if recovery_step < 0 else round(recovery_step * env.step_dt, 4)
                ),
                "peak_torque_ratio": round(peak_torque_ratio, 7),
                "peak_velocity_ratio": round(peak_velocity_ratio, 7),
                "peak_position_excess_rad": round(peak_position_excess, 7),
                "torque_saturation_fraction": round(
                    saturation_samples / total_samples, 7
                ),
                "invalid_termination_count": done_count,
                "hardware_safety_pass": safety_pass,
            }
        )

    output = {
        "checkpoint": args.checkpoint,
        "suite": "settled_pose_four_terrain_hardware_safety",
        "actuator_order": actuator_names,
        "torque_limits_nm": torque_limits.tolist(),
        "results": results,
        "success_rate": sum(row["success"] for row in results) / len(results),
        "hardware_safety_pass": all(row["hardware_safety_pass"] for row in results),
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()

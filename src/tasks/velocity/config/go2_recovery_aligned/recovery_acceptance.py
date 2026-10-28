"""Shared recovery acceptance helpers.

Only the simulator-side height helper uses privileged terrain geometry.  The
controller readiness predicate intentionally uses signals available from the
Go2 itself: IMU, joint encoders, estimated base velocity, and foot contacts.
"""

from __future__ import annotations

import torch


STAND_JOINT_TARGET = (
     0.08, 0.74, -1.42,
    -0.03, 0.79, -1.44,
    -0.02, 1.00, -1.45,
     0.02, 0.98, -1.45,
)


def foot_contact_mask(env, engage_force: float = 8.0) -> torch.Tensor:
    """Return one deployable contact bit per foot from onboard force estimates."""
    force = env.scene["feet_ground_contact"].data.force
    if force is None or force.shape[-2:] != (4, 3):
        raise ValueError("Recovery acceptance requires four foot-force estimates")
    # ContactSensor force points from the foot toward the terrain in this
    # model, so its world Z component is negative on a supporting foot.
    # Magnitude is invariant to that convention and to sloped terrain.
    return force.norm(dim=-1) > engage_force


def local_relative_height(env) -> torch.Tensor:
    """Robust simulator-only base height above the local central ground patch."""
    heights = env.scene["terrain_scan"].data.hit_pos_w[..., 2]
    if heights.shape[-1] != 187:
        raise ValueError(f"Expected 187 terrain rays, got {tuple(heights.shape)}")
    grid = heights.reshape(heights.shape[0], 17, 11)
    central = grid[:, 7:10, 4:7].reshape(heights.shape[0], -1)
    fallback = env.scene.terrain.env_origins[:, 2].unsqueeze(-1)
    central = torch.where(torch.isfinite(central), central, fallback)
    ground = central.median(dim=-1).values
    return env.scene["robot"].data.root_link_pos_w[:, 2] - ground


def onboard_stand_ready(env, *, contact_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Stand/hold gate composed only of signals available on the robot."""
    robot = env.scene["robot"]
    contact = foot_contact_mask(env) if contact_mask is None else contact_mask
    target = robot.data.joint_pos.new_tensor(STAND_JOINT_TARGET)
    joint_ready = (robot.data.joint_pos - target).abs().amax(dim=-1) < 0.55
    upright = robot.data.projected_gravity_b[:, 2] < -0.90
    low_angular_rate = robot.data.root_link_ang_vel_b.norm(dim=-1) < 0.35
    low_linear_rate = robot.data.root_link_lin_vel_b.norm(dim=-1) < 0.30
    return upright & low_angular_rate & low_linear_rate & contact.all(dim=-1) & joint_ready


def physical_acceptance(env) -> torch.Tensor:
    """Unified simulator acceptance: onboard readiness plus physical height."""
    return onboard_stand_ready(env) & (local_relative_height(env) > 0.30)

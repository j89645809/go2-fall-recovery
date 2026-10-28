"""Acceptance-aligned reward shaping for difficult recovery states."""

from __future__ import annotations

import torch

from .recovery_acceptance import (
    STAND_JOINT_TARGET, foot_contact_mask, local_relative_height,
    physical_acceptance,
)


def _height_gate(env) -> torch.Tensor:
    # Strong gradient in the 0.25--0.35 m band where most failures stop.
    return torch.sigmoid((local_relative_height(env) - 0.30) / 0.025)


def recovery_height_crossing(env) -> torch.Tensor:
    return _height_gate(env)


def recovery_supported_height(env) -> torch.Tensor:
    support = foot_contact_mask(env).float().mean(dim=-1)
    return support * _height_gate(env)


def recovery_stable_stand(env) -> torch.Tensor:
    robot = env.scene["robot"]
    upright = torch.sigmoid((-robot.data.projected_gravity_b[:, 2] - 0.90) / 0.025)
    angular = torch.exp(-robot.data.root_link_ang_vel_b.square().sum(dim=-1) / 0.20)
    linear = torch.exp(-robot.data.root_link_lin_vel_b.square().sum(dim=-1) / 0.20)
    support = foot_contact_mask(env).float().mean(dim=-1)
    target = robot.data.joint_pos.new_tensor(STAND_JOINT_TARGET)
    joint_error = (robot.data.joint_pos - target).square().mean(dim=-1)
    joint_ready = torch.exp(-joint_error / 0.35)
    return upright * angular * linear * support * _height_gate(env) * joint_ready


class RecoverySuccessBonus:
    """Bonus after 50 consecutive acceptance-aligned stable control steps."""

    def __init__(self, cfg, env):
        del cfg
        self.counter = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.counter.zero_()
        else:
            self.counter[env_ids] = 0

    def __call__(self, env):
        stable = physical_acceptance(env)
        self.counter = torch.where(stable, self.counter + 1, torch.zeros_like(self.counter))
        return (self.counter >= 50).float()


def residual_stand_motion_penalty(env):
    """Suppress angular motion and action chatter only near upright standing."""
    robot=env.scene["robot"]
    gate=((-robot.data.projected_gravity_b[:,2] > 0.9) & (local_relative_height(env) > 0.30)).float()
    action=env.action_manager.get_term("recovery")
    return gate*(robot.data.root_link_ang_vel_b.square().sum(-1)+(action.current-action.previous).square().mean(-1))

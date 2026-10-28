"""FR-Net control/reset semantics, independent of either physics engine.

Verified against GO2RecoveryEnv, GO2 and LeggedRobot on 2026-09-10.
These functions deliberately do not expand the source reset distribution.
"""
import torch
from .official_go2_limits import clamp_joint_positions

JOINT_NAMES = tuple(f'{leg}_{joint}_joint' for leg in ('FL','FR','RL','RR')
                    for joint in ('hip','thigh','calf'))
PHYSICS_DT = .005
DECIMATION = 4
ACTION_SCALE = .25
KP = 30.
KD = .8

def sample_joint_reset(n, *, device, generator=None, enforce_official_limits=False):
    q = torch.empty((n, 12), device=device)
    q[:, 0::3] = 0.70 * torch.rand((n, 4), device=device, generator=generator) - 0.35
    q[:, 1::3] = 0.90 + 1.70 * torch.rand((n, 4), device=device, generator=generator)
    q[:, 2::3] = -2.90 + 1.80 * torch.rand((n, 4), device=device, generator=generator)
    modes = torch.randint(0, 10, (n,), device=device, generator=generator)
    masks = torch.zeros((n, 4), dtype=torch.bool, device=device)
    for leg in range(4):
        masks[:, leg] |= modes == leg
    masks[modes == 4, 0] = True; masks[modes == 4, 3] = True
    masks[modes == 5, 1] = True; masks[modes == 5, 2] = True
    masks[modes == 6, :] = True
    for leg in range(4):
        ids = masks[:, leg]
        side = 1.0 if leg in (0, 2) else -1.0
        q[ids, 3 * leg : 3 * leg + 3] = q.new_tensor([0.82 * side, 2.9 if leg < 2 else 3.5, -2.68])
    # The Isaac source does not clamp, and can request calf angles below the
    # physical stop.  Runtime alignment keeps the distribution but projects
    # those invalid samples into Unitree's official Go2 limits.
    if enforce_official_limits:
        q = clamp_joint_positions(q)
    return q, torch.zeros((n, 12), device=device)

def sample_root_reset(origins, *, generator=None):
    n = len(origins)
    root = torch.zeros((n, 13), device=origins.device, dtype=origins.dtype)
    root[:, :3] = origins + origins.new_tensor([0., 0., .6])
    root[:, :2] += 6 * torch.rand((n, 2), device=origins.device, generator=generator) - 3
    quat = torch.randn((n, 4), device=origins.device, generator=generator)
    # Uniform unit quaternion: wxyz/xyzw permutation has the same distribution.
    root[:, 3:7] = quat / quat.norm(dim=-1, keepdim=True)
    root[:, 7:] = torch.rand((n, 6), device=origins.device, generator=generator) - .5
    return root

def actor_noise_std(like, noise_level=1.):
    std = like.new_zeros(45)
    std[:3] = .2 * .25
    std[3:6] = .05 * noise_level
    std[9:21] = .01
    std[21:33] = 1.5 * .05
    # Preserve GO2's double noise_level multiplication on gravity.
    return std * noise_level

def pd_torques(delayed_actions, q, dq, reference, limits, kp_factor, kd_factor):
    torque = KP * kp_factor * (ACTION_SCALE * delayed_actions + reference - q) - KD * kd_factor * dq
    return torch.maximum(torch.minimum(torque, limits), -limits)

class ActionDelay:
    """Per-env 0..5 policy-step delay, after action clipping, before PD."""
    def __init__(self, n, device):
        self.buffer = torch.zeros((6, n, 12), device=device)
        self.delays = torch.zeros(n, dtype=torch.long, device=device)

    def reset(self, ids, generator=None):
        self.buffer[:, ids] = 0
        self.delays[ids] = torch.randint(0, 6, (len(ids),), device=self.buffer.device, generator=generator)

    def push(self, actions):
        self.buffer = self.buffer.roll(1, dims=0)
        self.buffer[0] = actions.clamp(-10, 10)
        return self.buffer[self.delays, torch.arange(len(actions), device=actions.device)]

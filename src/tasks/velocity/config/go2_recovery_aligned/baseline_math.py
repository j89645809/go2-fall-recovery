"""FR-Net recovery formulas. Joint tensors use FL, FR, RL, RR leg order.

Pure tensor functions keep engine adapters separate from baseline semantics.
Source: legged_gym/envs/go2/go2_recovery/go2_recovery_env.py.
"""
import torch

SCALES = {
    'orientation': -0.5, 'orientation_pos_gaussian': 6.0,
    'base_height_pos': 1.0, 'foot_contact_pos': 0.1,
    'step2_target_pos': 6.0, 'ang_vel_xyz': -0.05,
    'torques': -0.0002, 'dof_acc': -1e-6, 'dof_vel': -0.006,
    'dof_pos_limits': -10.0, 'collision': -0.2, 'action': -0.01,
    'action_rate': -0.02, 'action_smoothness_2': -0.05,
    'max_velocity': -0.5, 'feet_stumble': -1.0,
}

def orientation(gravity):
    return (gravity - gravity.new_tensor([0., 0., -1.])).square().sum(-1)

def orientation_pos_gaussian(gravity, epsilon=0.15):
    return torch.exp(-(gravity[..., 2] + 1.).square() / (2 * epsilon**2))

def base_height_pos(root_z, measured_heights, target=0.45):
    height = (root_z.unsqueeze(-1) - measured_heights).mean(-1)
    return torch.exp(-(height-target).square())

def foot_contact_pos(filtered_contact):
    return filtered_contact.float().sum(-1)

def step2_target_pos(joint_pos, gravity, iteration, epsilon=0.2):
    low = joint_pos.new_tensor([0., 1.5, -2.4] * 4)
    high = joint_pos.new_tensor([-.05,.8,-1.5,.05,.8,-1.5,
                                 -.05,1.,-1.5,.05,1.,-1.5])
    progress = min(iteration / 1500., 1.)
    target = low + (high-low)*progress
    eligible = (gravity[...,2] >= -1.-epsilon) & (gravity[...,2] <= -1.+epsilon)
    return torch.where(eligible, torch.exp(-(joint_pos-target).square().sum(-1)),
                       torch.zeros_like(gravity[...,2]))

def action_smoothness_2(actions, last_actions, last_last_actions):
    mask = (last_actions != 0) & (last_last_actions != 0)
    return ((actions-2*last_actions+last_last_actions).square()*mask).sum(-1)

def max_velocity(joint_vel, threshold=0.8):
    return (joint_vel.abs()-threshold).clamp_min(0).sum(-1)

def squared_norm(value):
    return value.square().sum(-1)

def dof_acc(last_velocity, velocity, dt):
    return squared_norm((last_velocity-velocity)/dt)

def dof_pos_limits(position, soft_limits):
    return ((soft_limits[...,0]-position).clamp_min(0)
            +(position-soft_limits[...,1]).clamp_min(0)).sum(-1)

def collision(selected_body_forces):
    return (selected_body_forces.norm(dim=-1)>.1).float().sum(-1)

def feet_stumble(foot_forces):
    return (foot_forces[...,:2].norm(dim=-1)>4*foot_forces[...,2].abs()).any(-1)

def total_reward(terms, dt):
    """Original scales multiply control dt, then total is clipped positive."""
    if set(terms) != set(SCALES):
        raise ValueError('All and only the 16 baseline reward terms are required')
    return sum(terms[name]*scale*dt for name, scale in SCALES.items()).clamp_min(0)

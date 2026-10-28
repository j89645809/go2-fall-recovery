import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

def orientation_pos_gaussian(env, epsilon=0.15, asset_cfg=SceneEntityCfg("robot")):
    g = env.scene[asset_cfg.name].data.projected_gravity_b
    target = torch.tensor([0., 0., -1.], device=g.device, dtype=g.dtype)
    return torch.exp(-torch.sum((g - target) ** 2, dim=1) / (epsilon * epsilon))

def base_height_pos(env, target=0.45, asset_cfg=SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2]
    # FR-Net measures height relative to the terrain height samples.
    heights = env.scene["terrain_scan"].data.hit_pos_w[..., 2]
    measured = (z.unsqueeze(-1) - heights).mean(-1)
    return torch.exp(-torch.square(measured - target))

def foot_contact_pos(env, sensor_name="feet_ground_contact"):
    found = env.scene[sensor_name].data.found
    return found.squeeze(-1).float().sum(dim=1)

def step2_target_pos(env, asset_cfg=SceneEntityCfg("robot")):
    asset: Entity = env.scene[asset_cfg.name]
    # Stand-pose proxy: reward joints near their configured default pose.
    return torch.exp(-torch.mean(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1))

def recovery_success_shaping(env, asset_cfg=SceneEntityCfg("robot"), sensor_name="feet_ground_contact"):
    """Dense proxy for the actual recovery objective: upright, elevated, stable and supported."""
    asset = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b
    upright = torch.exp(-torch.sum((g - torch.tensor([0., 0., -1.], device=g.device, dtype=g.dtype)) ** 2, dim=1) / 0.20**2)
    z = asset.data.root_link_pos_w[:, 2]
    height = torch.sigmoid((z - 0.38) / 0.035)
    contacts = env.scene[sensor_name].data.found.squeeze(-1).float().sum(dim=1)
    support = torch.clamp(contacts / 4.0, 0.0, 1.0)
    ang = torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=1)
    stable = torch.exp(-torch.square(ang) / 0.8**2)
    return upright * height * support * stable

def recovery_success_bonus(env, asset_cfg=SceneEntityCfg("robot"), sensor_name="feet_ground_contact"):
    """Small terminal-style bonus, gated so lying still cannot score it."""
    asset = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b
    upright = g[:, 2] < -0.92
    z = asset.data.root_link_pos_w[:, 2] > 0.38
    contacts = env.scene[sensor_name].data.found.squeeze(-1).float().sum(dim=1)
    ang = torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=1)
    return (upright & z & (contacts >= 3) & (ang < 1.0)).float()

def recovery_success_rate(env, asset_cfg=SceneEntityCfg("robot"), sensor_name="feet_ground_contact"):
    """Per-environment binary success; rollout mean is the recovery success rate."""
    return recovery_success_bonus(env, asset_cfg=asset_cfg, sensor_name=sensor_name)

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

def orientation_pos_gaussian(env, epsilon=0.15, asset_cfg=SceneEntityCfg("robot")):
    g = env.scene[asset_cfg.name].data.projected_gravity_b
    target = torch.tensor([0., 0., -1.], device=g.device, dtype=g.dtype)
    return torch.exp(-torch.sum((g - target) ** 2, dim=1) / (epsilon * epsilon))

def base_height_pos(env, target=0.45, asset_cfg=SceneEntityCfg("robot")):
    z = env.scene[asset_cfg.name].data.root_link_pos_w[:, 2]
    return torch.exp(-torch.square(z - target))

def foot_contact_pos(env, sensor_name="feet_ground_contact"):
    found = env.scene[sensor_name].data.found
    return found.squeeze(-1).float().sum(dim=1)

def step2_target_pos(env, asset_cfg=SceneEntityCfg("robot")):
    asset: Entity = env.scene[asset_cfg.name]
    # Stand-pose proxy: reward joints near their configured default pose.
    return torch.exp(-torch.mean(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1))

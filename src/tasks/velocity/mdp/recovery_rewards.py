import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

def upright_gaussian(env, asset_cfg=SceneEntityCfg("robot"), epsilon=0.15):
    asset: Entity = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b
    target = torch.tensor([0., 0., -1.], device=g.device, dtype=g.dtype)
    return torch.exp(-torch.sum((g - target) ** 2, dim=1) / (epsilon * epsilon))

def base_height_gaussian(env, target=0.45, std=0.12, asset_cfg=SceneEntityCfg("robot")):
    asset: Entity = env.scene[asset_cfg.name]
    return torch.exp(-torch.square(asset.data.root_link_pos_w[:, 2] - target) / (std * std))

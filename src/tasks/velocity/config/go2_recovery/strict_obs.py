import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

def zeros(env, width):
    return torch.zeros((env.num_envs, width), device=env.device)

def push_force(env): return zeros(env, 2)
def push_torque(env): return zeros(env, 3)
def friction(env): return zeros(env, 1)
def body_mass(env): return zeros(env, 1)
def limb_mass(env): return zeros(env, 18)

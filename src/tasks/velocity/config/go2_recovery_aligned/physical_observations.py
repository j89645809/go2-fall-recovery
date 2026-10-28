"""Read physical parameters by named robot bodies, never by padding dimensions.

Callers must supply a verified body mapping. This module deliberately rejects
the existing 15-body custom robot when an 18-limb baseline layout is requested.
"""
import torch

def body_masses(env, names):
    robot = env.scene['robot']
    if len(set(names)) != len(names):
        raise ValueError('Body mapping must not duplicate masses')
    missing = set(names) - set(robot.body_names)
    if missing:
        raise ValueError(f'Missing physical bodies: {sorted(missing)}')
    local = [robot.body_names.index(name) for name in names]
    ids = robot.indexing.body_ids[local].long()
    value = env.sim.model.body_mass
    if value.ndim != 2 or value.shape[0] not in (1, env.num_envs):
        raise ValueError(f'Unexpected batched mass shape: {value.shape}')
    return value[:, ids].expand(env.num_envs, -1)

def privileged_masses(env, base_name, limb_names):
    if len(limb_names) != 18 or base_name in limb_names:
        raise ValueError('Baseline requires one base plus 18 distinct mapped limbs')
    masses = body_masses(env, (base_name, *limb_names))
    return masses[:, :1] / 15., masses[:, 1:] / 5.

def shared_robot_friction(env, collision_geom_ids):
    """Return the actual shared sliding coefficient, fail if no scalar exists.

    IDs are global simulation geometry IDs resolved from the robot collisions,
    not terrain or visual geometry. Randomization must assign one value per env.
    """
    if len(collision_geom_ids) == 0:
        raise ValueError('At least one robot collision geom must be selected')
    value = env.sim.model.geom_friction[:, collision_geom_ids, 0]
    if not torch.allclose(value, value[:, :1].expand_as(value)):
        raise ValueError('Robot collision frictions differ; cannot encode as one scalar')
    return value[:, :1].expand(env.num_envs, 1)

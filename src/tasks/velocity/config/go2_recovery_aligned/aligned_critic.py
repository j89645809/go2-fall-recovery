"""Critic assembly with required live parameters; no missing-field defaults.

Terrain heights must come from the source-style heightfield sampler, not robot
ray hits. Wrench labels must be supplied by the selected perturbation protocol.
"""
from .baseline_observations import assemble
from .physical_observations import body_masses, shared_robot_friction
from .aligned_asset import BODY_NAMES
from .aligned_wrench import applied_base_wrench


def measured_terrain_heights(
    env, sensor_name='terrain_scan', allow_nonfinite=False
):
    """Absolute terrain heights at the source 17x11 sample locations.

    The mjlab ray sensor is configured with the same 1.6m x 1.0m, 0.1m grid
    and yaw alignment.  Reading hit z coordinates keeps this channel physical
    and eliminates the former zero/synthetic placeholder.
    """
    sensor=env.scene[sensor_name]
    heights=sensor.data.hit_pos_w[...,2]
    if heights.shape!=(env.num_envs,187):
        raise ValueError(f'Expected 187 terrain samples, got {tuple(heights.shape)}')
    if not allow_nonfinite and not heights.isfinite().all():
        raise ValueError('Terrain height sensor contains a miss/nonfinite value')
    return heights

def critic_observation(env):
    r=env.scene['robot']
    masses=body_masses(env,BODY_NAMES)
    friction=shared_robot_friction(env,r.indexing.geom_ids.long())
    force,torque=applied_base_wrench(env)
    _, critic=assemble(lin_vel=r.data.root_link_lin_vel_b,
        ang_vel=r.data.root_link_ang_vel_b,gravity=r.data.projected_gravity_b,
        commands=env.command_manager.get_command('twist')[:,:3],
        joint_pos=r.data.joint_pos,default_joint_pos=r.data.default_joint_pos,
        joint_vel=r.data.joint_vel,actions=env.action_manager.get_term('recovery').raw_action,
        push_force=force[:,:2],push_torque=torque,
        friction=friction,body_mass=masses[:,:1],limb_mass=masses[:,1:],
        root_z=r.data.root_link_pos_w[:,2],
        measured_heights=measured_terrain_heights(env))
    return critic.clamp(-100.,100.)

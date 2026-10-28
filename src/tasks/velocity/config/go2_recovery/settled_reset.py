"""Play-only hidden settling before policy starts; never used for training."""
import math
import torch
from mjlab.envs.mdp.events import reset_root_state_uniform

def settle_fallen(env, env_ids):
    if env.num_envs != 1:
        raise ValueError('Settled Viser reset supports one environment only')
    robot = env.scene['robot']
    # Existing reset events have already selected terrain and randomized pose.
    # Hold the reset joint configuration, without executing the learned policy.
    q = robot.data.joint_pos.clone()
    robot.set_joint_position_target(q)
    for attempt in range(12):
        # Terrain randomization has now run. Sample relative to its NEW origin.
        # Clearance exists only during this hidden preparation, never in the viewer.
        reset_root_state_uniform(env, env_ids, pose_range={
            # Sample across the selected terrain tile, not just beside its origin.
            # Rough terrain tiles are 8x8 m; leave a 0.5 m safety margin.
            'x': (-3.5, 3.5), 'y': (-3.5, 3.5), 'z': (.4, .4),
            'roll': (-math.pi, math.pi), 'pitch': (-math.pi, math.pi),
            'yaw': (-math.pi, math.pi)}, velocity_range={})
        robot.write_joint_velocity_to_sim(torch.zeros_like(robot.data.joint_vel))
        env.sim.forward()
        stable = 0
        for step in range(1600):
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(dt=env.physics_dt)
            env.sim.forward()
            speed = robot.data.root_link_vel_w.abs().max().item()
            joints = robot.data.joint_vel.abs().max().item()
            stable = stable + 1 if speed < .025 and joints < .08 else 0
            if stable >= 40:
                break
        env.sim.sense()
        contact = bool(env.scene['nonfoot_ground_touch'].data.found.any().item())
        fallen = robot.data.projected_gravity_b[0, 2].item() > -.7
        if stable >= 40 and fallen and contact:
            break
    else:
        raise RuntimeError('Could not obtain a stable fallen contact pose after 12 attempts')
    root = torch.cat((robot.data.root_link_pose_w.clone(),
                      torch.zeros((1,6),device=env.device)),dim=-1)
    robot.write_root_state_to_sim(root)
    robot.write_joint_velocity_to_sim(torch.zeros_like(robot.data.joint_vel))
    env.sim.forward()
    print(f'[settled-reset] attempts={attempt+1} contact={contact} hidden_steps={step+1} stable_steps={stable} '
          f'gravity_z={robot.data.projected_gravity_b[0,2].item():.3f}',flush=True)

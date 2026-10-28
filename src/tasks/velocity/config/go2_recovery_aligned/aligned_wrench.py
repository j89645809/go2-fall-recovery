"""World-frame applied base wrench, a documented correction to source labels.

Not contact force, not an impulse velocity. Zero is valid when no force acts.
No default disturbance strength is invented here: the caller supplies it.
"""
import torch

def applied_base_wrench(env):
    robot=env.scene['robot']
    base=robot.indexing.body_ids[robot.body_names.index('base')].long()
    wrench=env.sim.data.xfrc_applied[:,base,:]
    return wrench[:,:3],wrench[:,3:]

def write_base_wrench(env,env_ids,force,torque):
    if force.shape!=(len(env_ids),3) or torque.shape!=force.shape:
        raise ValueError('Expected one world-frame force and torque per selected environment')
    if not torch.isfinite(force).all() or not torch.isfinite(torque).all():
        raise ValueError('External wrench must be finite')
    robot=env.scene['robot']
    robot.write_external_wrench_to_sim(force[:,None,:],torque[:,None,:],
        env_ids=env_ids,body_ids=[robot.body_names.index('base')])

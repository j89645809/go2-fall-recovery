"""Engine-independent observation assembly matching GO2.compute_observations.

Inputs must already use original body/joint order and base-frame velocities.
Noise and final clipping belong to the engine adapter, not this function.
"""
import torch

def assemble(*, lin_vel, ang_vel, gravity, commands, joint_pos,
             default_joint_pos, joint_vel, actions, push_force, push_torque,
             friction, body_mass, limb_mass, root_z, measured_heights):
    n=actions.shape[0]
    fields = dict(lin_vel=(lin_vel,3),ang_vel=(ang_vel,3),gravity=(gravity,3),
        commands=(commands,3),joint_pos=(joint_pos,12),joint_vel=(joint_vel,12),
        actions=(actions,12),push_force=(push_force,2),push_torque=(push_torque,3),
        friction=(friction,1),body_mass=(body_mass,1),limb_mass=(limb_mass,18),
        measured_heights=(measured_heights,187))
    for name,(value,width) in fields.items():
        if value.shape != (n,width):
            raise ValueError(f'{name}: expected {(n,width)}, got {tuple(value.shape)}; do not pad missing body masses')
    q=joint_pos-default_joint_pos
    dq=joint_vel*.05
    command_input=commands*commands.new_tensor([2.,2.,.25])
    angular=ang_vel*.25
    actor=torch.cat((angular,gravity,command_input,q,dq,actions),dim=-1)
    heights=(root_z.unsqueeze(-1)-.5-measured_heights).clamp(-1.,1.)*5.
    critic=torch.cat((lin_vel*2.,angular,command_input,q,dq,actions,gravity,
                      push_force,push_torque,friction,body_mass/15.,limb_mass/5.,heights),dim=-1)
    assert actor.shape==(n,45) and critic.shape==(n,260)
    return actor,critic

"""All reward terms from a coherent post-physics snapshot.

Caller owns snapshot timing, source body ordering and reset/history updates.
This does not silently use a cached sensor from a different physics substep.
"""
from dataclasses import dataclass
import torch
from . import baseline_math as m

@dataclass
class RewardSnapshot:
    gravity: torch.Tensor
    angular_velocity: torch.Tensor
    root_z: torch.Tensor
    heights: torch.Tensor
    joints: torch.Tensor
    joint_velocity: torch.Tensor
    last_joint_velocity: torch.Tensor
    torques: torch.Tensor
    soft_limits: torch.Tensor
    base_contact_forces: torch.Tensor
    foot_contact_forces: torch.Tensor
    previous_filtered_foot_contact: torch.Tensor
    actions: torch.Tensor
    last_actions: torch.Tensor
    last_last_actions: torch.Tensor
    iteration: int
    dt: float

    def terms(self):
        return {
            'orientation':m.orientation(self.gravity),
            'orientation_pos_gaussian':m.orientation_pos_gaussian(self.gravity),
            'base_height_pos':m.base_height_pos(self.root_z,self.heights),
            'foot_contact_pos':m.foot_contact_pos(self.previous_filtered_foot_contact),
            'step2_target_pos':m.step2_target_pos(self.joints,self.gravity,self.iteration),
            'ang_vel_xyz':m.squared_norm(self.angular_velocity),
            'torques':m.squared_norm(self.torques),
            'dof_acc':m.dof_acc(self.last_joint_velocity,self.joint_velocity,self.dt),
            'dof_vel':m.squared_norm(self.joint_velocity),
            'dof_pos_limits':m.dof_pos_limits(self.joints,self.soft_limits),
            'collision':m.collision(self.base_contact_forces),
            'action':m.squared_norm(self.actions),
            'action_rate':m.squared_norm(self.last_actions-self.actions),
            'action_smoothness_2':m.action_smoothness_2(self.actions,self.last_actions,self.last_last_actions),
            'max_velocity':m.max_velocity(self.joint_velocity),
            'feet_stumble':m.feet_stumble(self.foot_contact_forces),
        }

    def compute(self):
        terms=self.terms()
        if any(not torch.isfinite(value).all() for value in terms.values()):
            raise ValueError('Nonfinite reward state; do not conceal invalid physics')
        return m.total_reward(terms,self.dt),terms

def next_contact_history(foot_forces,last_contacts):
    # MuJoCo reports the force on each foot pointing toward the terrain;
    # its vertical component is negative for a supported foot in this model.
    contact=foot_forces.norm(dim=-1)>1.
    return contact|last_contacts,contact

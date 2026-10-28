"""Integrated single-term reward preserving FR-Net's final positive clipping."""
import torch
import os
from .aligned_critic import measured_terrain_heights
from .aligned_reward_state import RewardSnapshot, next_contact_history
from .baseline_math import SCALES
from .official_go2_limits import POSITION_LOWER, POSITION_UPPER, VELOCITY_LIMIT


def source_iteration(common_step_counter,rollout_steps=24):
    return max(common_step_counter-1,0)//rollout_steps


class FrnetTotalReward:
    """Stateful reward adapter evaluated once after every policy step.

    mjlab's RewardManager scales a returned rate by dt.  This class computes
    FR-Net's already-dt-scaled total and divides by dt so the manager's final
    multiplication produces exactly one dt factor and clips only after all 16
    weighted terms have been summed.
    """
    def __init__(self, cfg, env):
        del cfg
        self.last_joint_velocity=torch.zeros((env.num_envs,12),device=env.device)
        self.last_contact=torch.zeros((env.num_envs,4),dtype=torch.bool,device=env.device)
        self.filtered_contact=torch.zeros_like(self.last_contact)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids=slice(None)
        self.last_joint_velocity[env_ids]=0
        self.last_contact[env_ids]=False
        self.filtered_contact[env_ids]=False

    def __call__(self, env):
        # FR-Net refreshes contact/height tensors before computing reward. The
        # generic mjlab step intentionally postpones this refresh until after
        # reward, so do it explicitly here to remove the one-substep lag.
        env.sim.forward()
        env.sim.sense()
        robot=env.scene['robot']
        feet=env.scene['feet_ground_contact'].data.force
        base=env.scene['base_ground_contact'].data.force
        if feet is None or base is None or feet.shape!=(env.num_envs,4,3):
            raise ValueError('FR-Net reward requires 4 foot forces and base forces')
        action=env.action_manager.get_term('recovery')
        # Reward is evaluated after termination detection but before reset.
        # A rare MuJoCo contact blow-up can therefore reach this function for
        # one final sample. Detect non-finite values and finite values far
        # outside Go2's physical envelope, then bound only that terminal sample.
        heights=measured_terrain_heights(env, allow_nonfinite=True)
        valid=physical_state_valid(robot, heights, action)
        if not valid.all():
            env.invalid_physics_count=(
                getattr(env, 'invalid_physics_count', 0)
                + int((~valid).sum().item())
            )
        def safe(x, limit):
            return torch.nan_to_num(
                x, nan=0.0, posinf=limit, neginf=-limit
            ).clamp(-limit, limit)
        snapshot=RewardSnapshot(
            gravity=safe(robot.data.projected_gravity_b, 2.0),
            angular_velocity=safe(robot.data.root_link_ang_vel_b, 100.0),
            root_z=safe(robot.data.root_link_pos_w[:,2], 100.0),
            heights=safe(heights, 100.0),
            joints=safe(robot.data.joint_pos, 20.0),
            joint_velocity=safe(robot.data.joint_vel, 200.0),
            last_joint_velocity=safe(self.last_joint_velocity, 200.0),
            torques=safe(robot.data.actuator_force, 1000.0),
            soft_limits=robot.data.soft_joint_pos_limits,
            base_contact_forces=safe(base, 1.0e6),
            foot_contact_forces=safe(feet, 1.0e6),
            previous_filtered_foot_contact=self.filtered_contact,
            actions=safe(action.current, 100.0),
            last_actions=safe(action.previous, 100.0),
            last_last_actions=safe(action.previous_previous, 100.0),
            # Runner assigns source env.it before collecting each 24-step
            # rollout. mjlab increments common_step_counter before reward, so
            # subtract one to avoid advancing the target on the final sample.
            iteration=(source_iteration(env.common_step_counter)
                       +int(os.environ.get('FRNET_SOURCE_ITERATION_OFFSET','0'))),
            dt=env.step_dt,
        )
        reward,terms=snapshot.compute()
        env.frnet_reward_terms=terms
        self.last_joint_velocity.copy_(robot.data.joint_vel)
        filtered,last=next_contact_history(feet,self.last_contact)
        self.filtered_contact.copy_(filtered)
        self.last_contact.copy_(last)
        reward=torch.where(valid, reward, torch.zeros_like(reward))
        return reward/env.step_dt


class SourceRotationTimeout:
    """Independent consecutive counter matching GO2RecoveryEnv.rotate_time.

    The reference predicate ``(g_z > .8) | (g_z < 1.2)`` is always true for
    finite g_z.  We preserve that source behavior, including its strict >400
    threshold, without incorrectly tying it to randomized episode length.
    """
    def __init__(self,cfg,env):
        del cfg
        self.rotate_time=torch.zeros(env.num_envs,dtype=torch.int32,device=env.device)

    def reset(self,env_ids=None):
        if env_ids is None:
            env_ids=slice(None)
        self.rotate_time[env_ids]=0

    def __call__(self,env):
        self.rotate_time+=1
        return self.rotate_time>400


def weighted_reward_metric(env,name):
    """Per-step weighted rate for inspecting every source reward curve."""
    return env.frnet_reward_terms[name]*SCALES[name]


def _bounded(value, limit):
    return value.isfinite().all(dim=-1) & (value.abs() <= limit).all(dim=-1)


def physical_state_valid(robot, heights, action):
    """Return environments inside a generous, hardware-impossible envelope."""
    return (
        _bounded(robot.data.root_link_pos_w, 100.0)
        & _bounded(robot.data.root_link_lin_vel_w, 100.0)
        & _bounded(robot.data.root_link_ang_vel_b, 100.0)
        & _bounded(robot.data.projected_gravity_b, 2.0)
        & _bounded(robot.data.joint_pos, 20.0)
        & _bounded(robot.data.joint_vel, 200.0)
        & _bounded(robot.data.actuator_force, 1000.0)
        & _bounded(heights, 100.0)
        & _bounded(action.current, 100.0)
    )


def invalid_physics(env):
    robot=env.scene['robot']
    action=env.action_manager.get_term('recovery')
    heights=measured_terrain_heights(env, allow_nonfinite=True)
    return ~physical_state_valid(robot, heights, action)


def hardware_velocity_margin(env):
    """Squared normalized penalty above 70% of the Go2 rated speed."""
    vel=env.scene["robot"].data.joint_vel
    limit=vel.new_tensor(VELOCITY_LIMIT)
    return ((vel.abs()/limit-0.70).clamp_min(0.0).square()).sum(dim=-1)

def hardware_position_margin(env):
    """Penalty in the outer 10% of each certified joint range."""
    pos=env.scene["robot"].data.joint_pos
    lower=pos.new_tensor(POSITION_LOWER); upper=pos.new_tensor(POSITION_UPPER)
    span=upper-lower; safe_lower=lower+0.10*span; safe_upper=upper-0.10*span
    return (((safe_lower-pos).clamp_min(0.0)/span).square()+((pos-safe_upper).clamp_min(0.0)/span).square()).sum(dim=-1)

import math
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import EventTermCfg, RewardTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_rough_env_cfg
from .strict_rewards import orientation_pos_gaussian, base_height_pos, foot_contact_pos, step2_target_pos
from . import strict_obs

def unitree_go2_recovery_strict_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = unitree_go2_rough_env_cfg(play=play)
    cfg.scene.num_envs = 4096 if not play else 1
    cfg.episode_length_s = 24.0 if not play else 1e9
    actor = cfg.observations["actor"]
    actor.terms.pop("phase", None); actor.terms.pop("height_scan", None)
    for n in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
        actor.terms[n].noise = None
    # Exact FR-Net privileged layout: actor(45) + base velocity(3) +
    # push force(2) + push torque(3) + friction(1) + body mass(1) +
    # limb masses(18) + terrain heights(187) = 260.
    critic = cfg.observations["critic"]
    old_critic_terms = dict(critic.terms)
    critic.terms = dict(actor.terms)
    critic.terms.update({
      "base_lin_vel": old_critic_terms["base_lin_vel"],
      "push_force": ObservationTermCfg(func=strict_obs.push_force),
      "push_torque": ObservationTermCfg(func=strict_obs.push_torque),
      "friction": ObservationTermCfg(func=strict_obs.friction),
      "body_mass": ObservationTermCfg(func=strict_obs.body_mass),
      "limb_mass": ObservationTermCfg(func=strict_obs.limb_mass),
      "height_scan": old_critic_terms["height_scan"],
    })
    # Original recovery reset: arbitrary fallen pose, small initial velocity.
    cfg.events["reset_base"] = EventTermCfg(
      func=mdp.reset_root_state_uniform, mode="reset",
      params={"pose_range":{"x":(-.2,.2),"y":(-.2,.2),"z":(.35,.65),
                              "roll":(-math.pi,math.pi),"pitch":(-math.pi,math.pi),"yaw":(-math.pi,math.pi)},
              "velocity_range":{"x":(-.2,.2),"y":(-.2,.2),"z":(-.2,.2),
                                 "roll":(-.5,.5),"pitch":(-.5,.5),"yaw":(-.5,.5)}})
    # The Isaac Gym baseline has timeout/rotation termination, but no illegal-contact termination.
    cfg.terminations.pop("illegal_contact", None)
    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(179.0)
    for name in ("track_linear_velocity","track_angular_velocity","pose","foot_gait","foot_clearance",
                 "foot_slip","soft_landing","stand_still","angular_momentum","is_terminated"):
        if name in cfg.rewards: cfg.rewards[name].weight = 0.0
    cfg.rewards["body_orientation_l2"].weight = -0.5
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["orientation_pos_gaussian"] = RewardTermCfg(func=orientation_pos_gaussian, weight=6.0)
    cfg.rewards["base_height_pos"] = RewardTermCfg(func=base_height_pos, weight=1.0)
    cfg.rewards["foot_contact_pos"] = RewardTermCfg(func=foot_contact_pos, weight=0.1)
    cfg.rewards["step2_target_pos"] = RewardTermCfg(func=step2_target_pos, weight=6.0)
    if play:
        from .settled_reset import settle_fallen
        # For Viser evaluation, start already in contact with the terrain:
        # random fallen orientation, zero initial velocity, no airborne drop.
        cfg.events["reset_base"] = EventTermCfg(
          func=mdp.reset_root_state_uniform, mode="reset",
          # z is an offset from the Go2 default root height (0.32 m).
          # Keep it near zero so reset starts on the terrain, not airborne.
          params={"pose_range":{"x":(-3.5,3.5),"y":(-3.5,3.5),"z":(-.04,.04),
                                  "roll":(-math.pi,math.pi),"pitch":(-math.pi,math.pi),"yaw":(-math.pi,math.pi)},
                  "velocity_range":{"x":(0.,0.),"y":(0.,0.),"z":(0.,0.),
                                     "roll":(0.,0.),"pitch":(0.,0.),"yaw":(0.,0.)}})
        cfg.events.pop("push_robot", None); cfg.terminations.pop("fell_over", None)
        cfg.events["settle_fallen"] = EventTermCfg(func=settle_fallen, mode="reset")
    return cfg

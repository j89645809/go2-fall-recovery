"""mjlab port of the FR-Net Go2 recovery baseline."""
import math
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import TerminationTermCfg, EventTermCfg, RewardTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.assets.robots import get_go2_robot_cfg
from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_rough_env_cfg
from src.tasks.velocity.mdp.recovery_rewards import upright_gaussian, base_height_gaussian

def unitree_go2_recovery_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = unitree_go2_rough_env_cfg(play=play)
    cfg.scene.num_envs = 4096
    cfg.episode_length_s = 20.0
    # FR-Net actor input: ang_vel(3), gravity(3), command(3), q(12), dq(12), action(12) = 45.
    actor = cfg.observations["actor"]
    actor.terms.pop("phase", None)
    actor.terms.pop("height_scan", None)
    actor.terms["base_ang_vel"].noise = None
    actor.terms["projected_gravity"].noise = None
    actor.terms["joint_pos"].noise = None
    actor.terms["joint_vel"].noise = None
    # Random fallen pose at every reset (mjlab samples Euler ranges and converts to qpos).
    cfg.events["reset_base"] = EventTermCfg(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2),
                                "z": (0.35, 0.65), "roll": (-math.pi, math.pi),
                                "pitch": (-math.pi, math.pi), "yaw": (-math.pi, math.pi)},
                "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2),
                                    "z": (-0.2, 0.2), "roll": (-0.5, 0.5),
                                    "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5)}})
    # Recovery shaping: upright orientation and stand height dominate locomotion terms.
    for name in ("track_linear_velocity", "track_angular_velocity", "pose", "foot_gait",
                 "foot_clearance", "foot_slip", "soft_landing", "stand_still",
                 "angular_momentum", "is_terminated"):
        if name in cfg.rewards: cfg.rewards[name].weight = 0.0
    cfg.rewards["upright_gaussian"] = RewardTermCfg(func=upright_gaussian, weight=6.0)
    cfg.rewards["base_height_gaussian"] = RewardTermCfg(func=base_height_gaussian, weight=1.0)
    cfg.rewards["body_orientation_l2"].weight = -0.5
    cfg.rewards["body_orientation_l2"].params["asset_cfg"] = SceneEntityCfg("robot", body_names=("base_link",))
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(179.0)
    if play:
        cfg.scene.num_envs = 1
        cfg.events.pop("push_robot", None)
        cfg.terminations.pop("fell_over", None)
        cfg.episode_length_s = 1e9
    return cfg

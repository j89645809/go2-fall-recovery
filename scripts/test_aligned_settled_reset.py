import torch
import os

import mjlab.tasks
import src.tasks.velocity.config.go2_recovery_aligned
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


torch.manual_seed(42)
cfg = load_env_cfg("Unitree-Go2-Recovery-FRNet-Aligned", play=True)
cfg.scene.num_envs = 1
cfg.viewer = None
env = ManagerBasedRlEnv(cfg=cfg, device=os.environ.get("TEST_DEVICE", "cuda:0"))
for i in range(12):
    env.reset()
    robot = env.scene["robot"]
    root_speed = robot.data.root_link_vel_w.abs().max().item()
    joint_speed = robot.data.joint_vel.abs().max().item()
    gravity_z = robot.data.projected_gravity_b[0, 2].item()
    contact = env.scene["nonfoot_ground_touch"].data.found.any().item()
    assert root_speed == 0.0
    assert joint_speed == 0.0
    assert gravity_z > -0.7
    assert contact
    print(
        "RESET_TEST", i, "root_speed", root_speed,
        "joint_speed", joint_speed, "gravity_z", gravity_z,
        "nonfoot_contact", bool(contact), flush=True,
    )
env.close()

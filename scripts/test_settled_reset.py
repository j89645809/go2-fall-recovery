import torch
import mjlab.tasks
import src.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

torch.manual_seed(42)
cfg = load_env_cfg('Unitree-Go2-Recovery-Strict', play=True)
cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
for i in range(8):
    env.reset()
    r = env.scene['robot']
    assert r.data.root_link_vel_w.abs().max().item() == 0
    assert r.data.joint_vel.abs().max().item() == 0
    assert r.data.projected_gravity_b[0, 2].item() > -.7
    assert env.scene['nonfoot_ground_touch'].data.found.any().item()
    print('RESET_TEST', i, 'velocity', r.data.root_link_vel_w.abs().max().item(),
          'joint_velocity', r.data.joint_vel.abs().max().item(),
          'gravity_z', r.data.projected_gravity_b[0, 2].item(), flush=True)
env.close()

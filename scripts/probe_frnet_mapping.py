"""Read-only engine mapping audit; no training and no viewer mutations."""
import json
import mujoco
import mjlab.tasks
import src.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv

cfg = load_env_cfg('Unitree-Go2-Recovery-Strict', play=True)
cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
robot = env.scene['robot']
print('MAPPING_BEGIN', flush=True)
print('joints', robot.joint_names, flush=True)
print('bodies', robot.body_names, flush=True)
print('indexing', robot.indexing, flush=True)
for name in ('body_mass', 'geom_friction', 'body_inertia'):
    value = getattr(env.sim.model, name)
    print(name, value.shape, value.detach().cpu().tolist(), flush=True)
print('timing', env.physics_dt, env.step_dt, flush=True)
print('MAPPING_END', flush=True)
env.close()

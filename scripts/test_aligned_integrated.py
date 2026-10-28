"""Live integration gate for the complete FR-Net-aligned training task."""
import torch
import src.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from src.tasks.velocity.config.go2_recovery_aligned.baseline_math import SCALES
from src.tasks.velocity.config.go2_recovery_aligned.official_go2_limits import position_limits
from src.tasks.velocity.config.go2_recovery_aligned.aligned_reward_term import source_iteration,SourceRotationTimeout

TASK='Unitree-Go2-Recovery-FRNet-Aligned'
cfg=load_env_cfg(TASK)
cfg.scene.num_envs=8
assert cfg.sim.mujoco.timestep==.005 and cfg.decimation==4
assert cfg.scene.terrain.terrain_generator.num_rows==10
assert cfg.scene.terrain.terrain_generator.num_cols==8
assert cfg.scene.terrain.terrain_generator.border_width==5.0
assert cfg.episode_length_s==24.
agent=load_rl_cfg(TASK)
assert agent.num_steps_per_env==24 and agent.max_iterations==4000
assert [source_iteration(i) for i in (1,24,25,48,49)]==[0,0,1,1,2]
fake=type('FakeEnv',(),{'num_envs':2,'device':'cuda:0'})()
rotation_gate=SourceRotationTimeout(None,fake)
for _ in range(400):
    assert not rotation_gate(fake).any()
assert rotation_gate(fake).all()
rotation_gate.reset(torch.tensor([1],device='cuda:0'))
assert rotation_gate.rotate_time.tolist()==[401,0]
env=ManagerBasedRlEnv(cfg,device='cuda:0')
obs,_=env.reset(seed=55)
assert obs['actor'].shape==(8,45)
assert obs['critic'].shape==(8,260)
assert torch.isfinite(obs['actor']).all() and torch.isfinite(obs['critic']).all()
robot=env.scene['robot']
lower,upper=position_limits(device=env.device,dtype=robot.data.joint_pos.dtype)
assert ((robot.data.joint_pos>=lower)&(robot.data.joint_pos<=upper)).all()
assert env.physics_dt==.005 and env.step_dt==.02
assert tuple(robot.joint_names)==tuple(
    f'{leg}_{joint}_joint' for leg in ('FL','FR','RL','RR')
    for joint in ('hip','thigh','calf'))
for _ in range(32):
    obs,reward,terminated,truncated,extras=env.step(torch.randn((8,12),device=env.device))
    assert obs['actor'].shape==(8,45) and obs['critic'].shape==(8,260)
    assert torch.isfinite(obs['actor']).all() and torch.isfinite(obs['critic']).all()
    assert torch.isfinite(reward).all() and (reward>=0).all()
    assert set(env.frnet_reward_terms)==set(SCALES)
critic=obs['critic']
assert (critic[:,53:73].abs().sum(-1)>0).all()  # friction and 19 real masses
assert critic[:,73:].std()>0                    # 187 live terrain heights
assert env.scene['feet_ground_contact'].data.force is not None
assert env.scene['base_ground_contact'].data.force is not None
actor_term=env.observation_manager._group_obs_term_cfgs['actor'][0].func
assert actor_term.max_lag==1 and set(actor_term.lag.tolist())<=set((0,1))
actor_term.enable_noise=False
actor_term.lag[:]=1
previous_proprio=actor_term.buffer[0].clone()
delayed_obs=actor_term(env)
delayed_proprio=torch.cat((delayed_obs[:,:6],delayed_obs[:,9:33]),dim=-1)
torch.testing.assert_close(delayed_proprio,previous_proprio)
rotation_index=env.termination_manager.active_terms.index('source_rotation_timeout')
rotation_term=env.termination_manager._term_cfgs[rotation_index].func
assert (rotation_term.rotate_time==32).all()
action=env.action_manager.get_term('recovery')
assert action.delay.buffer.shape==(6,8,12)
assert len(env.metrics_manager.active_terms)==16
print('PASS: registered integrated task; 45D actor, 260D critic, 16 rewards, '
      'Go2-limited source resets, 50Hz PD/action delay, sensor latency/noise and live DR')
env.close()

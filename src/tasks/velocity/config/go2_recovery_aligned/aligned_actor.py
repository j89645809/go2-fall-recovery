"""45D actor adapter; current clipped actions, not delayed PD commands."""
import os
import torch
from .baseline_protocol import JOINT_NAMES, actor_noise_std


def _actor_fields(env):
    r=env.scene['robot']
    if tuple(r.joint_names)!=JOINT_NAMES:
        raise ValueError('Actor joint ordering mismatch')
    commands=env.command_manager.get_command('twist')[:,:3]
    action=env.action_manager.get_term('recovery').raw_action
    proprio=torch.cat((r.data.root_link_ang_vel_b*.25,
        r.data.projected_gravity_b,
        r.data.joint_pos-r.data.default_joint_pos,
        r.data.joint_vel*.05),dim=-1)
    return proprio,commands*commands.new_tensor([2.,2.,.25]),action


def _assemble(proprio,commands,action):
    # Proprio layout is [ang_vel(3), gravity(3), q(12), dq(12)].
    obs=torch.cat((proprio[:,:6],commands,proprio[:,6:],action),dim=-1)
    if obs.shape[1]!=45:
        raise ValueError(f'Unexpected actor shape {obs.shape}')
    return obs

def actor_observation(env, noise_enabled=True, noise_level=1.):
    proprio,commands,action=_actor_fields(env)
    obs=_assemble(proprio,commands,action)
    if noise_enabled:
        obs=obs+torch.randn_like(obs)*actor_noise_std(obs,noise_level)
    return obs.clamp(-100.,100.)


class FrnetActorSensorModel:
    """45D actor with source Gaussian noise and configurable sensor latency.

    Commands and previous actions are software-side quantities and remain
    current. Only IMU/encoder-derived proprioception is delayed. A 0..1 policy
    step range models up to 20 ms latency and can be replaced by calibrated Go2
    values without changing the observation contract.
    """
    def __init__(self,cfg,env):
        params=cfg.params
        self.enable_noise=params.get('noise_enabled',True)
        self.noise_level=params.get('noise_level',1.)
        self.max_lag=params.get('sensor_delay_max_steps',1)
        if self.max_lag<0:
            raise ValueError('Sensor delay cannot be negative')
        self.sim2real_profile=os.environ.get('FRNET_SIM2REAL_PROFILE','0') == '1'
        self.drift_initial=float(os.environ.get('FRNET_SIM2REAL_DRIFT_INITIAL','0.02'))
        self.drift_random_walk=float(os.environ.get('FRNET_SIM2REAL_DRIFT_RANDOM_WALK','0.0005'))
        if self.drift_initial < 0.0 or self.drift_random_walk < 0.0:
            raise ValueError('sim2real drift values must be non-negative')
        self.buffer=torch.zeros((self.max_lag+1,env.num_envs,30),device=env.device)
        self.lag=torch.randint(0,self.max_lag+1,(env.num_envs,),device=env.device)
        self.bias=torch.zeros((env.num_envs,30),device=env.device)
        self.ready=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)

    def reset(self,env_ids=None):
        if env_ids is None:
            env_ids=slice(None)
        self.buffer[:,env_ids]=0
        if isinstance(env_ids,slice):
            count=self.buffer.shape[1]
        else:
            count=len(env_ids)
        self.lag[env_ids]=torch.randint(0,self.max_lag+1,(count,),device=self.lag.device)
        if self.sim2real_profile:
            self.bias[env_ids]=torch.randn((count,30),device=self.bias.device)*self.drift_initial
        else:
            self.bias[env_ids]=0
        self.ready[env_ids]=False

    def __call__(self,env,**kwargs):
        del kwargs
        proprio,commands,action=_actor_fields(env)
        if self.sim2real_profile and self.drift_random_walk > 0.0:
            self.bias.mul_(0.9995).add_(
                torch.randn_like(self.bias)*self.drift_random_walk
            )
            self.bias.clamp_(-0.10,0.10)
            proprio=proprio+self.bias
        missing=~self.ready
        if missing.any():
            self.buffer[:,missing]=proprio[missing].unsqueeze(0)
            self.ready[missing]=True
        self.buffer=self.buffer.roll(1,dims=0)
        self.buffer[0]=proprio
        ids=torch.arange(env.num_envs,device=env.device)
        delayed=self.buffer[self.lag,ids]
        obs=_assemble(delayed,commands,action)
        if self.enable_noise:
            obs=obs+torch.randn_like(obs)*actor_noise_std(obs,self.noise_level)
        return obs.clamp(-100.,100.)

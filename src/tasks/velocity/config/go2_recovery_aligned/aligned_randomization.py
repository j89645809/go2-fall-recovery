"""Source parameter ranges. Inertia equivalence still needs engine comparison."""
import os
from mjlab.managers import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs.mdp import dr
from .aligned_asset import BODY_NAMES
import torch
from mjlab.managers.event_manager import requires_model_fields, RecomputeLevel

@requires_model_fields('body_mass','body_inertia',recompute=RecomputeLevel.set_const)
def mass_and_inertia(env, env_ids):
    """Source mass ranges with consistent fixed-shape density scaling.

    This preserves COM and principal axes. It is not a claim that PhysX's
    recomputeInertia=True reconstructs identical inertias from collision shapes.
    """
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    robot=env.scene['robot']
    local=[robot.body_names.index(name) for name in BODY_NAMES]
    bodies=robot.indexing.body_ids[local].long()
    default_mass=env.sim.get_default_field('body_mass')[bodies]
    default_inertia=env.sim.get_default_field('body_inertia')[bodies]
    positive = default_mass > 1e-8
    masses=default_mass[None,:]*(.9+.2*torch.rand((len(ids),19),device=env.device))
    # The exact ZIP model has no head/foot inertial bodies; those placeholders
    # exist only to preserve the 19-channel critic contract. Keep them exactly
    # massless instead of forming 0/0 during randomized inertia scaling.
    masses[:, ~positive] = 0.0
    masses[:,0]=default_mass[0]+1.0*torch.rand(len(ids),device=env.device)-.5
    if positive.any() and not (masses[:, positive] > 0).all():
        raise ValueError('Randomized body mass must remain positive')
    ratio=torch.where(positive[None,:], masses/default_mass[None,:], torch.ones_like(masses))
    env.sim.model.body_mass[ids[:,None],bodies[None,:]]=masses
    env.sim.model.body_inertia[ids[:,None],bodies[None,:]]=default_inertia[None,:,:]*ratio[:,:,None]

@requires_model_fields('geom_friction')
def bucket_friction(env, env_ids):
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    if not hasattr(env, 'frnet_friction_buckets'):
        env.frnet_friction_buckets = .5+.75*torch.rand(64,device=env.device)
    values = env.frnet_friction_buckets[torch.randint(64,(len(ids),),device=env.device)]
    geoms = env.scene['robot'].indexing.geom_ids.long()
    env.sim.model.geom_friction[ids[:,None],geoms[None,:],0] = values[:,None]

def shared_pd_factors(env, env_ids):
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    kp = .8+.4*torch.rand((len(ids),1),device=env.device)
    kd = .8+.4*torch.rand((len(ids),1),device=env.device)
    # Source draws one Kp and one Kd factor per environment, shared by all joints.
    for actuator in env.scene['robot'].actuators:
        actuator.set_gains(ids,kp=actuator.default_stiffness[ids]*kp,
                          kd=actuator.default_damping[ids]*kd)



@requires_model_fields("geom_solref")
def crossbackend_contact_compliance(env, env_ids):
    """Randomize contact time constant and damping for cross-backend training.

    This is opt-in so the frozen baseline keeps its original dynamics.
    The same sampled parameters are used for all geoms in one environment.
    """
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    n_geom = env.sim.model.geom_solref.shape[1]
    geoms = torch.arange(n_geom, device=env.device, dtype=torch.long)
    default = env.sim.get_default_field("geom_solref")
    time_scale = 0.70 + 0.60 * torch.rand((len(ids), n_geom), device=env.device)
    damp_scale = 0.70 + 0.60 * torch.rand((len(ids), n_geom), device=env.device)
    solref = default[None, :, :].expand(len(ids), -1, -1).clone()
    solref[:, :, 0] = (solref[:, :, 0] * time_scale).clamp_min(1.0e-4)
    solref[:, :, 1] = (solref[:, :, 1] * damp_scale).clamp(0.05, 1.0)
    env.sim.model.geom_solref[ids[:, None], geoms[None, :], :] = solref

def parameter_events():
    events = {
        'mass_inertia': EventTermCfg(func=mass_and_inertia, mode='startup'),
        'friction': EventTermCfg(func=bucket_friction, mode='startup'),
        'pd_gains': EventTermCfg(func=shared_pd_factors, mode='startup'),
        'base_com': EventTermCfg(func=dr.body_com_offset, mode='startup', params={
            'asset_cfg': SceneEntityCfg('robot', body_names=('base',)),
            'operation':'add',
            'ranges':{0:(-.03,.03),1:(-.03,.03),2:(-.03,.03)},
        }),
    }
    if os.environ.get('FRNET_CROSSBACKEND_RANDOMIZATION', '0') == '1':
        events['contact_compliance'] = EventTermCfg(func=crossbackend_contact_compliance, mode='startup')
        events['torque_derating'] = EventTermCfg(func=dr.effort_limits, mode='startup', params={
            'effort_limit_range': (0.85, 1.0), 'operation': 'scale'
        })
    return events

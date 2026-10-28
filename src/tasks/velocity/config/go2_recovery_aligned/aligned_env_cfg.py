"""One integrated, isolated FR-Net-compatible mjlab training task."""
from pathlib import Path
import math
import os
from mjlab.envs import mdp as env_mdp
from mjlab.managers import EventTermCfg,MetricsTermCfg,ObservationGroupCfg,ObservationTermCfg,RewardTermCfg,TerminationTermCfg
from mjlab.sensor import ContactMatch,ContactSensorCfg,GridPatternCfg,ObjRef,RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_rough_env_cfg
from .aligned_actions import RecoveryActionCfg
from .aligned_actor import FrnetActorSensorModel
from .aligned_asset import get_robot_cfg
from .aligned_critic import critic_observation
from .aligned_randomization import parameter_events
from .aligned_reset import reset_source_state
from .aligned_reward_term import FrnetTotalReward,SourceRotationTimeout,weighted_reward_metric,invalid_physics,hardware_velocity_margin,hardware_position_margin
from .recovery_shaping import recovery_height_crossing,recovery_supported_height,recovery_stable_stand,RecoverySuccessBonus
from .baseline_math import SCALES
from .settled_reset import settle_fallen


def _asset_path():
    # src/tasks/velocity/config/<package>/file.py -> repository root
    # Exact Unitree Go2 MJCF extracted from the user's unitree_go2.zip.
    return Path(__file__).resolve().parents[5]/'frnet_aligned_assets'/'real_go2_head.xml'


def frnet_aligned_env_cfg(play=False):
    cfg=unitree_go2_rough_env_cfg(play=False)
    cfg.scene.entities={'robot':get_robot_cfg(_asset_path())}
    cfg.scene.num_envs=1 if play else 4096
    cfg.scene.extent=2.0
    targeted_v2=os.environ.get('PYTHON_FRNET_TARGETED_V2','0')=='1'
    static_only=os.environ.get('PYTHON_FRNET_STATIC_ONLY','0')=='1'
    acceptance_aligned=os.environ.get('PYTHON_FRNET_ACCEPTANCE_ALIGNED','0')=='1'
    staged_unfold=os.environ.get('PYTHON_FRNET_STAGED_UNFOLD','0')=='1'
    predictive_all_joints=os.environ.get('PYTHON_FRNET_PREDICTIVE_ALL_JOINTS','0')=='1'
    predictive_general_trigger=float(os.environ.get('FRNET_PREDICTIVE_GENERAL_TRIGGER','0.50'))
    predictive_calf_trigger=float(os.environ.get('FRNET_PREDICTIVE_CALF_TRIGGER','0.30'))
    predictive_general_offset=float(os.environ.get('FRNET_PREDICTIVE_GENERAL_OFFSET','0.015'))
    predictive_calf_offset=float(os.environ.get('FRNET_PREDICTIVE_CALF_OFFSET','0.030'))
    predictive_brake_release_ratio=float(os.environ.get('FRNET_PREDICTIVE_BRAKE_RELEASE_RATIO','0.20'))
    predictive_latch_enabled=os.environ.get('PYTHON_FRNET_PREDICTIVE_LATCH','1')=='1'
    velocity_brake_enabled=os.environ.get('PYTHON_FRNET_VELOCITY_BRAKE','0')=='1'
    velocity_brake_general_trigger=float(os.environ.get('FRNET_VELOCITY_BRAKE_GENERAL_TRIGGER','0.80'))
    velocity_brake_calf_trigger=float(os.environ.get('FRNET_VELOCITY_BRAKE_CALF_TRIGGER','0.70'))
    safety_general_margin=float(os.environ.get('FRNET_SAFETY_GENERAL_MARGIN','0.020'))
    safety_calf_margin=float(os.environ.get('FRNET_SAFETY_CALF_MARGIN','0.040'))
    initial_target_hold_steps=int(os.environ.get('FRNET_INITIAL_TARGET_HOLD_STEPS','0'))
    global_impact_guard_enabled=(
        os.environ.get(
            'PYTHON_FRNET_GLOBAL_IMPACT_GUARD',
            os.environ.get('FRNET_GLOBAL_IMPACT_TRANSFER_GUARD','0'),
        ) == '1'
    )
    global_impact_guard_trigger=float(os.environ.get('FRNET_GLOBAL_IMPACT_GUARD_TRIGGER','0.75'))
    global_impact_guard_margin=float(os.environ.get('FRNET_GLOBAL_IMPACT_GUARD_MARGIN','0.10'))
    global_impact_guard_cooldown_steps=int(os.environ.get('FRNET_GLOBAL_IMPACT_GUARD_COOLDOWN_STEPS','3'))
    global_impact_guard_cross_joint_scale=float(os.environ.get('FRNET_GLOBAL_IMPACT_GUARD_CROSS_SCALE','1.0'))
    global_impact_guard_target_error=float(os.environ.get('FRNET_GLOBAL_IMPACT_GUARD_TARGET_ERROR','0.25'))
    if not 0.0 <= safety_general_margin < 0.40:
        raise ValueError('FRNET_SAFETY_GENERAL_MARGIN must be in [0, 0.40) radians')
    if not 0.0 <= safety_calf_margin < 0.40:
        raise ValueError('FRNET_SAFETY_CALF_MARGIN must be in [0, 0.40) radians')
    if not 0 <= initial_target_hold_steps <= 100:
        raise ValueError('FRNET_INITIAL_TARGET_HOLD_STEPS must be in [0, 100]')
    if not 0.0 < global_impact_guard_trigger <= 1.5:
        raise ValueError('FRNET_GLOBAL_IMPACT_GUARD_TRIGGER must be in (0, 1.5]')
    if not 0.0 <= global_impact_guard_margin < 0.40:
        raise ValueError('FRNET_GLOBAL_IMPACT_GUARD_MARGIN must be in [0, 0.40) radians')
    if not 0 <= global_impact_guard_cooldown_steps <= 120:
        raise ValueError('FRNET_GLOBAL_IMPACT_GUARD_COOLDOWN_STEPS must be in [0, 120]')
    if not 0.0 <= global_impact_guard_cross_joint_scale <= 1.0:
        raise ValueError('FRNET_GLOBAL_IMPACT_GUARD_CROSS_SCALE must be in [0, 1]')
    if not 0.05 <= global_impact_guard_target_error <= 1.0:
        raise ValueError('FRNET_GLOBAL_IMPACT_GUARD_TARGET_ERROR must be in [0.05, 1.0]')
    if not 0.0 < velocity_brake_general_trigger <= 1.0:
        raise ValueError('FRNET_VELOCITY_BRAKE_GENERAL_TRIGGER must be in (0, 1]')
    if not 0.0 < velocity_brake_calf_trigger <= 1.0:
        raise ValueError('FRNET_VELOCITY_BRAKE_CALF_TRIGGER must be in (0, 1]')
    cfg.episode_length_s=12.0 if targeted_v2 else 24.0
    cfg.sim.mujoco.timestep=.005
    cfg.decimation=4
    # Match FR-Net's recovery terrain layout: 10 difficulty rows, 8 terrain
    # columns, 8m sub-terrain patches and a 5m global border. The inherited
    # mjlab Go2 velocity task uses 20 columns/20m border, which would change
    # the mapping between the 187 height samples and terrain locations.
    terrain_generator=cfg.scene.terrain.terrain_generator
    if terrain_generator is None:
        raise ValueError('FR-Net aligned task requires generated rough terrain')
    terrain_generator.num_rows=10
    terrain_generator.num_cols=8
    terrain_generator.size=(8.0,8.0)
    terrain_generator.border_width=5.0
    # Staged Go2 slope curriculum. Default is about 5 degrees; later stages
    # use FRNET_SLOPE_MAX_DEG=7, 9, or 12 without editing the task code.
    slope_max_deg=float(os.environ.get('FRNET_SLOPE_MAX_DEG','5.0'))
    if not 0.0 < slope_max_deg <= 12.0:
        raise ValueError('FRNET_SLOPE_MAX_DEG must be in (0, 12] degrees')
    for name in ('hf_pyramid_slope', 'hf_pyramid_slope_inv'):
        if name in terrain_generator.sub_terrains:
            terrain_generator.sub_terrains[name].slope_range=(0.0,math.radians(slope_max_deg))

    terrain_scan=RayCastSensorCfg(name='terrain_scan',
        frame=ObjRef(type='body',name='base',entity='robot'),ray_alignment='yaw',
        pattern=GridPatternCfg(size=(1.6,1.0),resolution=.1),max_distance=5.,
        exclude_parent_body=True,debug_vis=play,
        viz=RayCastSensorCfg.VizCfg(show_normals=True))
    # The exact ZIP MJCF keeps the foot collision sphere on each calf body;
    # get_robot_cfg assigns stable per-body geom names, with the foot sphere
    # being geom index 5 after the four visual/collision leg geoms.
    feet=tuple(f'{leg}_calf_collision_5' for leg in ('FL','FR','RL','RR'))
    feet_contact=ContactSensorCfg(name='feet_ground_contact',
        primary=ContactMatch(mode='geom',pattern=feet,entity='robot'),
        secondary=ContactMatch(mode='body',pattern='terrain'),fields=('found','force'),
        reduce='netforce',num_slots=1,track_air_time=True,history_length=4)
    base_contact=ContactSensorCfg(name='base_ground_contact',
        primary=ContactMatch(mode='geom',pattern=r'base_collision_.*',entity='robot'),
        secondary=ContactMatch(mode='body',pattern='terrain'),fields=('found','force'),
        reduce='netforce',num_slots=1,history_length=4)
    nonfoot_contact=ContactSensorCfg(name='nonfoot_ground_touch',
        primary=ContactMatch(mode='geom',pattern=r'.*_collision_\d+$',
            exclude=feet,entity='robot'),
        secondary=ContactMatch(mode='body',pattern='terrain'),
        fields=('found','force'),reduce='none',num_slots=1,history_length=4)
    cfg.scene.sensors=(terrain_scan,feet_contact,base_contact,nonfoot_contact)

    cfg.actions={'recovery':RecoveryActionCfg(entity_name='robot',
        stand_hold_enabled=play or targeted_v2,
        staged_unfold_enabled=staged_unfold,
        predictive_all_joints=predictive_all_joints,
        predictive_general_trigger=predictive_general_trigger,
        predictive_calf_trigger=predictive_calf_trigger,
        predictive_general_reverse_offset=predictive_general_offset,
        predictive_calf_reverse_offset=predictive_calf_offset,
        predictive_brake_release_ratio=predictive_brake_release_ratio,
        predictive_latch_enabled=predictive_latch_enabled,
        velocity_brake_enabled=velocity_brake_enabled,
        velocity_brake_general_trigger=velocity_brake_general_trigger,
        velocity_brake_calf_trigger=velocity_brake_calf_trigger,
        safety_general_margin=safety_general_margin,
        safety_calf_margin=safety_calf_margin,
        initial_target_hold_steps=initial_target_hold_steps,
        global_impact_guard_enabled=global_impact_guard_enabled,
        global_impact_guard_trigger=global_impact_guard_trigger,
        global_impact_guard_margin=global_impact_guard_margin,
        global_impact_guard_cooldown_steps=global_impact_guard_cooldown_steps,
        global_impact_guard_cross_joint_scale=global_impact_guard_cross_joint_scale,
        global_impact_guard_target_error=global_impact_guard_target_error)}
    sim2real_profile=os.environ.get('FRNET_SIM2REAL_PROFILE','0') == '1'
    sim2real_noise=float(os.environ.get('FRNET_SIM2REAL_NOISE_LEVEL','1.0'))
    sim2real_sensor_delay=int(os.environ.get('FRNET_SIM2REAL_SENSOR_DELAY_STEPS','1'))
    if sim2real_noise < 0.0 or sim2real_noise > 4.0:
        raise ValueError('FRNET_SIM2REAL_NOISE_LEVEL must be in [0, 4]')
    if sim2real_sensor_delay < 0 or sim2real_sensor_delay > 5:
        raise ValueError('FRNET_SIM2REAL_SENSOR_DELAY_STEPS must be in [0, 5]')
    cfg.observations={
        'actor':ObservationGroupCfg(terms={'frnet_actor':ObservationTermCfg(
            func=FrnetActorSensorModel,params={
                'noise_enabled':(not play) or sim2real_profile,
                'noise_level':sim2real_noise if sim2real_profile else 1.,
                'sensor_delay_max_steps':sim2real_sensor_delay if sim2real_profile else (0 if play else 1)})},
            concatenate_terms=True,enable_corruption=False,nan_policy='error'),
        'critic':ObservationGroupCfg(terms={'frnet_critic':ObservationTermCfg(
            func=critic_observation)},concatenate_terms=True,
            enable_corruption=False,nan_policy='error'),
    }
    cfg.rewards={'frnet_total':RewardTermCfg(func=FrnetTotalReward,weight=1.0),
        'hardware_velocity_margin':RewardTermCfg(func=hardware_velocity_margin,weight=-100.0),
        'hardware_position_margin':RewardTermCfg(func=hardware_position_margin,weight=-300.0),
        'recovery_height_crossing':RewardTermCfg(func=recovery_height_crossing,weight=4.0),
        'recovery_supported_height':RewardTermCfg(func=recovery_supported_height,weight=2.0),
        'recovery_stable_stand':RewardTermCfg(func=recovery_stable_stand,weight=8.0),
        'recovery_success_bonus':RewardTermCfg(func=RecoverySuccessBonus,weight=5.0)}
    cfg.terminations={
        'time_out':TerminationTermCfg(func=mdp.time_out,time_out=True),
        'invalid_physics':TerminationTermCfg(func=invalid_physics),
        'source_rotation_timeout':TerminationTermCfg(func=SourceRotationTimeout),
    }
    if static_only:
        # Static-only recovery uses the full acceptance horizon and no external push.
        cfg.terminations.pop('source_rotation_timeout',None)
    if targeted_v2:
        # Train over the same 12 s horizon as acceptance instead of the source's
        # unconditional 401-step rotation timeout.
        cfg.terminations.pop('source_rotation_timeout',None)
    cfg.events={**parameter_events(),
        'reset_source':EventTermCfg(func=reset_source_state,mode='reset'),
        'push_robot':EventTermCfg(func=mdp.push_by_setting_velocity,mode='interval',
            interval_range_s=(15.,15.),params={'velocity_range':{
                'x':(-1.,1.),'y':(-1.,1.)}}),
    }
    if static_only:
        cfg.events.pop('push_robot',None)
        if not acceptance_aligned:
            cfg.rewards['recovery_stable_stand'].weight = 16.0
            cfg.rewards['recovery_success_bonus'].weight = 10.0
    cfg.commands={'twist':UniformVelocityCommandCfg(entity_name='robot',
        resampling_time_range=(8.,8.),rel_standing_envs=0.,heading_command=True,
        heading_control_stiffness=.5,debug_vis=play,
        ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(-1.,1.),lin_vel_y=(-1.,1.),
            ang_vel_z=(-1.,1.),heading=(-3.14,3.14)))}
    if targeted_v2 or acceptance_aligned or play:
        # Recovery is an onboard, stationary safety behavior.  A nonzero
        # locomotion command after standing makes the policy walk before the
        # one-second hold gate can finish and contradicts the acceptance task.
        cfg.commands={'twist':UniformVelocityCommandCfg(entity_name='robot',
            resampling_time_range=(8.,8.),rel_standing_envs=1.0,
            heading_command=False,heading_control_stiffness=.5,debug_vis=play,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.,0.),lin_vel_y=(0.,0.),ang_vel_z=(0.,0.)))}
    if os.environ.get("FRNET_RESIDUAL", "0") == "1":
        from .recovery_shaping import residual_stand_motion_penalty
        cfg.rewards["residual_stand_motion"] = RewardTermCfg(func=residual_stand_motion_penalty,weight=-0.5)
        if not play:
            cfg.episode_length_s=12.0
    cfg.curriculum={}
    cfg.metrics={f'reward_{name}':MetricsTermCfg(func=weighted_reward_metric,
        params={'name':name}) for name in SCALES}
    cfg.viewer.body_name='base'
    if play:
        cfg.events.pop('push_robot',None)
        cfg.events['settle_fallen']=EventTermCfg(func=settle_fallen,mode='reset')
        cfg.terminations.pop('source_rotation_timeout',None)
        cfg.episode_length_s=1e9
    return cfg

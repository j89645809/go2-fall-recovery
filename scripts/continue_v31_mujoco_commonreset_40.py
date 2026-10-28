from dataclasses import replace
import os, sys
ROOT='/home/zy/old_data/unitree_rl_mjlab';sys.path.insert(0,ROOT)
os.environ.update({
 'PYTHON_FRNET_TARGETED_RECOVERY':'1','PYTHON_FRNET_TARGETED_V2':'1','FRNET_SOURCE_ITERATION_OFFSET':'14469',
 'PYTHON_FRNET_SUCCESS_REPLAY_PATH':f'{ROOT}/run_logs/static_library_eval/failure_replay_v26_full2560.json',
 'PYTHON_FRNET_STATIC_LIBRARY_PATH':f'{ROOT}/run_logs/static_library_eval/mujoco_static_reset_v3_search256.json',
 'PYTHON_FRNET_STATIC_LIBRARY_TARGET_FRACTION':'1.0','PYTHON_FRNET_STATIC_JOINT_SELECT':'1','PYTHON_FRNET_STATIC_ACCEL_SELECT':'1',
 'FRNET_SIM2REAL_PROFILE':'1','FRNET_SIM2REAL_NOISE_LEVEL':'0.5','FRNET_SIM2REAL_SENSOR_DELAY_STEPS':'1',
 'FRNET_SIM2REAL_DRIFT_INITIAL':'0.005','FRNET_SIM2REAL_DRIFT_RANDOM_WALK':'0.0001',
 'PYTHON_FRNET_HARD_JOINT_LIMITS':'1','FRNET_JOINT_LIMIT_MARGIN':'0.20','FRNET_JOINT_LIMIT_DAMPRATIO':'2.0',
 'FRNET_JOINT_LIMIT_IMPEDANCE_MIN':'0.97','FRNET_JOINT_LIMIT_IMPEDANCE_MAX':'0.999',
 'FRNET_INITIAL_TARGET_HOLD_STEPS':'50','PYTHON_FRNET_PREDICTIVE_ALL_JOINTS':'1','PYTHON_FRNET_PREDICTIVE_LATCH':'1',
 'FRNET_PREDICTIVE_GENERAL_TRIGGER':'0.20','FRNET_PREDICTIVE_CALF_TRIGGER':'0.10','FRNET_PREDICTIVE_GENERAL_OFFSET':'0.012','FRNET_PREDICTIVE_CALF_OFFSET':'0.020','FRNET_PREDICTIVE_BRAKE_RELEASE_RATIO':'0.15',
 'PYTHON_FRNET_ACTUATOR_VELOCITY_BRAKE':'1','FRNET_ACTUATOR_GENERAL_BRAKE_TRIGGER':'0.10','FRNET_ACTUATOR_CALF_BRAKE_TRIGGER':'0.035',
 'PYTHON_FRNET_ACTUATOR_POSITION_BARRIER':'1','FRNET_ACTUATOR_POSITION_MARGIN':'0.30','FRNET_ACTUATOR_POSITION_BRAKE_GAIN':'2.5',
 'PYTHON_FRNET_ACTUATOR_POSITION_DAMPING':'1','FRNET_ACTUATOR_POSITION_DAMPING_MARGIN':'0.30','FRNET_ACTUATOR_POSITION_DAMPING_GAIN':'7.0',
 'FRNET_ACTUATOR_GENERAL_DAMPING':'3.0','FRNET_ACTUATOR_CALF_DAMPING':'7.5','PYTHON_FRNET_SUBSTEP_LOCAL_GUARD':'1',
 'FRNET_SUBSTEP_LOCAL_TRIGGER':'0.20','FRNET_SUBSTEP_LOCAL_CALF_SCALE':'0.15','FRNET_SUBSTEP_LOCAL_THIGH_SCALE':'0.65','FRNET_SUBSTEP_LOCAL_HIP_SCALE':'1.0',
 'FRNET_SUBSTEP_LOCAL_BRAKE':'1','FRNET_SUBSTEP_LOCAL_BRAKE_MIN_RATIO':'0.02','FRNET_SUBSTEP_LOCAL_BRAKE_GAIN':'0.35',
 'FRNET_GLOBAL_IMPACT_GUARD':'0','PYTHON_FRNET_GLOBAL_IMPACT_GUARD':'0','FRNET_GLOBAL_IMPACT_TRANSFER_GUARD':'0','FRNET_FOLDED_TRANSFER_GUARD':'0',
 'FRNET_ACTION_TARGET_RATE_SCALE':'0.05','FRNET_STARTUP_TARGET_RATE_SCALE':'0.10','FRNET_STARTUP_TARGET_RATE_STEPS':'50','FRNET_BRAKE_ACCEL_SCALE':'1.0','FRNET_CROSSBACKEND_RANDOMIZATION':'1',
})
import src.tasks.velocity.config.go2_recovery_aligned
from scripts.train import TrainConfig, launch_training
task='Unitree-Go2-Recovery-FRNet-Aligned';cfg=TrainConfig.from_task(task)
cfg.agent.max_iterations=40;cfg.agent.run_name='acceptance_v31_mujoco_commonreset_40';cfg.agent.resume=True;cfg.agent.load_run='2026-10-06_22-44-01_acceptance_v26_dynamic_receiver_60';cfg.agent.load_checkpoint='model_14469.pt';cfg.agent.logger='wandb';cfg.agent.wandb_project='mjlab';cfg.agent.upload_model=True;cfg.agent.wandb_tags=('FR-Net','Go2','sim2sim','ordinary-mujoco','common-static-reset','startup-local-safety','teacher-anchored')
cfg.agent.algorithm.teacher_checkpoint=f'{ROOT}/logs/rsl_rl/go2_recovery_frnet_aligned/2026-10-06_22-44-01_acceptance_v26_dynamic_receiver_60/model_14469.pt';cfg.agent.algorithm.teacher_pull=0.003;cfg.agent.algorithm.forced_learning_rate=1.5e-5;cfg.agent.algorithm.mirror_loss_weight=0.02
cfg.env.rewards['hardware_velocity_margin'].weight=-1000.0;cfg.env.rewards['hardware_position_margin'].weight=-1100.0;cfg.env.scene.num_envs=4096;cfg=replace(cfg,gpu_ids=[3]);launch_training(task,cfg)

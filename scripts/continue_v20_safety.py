from dataclasses import replace
import os,sys
ROOT="/home/zy/old_data/unitree_rl_mjlab"
sys.path.insert(0,ROOT)
os.environ.update({
 "PYTHON_FRNET_TARGETED_RECOVERY":"1","PYTHON_FRNET_TARGETED_V2":"1",
 "FRNET_SOURCE_ITERATION_OFFSET":"14305",
 "PYTHON_FRNET_HARD_JOINT_LIMITS":"1","FRNET_JOINT_LIMIT_MARGIN":"0.20",
 "FRNET_JOINT_LIMIT_DAMPRATIO":"2.0","FRNET_JOINT_LIMIT_IMPEDANCE_MIN":"0.97","FRNET_JOINT_LIMIT_IMPEDANCE_MAX":"0.999",
 "FRNET_INITIAL_TARGET_HOLD_STEPS":"25",
 "PYTHON_FRNET_ACTUATOR_VELOCITY_BRAKE":"1","FRNET_ACTUATOR_GENERAL_BRAKE_TRIGGER":"0.15","FRNET_ACTUATOR_CALF_BRAKE_TRIGGER":"0.05",
 "PYTHON_FRNET_ACTUATOR_POSITION_BARRIER":"1","FRNET_ACTUATOR_POSITION_MARGIN":"0.28","FRNET_ACTUATOR_POSITION_BRAKE_GAIN":"2.0",
 "PYTHON_FRNET_ACTUATOR_POSITION_DAMPING":"1","FRNET_ACTUATOR_POSITION_DAMPING_MARGIN":"0.28","FRNET_ACTUATOR_POSITION_DAMPING_GAIN":"6.0",
 "FRNET_ACTUATOR_GENERAL_DAMPING":"3.0","FRNET_ACTUATOR_CALF_DAMPING":"2.5",
 "PYTHON_FRNET_SUBSTEP_LOCAL_GUARD":"1","FRNET_SUBSTEP_LOCAL_TRIGGER":"0.30","FRNET_SUBSTEP_LOCAL_CALF_SCALE":"0.20","FRNET_SUBSTEP_LOCAL_THIGH_SCALE":"0.70","FRNET_SUBSTEP_LOCAL_HIP_SCALE":"1.0",
})
import src.tasks.velocity.config.go2_recovery_aligned
from scripts.train import TrainConfig,launch_training
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg

task="Unitree-Go2-Recovery-FRNet-Aligned"
cfg=TrainConfig.from_task(task)
cfg.agent.max_iterations=20
cfg.agent.run_name="acceptance_v20_local_leg_safety_20"
cfg.agent.resume=True
cfg.agent.load_run="2026-10-02_15-48-59_acceptance_v19_v15_aligned_safety_20"
cfg.agent.load_checkpoint="model_14305.pt"
cfg.agent.logger="wandb"
cfg.agent.wandb_project="mjlab"
cfg.agent.upload_model=True
cfg.agent.wandb_tags=("FR-Net","Go2","local-leg-safety","v20","teacher-anchored")
cfg.env.scene.num_envs=4096
cfg=replace(cfg,gpu_ids=[0])
launch_training(task,cfg)

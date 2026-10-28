"""PPO settings for baseline and acceptance-aligned continuation."""
import os
from dataclasses import dataclass
from mjlab.rl import RslRlModelCfg,RslRlOnPolicyRunnerCfg,RslRlPpoAlgorithmCfg


@dataclass
class TeacherPpoCfg(RslRlPpoAlgorithmCfg):
    class_name:str='src.tasks.velocity.config.go2_recovery_aligned.teacher_ppo:TeacherAnchoredPPO'
    teacher_checkpoint:str=os.environ.get('FRNET_TEACHER_CHECKPOINT', 'logs/rsl_rl/go2_recovery_frnet_aligned/2026-10-11_16-03-28_acceptance_v32_backend_static_reset_20/model_14475.pt')
    teacher_pull:float=float(os.environ.get('FRNET_TEACHER_PULL', '0.02'))
    forced_learning_rate:float=float(os.environ.get('FRNET_TEACHER_LR', '5e-6'))
    mirror_loss_weight:float=float(os.environ.get('FRNET_MIRROR_LOSS_WEIGHT', '0.01'))
    mirror_batch_size:int=8192


@dataclass
class RetentionPpoCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "src.tasks.velocity.config.go2_recovery_aligned.retention_ppo:RetentionPPO"


@dataclass
class MinibatchRetentionPpoCfg(RetentionPpoCfg):
    class_name: str = "src.tasks.velocity.config.go2_recovery_aligned.minibatch_retention_ppo:MinibatchRetentionPPO"


def frnet_aligned_ppo_cfg():
    targeted_v2=os.environ.get('PYTHON_FRNET_TARGETED_V2','0')=='1'
    algorithm_cls=TeacherPpoCfg if targeted_v2 else RslRlPpoAlgorithmCfg
    if os.environ.get("FRNET_RETENTION", "0") == "1":
        algorithm_cls=MinibatchRetentionPpoCfg if os.environ.get("FRNET_RETENTION_MINIBATCH", "0") == "1" else RetentionPpoCfg
    learning_rate=float(os.environ.get('FRNET_LEARNING_RATE', '1e-4' if targeted_v2 else '1e-3'))
    entropy_coef=float(os.environ.get('FRNET_ENTROPY_COEF', '0.01'))
    schedule=os.environ.get('FRNET_SCHEDULE', 'fixed' if targeted_v2 else 'adaptive')
    return RslRlOnPolicyRunnerCfg(
        seed=55,
        actor=RslRlModelCfg(class_name=('src.tasks.velocity.config.go2_recovery_aligned.residual_actor:ResidualActor' if os.environ.get('FRNET_RESIDUAL','0')=='1' else 'MLPModel'),hidden_dims=(512,256,128),activation='elu',
            obs_normalization=False,distribution_cfg={
                'class_name':'GaussianDistribution','init_std':1.0,'std_type':'scalar'}),
        critic=RslRlModelCfg(hidden_dims=(512,256,128),activation='elu',
            obs_normalization=False),
        algorithm=algorithm_cls(value_loss_coef=1.0,
            use_clipped_value_loss=True,clip_param=.2,entropy_coef=entropy_coef,
            num_learning_epochs=5,num_mini_batches=4,
            learning_rate=learning_rate,
            schedule=schedule,
            gamma=.99,lam=.95,desired_kl=.005 if targeted_v2 else .01,max_grad_norm=1.0),
        experiment_name='go2_recovery_frnet_aligned',save_interval=50 if targeted_v2 else 100,
        num_steps_per_env=24,max_iterations=4000,clip_actions=10.0)

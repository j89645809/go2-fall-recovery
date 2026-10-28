"""Registration entry point for the isolated aligned task."""
from mjlab.tasks.registry import register_mjlab_task
from .aligned_env_cfg import frnet_aligned_env_cfg
from .aligned_rl_cfg import frnet_aligned_ppo_cfg

register_mjlab_task(task_id='Unitree-Go2-Recovery-FRNet-Aligned',
    env_cfg=frnet_aligned_env_cfg(),play_env_cfg=frnet_aligned_env_cfg(play=True),
    rl_cfg=frnet_aligned_ppo_cfg())

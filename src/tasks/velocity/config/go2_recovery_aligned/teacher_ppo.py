"""PPO with a frozen model_13798 teacher anchor for safe continuation."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from rsl_rl.algorithms import PPO


_MIRROR_JOINT_INDEX=(3,4,5,0,1,2,9,10,11,6,7,8)
_HIP_INDEX=(0,3,6,9)


def mirror_action(action: torch.Tensor) -> torch.Tensor:
    """Mirror FL/FR and RL/RR actions across the robot sagittal plane."""
    mirrored=action[...,list(_MIRROR_JOINT_INDEX)].clone()
    mirrored[...,list(_HIP_INDEX)]*=-1.0
    return mirrored


def _mirror_joint_field(field: torch.Tensor) -> torch.Tensor:
    return mirror_action(field)


def mirror_actor_observation(obs: torch.Tensor) -> torch.Tensor:
    """Mirror the 45D actor observation using polar/axial vector physics."""
    if obs.shape[-1] != 45:
        raise ValueError(f"Expected 45D actor observation, got {tuple(obs.shape)}")
    mirrored=obs.clone()
    # Angular velocity is an axial vector: wx and wz change sign.
    mirrored[...,0]=-obs[...,0]
    mirrored[...,1]=obs[...,1]
    mirrored[...,2]=-obs[...,2]
    # Projected gravity and command velocity are polar vectors; yaw is axial.
    mirrored[...,3]=obs[...,3]
    mirrored[...,4]=-obs[...,4]
    mirrored[...,5]=obs[...,5]
    mirrored[...,6]=obs[...,6]
    mirrored[...,7]=-obs[...,7]
    mirrored[...,8]=-obs[...,8]
    mirrored[...,9:21]=_mirror_joint_field(obs[...,9:21])
    mirrored[...,21:33]=_mirror_joint_field(obs[...,21:33])
    mirrored[...,33:45]=mirror_action(obs[...,33:45])
    return mirrored


class TeacherAnchoredPPO(PPO):
    """PPO followed by a small proximal projection toward a frozen teacher.

    The projection is deliberately architecture-preserving, works with DDP,
    and prevents the targeted replay distribution from freely overwriting the
    original successful policy.  Action MSE is logged on every rollout.
    """

    def __init__(self,*args,teacher_checkpoint:str,teacher_pull:float=0.001,
                 forced_learning_rate:float=1.0e-4,
                 mirror_loss_weight:float=0.02,
                 mirror_batch_size:int=8192,**kwargs):
        super().__init__(*args,**kwargs)
        if not 0.0 <= teacher_pull < 1.0:
            raise ValueError("teacher_pull must be in [0,1)")
        path=Path(teacher_checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
        checkpoint=torch.load(path,weights_only=False,map_location=self.device)
        self.teacher=copy.deepcopy(self._raw_actor).to(self.device)
        self.teacher.load_state_dict(checkpoint['actor_state_dict'],strict=True)
        self.teacher.eval()
        for parameter in self.teacher.parameters(): parameter.requires_grad_(False)
        self.teacher_pull=float(teacher_pull)
        self.forced_learning_rate=float(forced_learning_rate)
        self.mirror_loss_weight=float(mirror_loss_weight)
        self.mirror_batch_size=int(mirror_batch_size)
        self.learning_rate=self.forced_learning_rate
        for group in self.optimizer.param_groups: group['lr']=self.learning_rate
        freeze_layers=int(__import__('os').environ.get('FRNET_FREEZE_ACTOR_LAYERS','2'))
        if freeze_layers > 0:
            frozen=0
            for module in self._raw_actor.mlp:
                if hasattr(module, 'weight') and hasattr(module, 'bias') and frozen < freeze_layers:
                    for parameter in module.parameters(): parameter.requires_grad_(False)
                    frozen += 1

    def update(self):
        # PPO clears its rollout storage at the end of update(). Capture the
        # observations before that lifecycle transition for the teacher metric.
        rollout_obs=self.storage.observations.flatten(0,1).clone()
        losses=super().update()
        mirror_loss=rollout_obs.new_zeros(())
        if self.mirror_loss_weight > 0.0:
            if len(rollout_obs) > self.mirror_batch_size:
                indices=torch.randperm(len(rollout_obs),device=rollout_obs.device)[:self.mirror_batch_size]
                symmetry_obs=rollout_obs[indices]
            else:
                symmetry_obs=rollout_obs
            mirrored_obs=symmetry_obs.clone()
            mirrored_obs["actor"]=mirror_actor_observation(symmetry_obs["actor"])
            with torch.no_grad():
                expected=mirror_action(self._raw_actor(symmetry_obs))
            predicted=self._raw_actor(mirrored_obs)
            predicted=predicted.clamp(-10.0,10.0)/10.0
            expected=expected.clamp(-10.0,10.0)/10.0
            mirror_loss=(predicted-expected).square().mean()
            self.optimizer.zero_grad()
            (self.mirror_loss_weight*mirror_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                self._raw_actor.parameters(),self.max_grad_norm
            )
            self.optimizer.step()
        with torch.no_grad():
            teacher_action=self.teacher(rollout_obs)
            student_action=self._raw_actor(rollout_obs)
            action_mse=(student_action-teacher_action).square().mean()
            for student,teacher in zip(self._raw_actor.parameters(),self.teacher.parameters(),strict=True):
                student.lerp_(teacher,self.teacher_pull)
        losses['teacher_action_mse']=float(action_mse.item())
        losses['teacher_pull']=self.teacher_pull
        losses['mirror_action_mse']=float(mirror_loss.item())
        losses['mirror_loss_weight']=self.mirror_loss_weight
        return losses

    def load(self,loaded_dict,load_cfg,strict):
        # Continuation loads actor/critic and iteration but never the high-LR
        # optimizer state from the source run.
        if load_cfg is None:
            load_cfg={'actor':True,'critic':True,'optimizer':False,
                      'iteration':True,'rnd':False}
        result=super().load(loaded_dict,load_cfg,strict)
        self.learning_rate=self.forced_learning_rate
        for group in self.optimizer.param_groups: group['lr']=self.learning_rate
        freeze_layers=int(__import__('os').environ.get('FRNET_FREEZE_ACTOR_LAYERS','2'))
        if freeze_layers > 0:
            frozen=0
            for module in self._raw_actor.mlp:
                if hasattr(module, 'weight') and hasattr(module, 'bias') and frozen < freeze_layers:
                    for parameter in module.parameters(): parameter.requires_grad_(False)
                    frozen += 1
        return result

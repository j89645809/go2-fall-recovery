"""PPO with explicit residual suppression on frozen-teacher success trajectories."""
import os
import torch
from rsl_rl.algorithms import PPO

def success_residual_loss(actor, observations):
    return torch.tanh(actor.mlp(observations)).square().mean()


class RetentionPPO(PPO):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        data=torch.load(os.environ["FRNET_SUCCESS_TRAJECTORIES"],map_location="cpu",weights_only=False)
        if data.get("protocol") != "strict_static_repeated_v3":
            raise ValueError("Trajectory protocol mismatch")
        self.success_obs=data["observations"].to(self.device)
        if self.success_obs.ndim!=2 or self.success_obs.shape[1]!=45 or not len(self.success_obs):
            raise ValueError("Invalid success trajectory observations")
        if not torch.isfinite(self.success_obs).all():
            raise ValueError("Non-finite trajectory observations")
        self.retention_weight=1.0

    def update(self):
        losses=super().update()
        ids=torch.randint(len(self.success_obs),(min(4096,len(self.success_obs)),),device=self.device)
        latent=self.success_obs[ids]
        # Actor normalization is Identity. Teacher is frozen; penalize only residual.
        loss=success_residual_loss(self._raw_actor,latent)
        self.optimizer.zero_grad()
        (self.retention_weight*loss).backward()
        torch.nn.utils.clip_grad_norm_(self._raw_actor.parameters(),self.max_grad_norm)
        self.optimizer.step()
        losses["success_trajectory_retention"]=loss.item()
        return losses

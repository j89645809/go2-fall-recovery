"""Frozen baseline plus a tanh-bounded residual in policy action units."""
import copy
import os
import torch
from rsl_rl.models import MLPModel

class ResidualActor(MLPModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path=os.environ["FRNET_RESIDUAL_TEACHER"]
        state=torch.load(path,map_location="cpu",weights_only=False)["actor_state_dict"]
        self.load_state_dict(state,strict=True)
        self.teacher_mlp=copy.deepcopy(self.mlp)
        self.teacher_mlp.requires_grad_(False)
        self.register_buffer("residual_limit",torch.tensor(0.1))
        # Same features initially, zero final output guarantees baseline equivalence.
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()
            self.distribution.std_param.fill_(0.2)
        self.distribution.std_param.requires_grad_(False)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        latent=self.get_latent(obs,masks,hidden_state)
        with torch.no_grad():
            self._teacher_action=self.teacher_mlp(latent).clamp(-10.,10.)
        raw=self.mlp(latent)
        if stochastic_output:
            self.distribution.update(raw)
            raw=self.distribution.sample()
        return self._teacher_action+self.residual_limit*torch.tanh(raw)

    def get_output_log_prob(self, outputs):
        y=((outputs-self._teacher_action)/self.residual_limit).clamp(-0.999999,0.999999)
        raw=torch.atanh(y)
        jac=(torch.log(self.residual_limit)+torch.log1p(-y.square())).sum(-1)
        return self.distribution.log_prob(raw)-jac

    def as_jit(self):
        raise NotImplementedError("Export must include frozen teacher and bounded residual")

    def as_onnx(self, verbose=False):
        raise NotImplementedError("Export must include frozen teacher and bounded residual")

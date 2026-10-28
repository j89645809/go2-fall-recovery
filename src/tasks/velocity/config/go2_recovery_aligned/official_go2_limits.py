"""Current Unitree Go2 joint limits in FR-Net's FL/FR/RL/RR ordering.

Position and actuator limits are taken from Unitree's official Go2 MuJoCo
model.  Velocity limits are recorded from the official Go2 URDF; mjlab's
IdealPdActuatorCfg has no hard velocity-limit field, so they are an explicit
validation/safety contract rather than a simulator clamp.
"""
import torch

JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf")
)

POSITION_LOWER = (
    -1.0472, -1.5708, -2.7227,
    -1.0472, -1.5708, -2.7227,
    -1.0472, -0.5236, -2.7227,
    -1.0472, -0.5236, -2.7227,
)
POSITION_UPPER = (
    1.0472, 3.4907, -0.83776,
    1.0472, 3.4907, -0.83776,
    1.0472, 4.5379, -0.83776,
    1.0472, 4.5379, -0.83776,
)
OFFICIAL_EFFORT_LIMIT = (23.7, 23.7, 45.43) * 4
# FR-Net's bundled Go2 URDF predates the current Unitree calf rating. Keeping
# its 35.55 Nm saturation makes the training controller source-exact while
# remaining inside the current official hardware envelope.
FRNET_EFFORT_LIMIT = (23.7, 23.7, 35.55) * 4
VELOCITY_LIMIT = (30.1, 30.1, 15.70) * 4


def position_limits(*, device=None, dtype=None):
    """Return lower/upper tensors without silently changing precision."""
    return (
        torch.tensor(POSITION_LOWER, device=device, dtype=dtype),
        torch.tensor(POSITION_UPPER, device=device, dtype=dtype),
    )


def clamp_joint_positions(q):
    """Project a batch of 12-DoF Go2 positions into official hard limits."""
    lower, upper = position_limits(device=q.device, dtype=q.dtype)
    return torch.maximum(torch.minimum(q, upper), lower)

# FR-Net aligned Go2 task

Task id: `Unitree-Go2-Recovery-FRNet-Aligned`

Server package:
`src/tasks/velocity/config/go2_recovery_aligned/`

## Locked contracts

- Go2 URDF-derived robot, official Unitree joint position stops.
- FR-Net joint order: FL, FR, RL, RR; hip, thigh, calf within each leg.
- Actor: 45 values in source order and scale.
- Critic: 260 values in source order; live wrench, friction, 19 masses and
  187 terrain-height values; no missing-field fallback or zero padding.
- Action: clip 10, scale 0.25, per-environment 0--5 policy-step delay.
- Controller: Kp 30, Kd 0.8, physics 200 Hz, policy 50 Hz.
- Torque saturation: 23.7/23.7/35.55 Nm per leg (FR-Net source); this is within
  the current official Go2 23.7/23.7/45.43 Nm hardware envelope.
- Rewards: all 16 source formulas and weights, one dt factor, positive clipping
  after summation. Every weighted component is logged separately.
- Reset: source independent joint factors and uniform unit quaternion; invalid
  angles are projected into official Go2 position limits.
- Randomization: source Gaussian actor noise, 0--20 ms proprioceptive latency,
  action latency, friction buckets, base/link mass with inertia, base COM and
  shared Kp/Kd factors.

## Verified gates

- Integrated 8-environment, 32-step state/action/reward test: passed.
- Source differential comparison for 16 reward formulas/weights: passed.
- Reward temporal lifecycle and 187-point source sampler unit tests: passed.
- 128-environment PPO, two updates: passed; W&B `aia8bapk`.
- 4096-environment PPO, one update / 98,304 samples: passed locally with W&B
  disabled; directory `logs/rsl_rl/go2_recovery_frnet_aligned/2026-09-11_02-59-22_fullscale_gate`.

## Deliberate qualifications

The integrated 187-value critic channel uses MuJoCo ray intersections over the
same grid, because the mjlab terrain is collision geometry rather than Isaac's
single integer height array. It is a live physical mapping, not a placeholder,
but it is not bit-identical to Isaac's three-neighbor minimum.

No stationary and dynamic logs from the physical Go2 were found on the server.
The noise amplitudes therefore reproduce FR-Net's simulation assumptions and
the latency range is a configurable robustness model. Do not label these values
as hardware-calibrated until real Go2 logs are collected and fitted.

The full 4000-update training is not running. The previous model_3999 and its
Viser process were not modified.

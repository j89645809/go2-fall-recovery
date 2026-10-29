# go2-fall-recovery

Unitree Go2 fall recovery from static fallen states (FRNet-aligned policy, mjlab/MuJoCo).

Project code, acceptance criteria, evaluation and training commands, the included checkpoints and the result reports are described in [PUBLIC_RELEASE.md](PUBLIC_RELEASE.md). 中文说明见 [README_zh.md](README_zh.md)。

| Path | Content |
|---|---|
| `src/tasks/velocity/config/go2_recovery_aligned/` | Recovery task `Unitree-Go2-Recovery-FRNet-Aligned` |
| `src/tasks/velocity/config/go2_recovery/` | Earlier recovery task variants |
| `src/tasks/velocity/config/go2/` | Go2 base environment the recovery tasks inherit from |
| `frnet_aligned_assets/` | Go2 MJCF and meshes used by the recovery task |
| `scripts/` | Training, evaluation, acceptance and audit scripts |
| `checkpoints/`, `data/` | Baseline policy and data needed to reproduce training |
| `public_results/` | Aggregate results of the reported evaluations |
| `deploy/robots/go2/`, `simulate/` | Go2 controller and unitree_mujoco simulator for sim-to-sim checks |

## Installation

See [doc/setup_en.md](doc/setup_en.md). Then run commands from the repository root:

```bash
export PYTHONPATH=. MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0
```

## Quick start

Evaluate the baseline policy (12 s horizon, as in the reported numbers):

```bash
python scripts/paired_static_acceptance.py \
  --task Unitree-Go2-Recovery-FRNet-Aligned \
  --checkpoint checkpoints/model_14475.pt \
  --device cuda:0 --seed 2345 --horizon 600 \
  --success-profile tiered \
  --output results/tiered_2345.json
```

Training and residual / retention fine-tuning commands are in [PUBLIC_RELEASE.md](PUBLIC_RELEASE.md).

## Sim-to-sim check (optional)

Build the simulator and the Go2 controller (requires [cyclonedds](https://github.com/eclipse-cyclonedds/cyclonedds.git) and [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2.git)):

```bash
cd simulate && mkdir build && cd build && cmake .. && make -j8
cd ../../deploy/robots/go2 && mkdir build && cd build && cmake .. && make
```

Run `./simulate/build/unitree_mujoco`, then `./go2_ctrl --network=lo` from `deploy/robots/go2/build`.

## License and acknowledgements

This repository is derived from [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) (Apache-2.0, see [LICENCE](LICENCE)); components for other robots and motion imitation were removed. Third-party licenses are in `doc/license/`.

Built on [mjlab](https://github.com/mujocolab/mjlab.git), [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git), [mujoco_warp](https://github.com/google-deepmind/mujoco_warp.git) and [mujoco](https://github.com/google-deepmind/mujoco.git).

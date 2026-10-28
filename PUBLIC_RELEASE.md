# Go2 Fall Recovery

Public release of the static-fallen-state Go2 fall-recovery project. The project includes aligned FRNet policy code, static reset configuration, recovery training and acceptance evaluation scripts, the baseline checkpoint, and selected aggregate reports.

## Scope

- Static fallen initialization for recovery training and evaluation (zero initial root and joint velocity).
- FRNet-aligned recovery environment and action/reset components (`src/tasks/velocity/config/go2_recovery_aligned/`, task id `Unitree-Go2-Recovery-FRNet-Aligned`).
- PPO, residual, success-retention and acceptance evaluation tooling.
- Aggregate acceptance and handoff audit reports in `public_results/`.

Raw training logs, videos, caches and temporary backups are excluded.

## Repository layout

| Path | Content |
|---|---|
| `src/tasks/velocity/config/go2_recovery_aligned/` | Recovery task: env/PPO config, static reset, residual actor, retention PPO, acceptance targets |
| `frnet_aligned_assets/` | Go2 MJCF (`real_go2_head.xml`) and meshes used by the recovery task |
| `scripts/train_aligned.py` | Training entry point |
| `scripts/paired_static_acceptance.py` | Static-fallen-state acceptance evaluation (strict / tiered / stable criteria) |
| `scripts/formal_recovery_acceptance.py` | Full formal acceptance suite |
| `scripts/acceptance_criteria.py`, `scripts/acceptance_physics.py` | Success criteria and per-environment friction assignment used by evaluation |
| `scripts/audit_relaxed_success.py`, `scripts/audit_stand_handoff.py` | Video/telemetry audits of relaxed successes and stand handoff |
| `checkpoints/` | Baseline policy and data needed to reproduce training (see below) |
| `data/` | Success replay states used by residual / retention training |
| `public_results/` | Aggregate results of the reported evaluations |

## Included checkpoints and data

| File | Purpose |
|---|---|
| `checkpoints/model_14475.pt` | Baseline static-fallen-state recovery policy (iteration 14475); also the default frozen teacher |
| `checkpoints/residual_baseline_init_model_14475.pt` | Initialization checkpoint for residual / retention fine-tuning (resume source) |
| `checkpoints/success_trajectories.pt` | Baseline success trajectories for the success-retention loss (`FRNET_SUCCESS_TRAJECTORIES`) |
| `data/residual_baseline_success_replay.json` | Success replay reset states (`PYTHON_FRNET_SUCCESS_REPLAY_PATH`) |

Under the default tiered (relaxed) success criterion `model_14475.pt` reaches about 91–92% success in simulation (1664 cases per seed, 12 s horizon); under the original strict criterion about 61%. These are simulation results only, not real-robot success rates.

## Environment

Python 3.11 with `mjlab`, `rsl_rl` and PyTorch (CUDA), as required by [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab). Run all commands from the repository root:

```bash
export PYTHONPATH=. MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0
```

## Evaluate the baseline

The reported numbers use a 12 s horizon. The script default is 1000 policy steps (20 s), so pass `--horizon 600` to reproduce them:

```bash
python scripts/paired_static_acceptance.py \
  --task Unitree-Go2-Recovery-FRNet-Aligned \
  --checkpoint checkpoints/model_14475.pt \
  --device cuda:0 --seed 2345 --horizon 600 \
  --success-profile tiered \
  --output results/tiered_2345.json
```

`--success-profile` selects which criterion is reported as `success_rate` (`strict`, `tiered` or `stable`); the output always contains all three rates. Every case is checked for a valid static fallen setup with zero initial velocity, and evaluation stops if any setup is invalid. Reference run of the published repository (seed 2345, `--horizon 600`): tiered 91.6%, strict 60.6%, stable 37.9%. Repeated runs vary by roughly ±1 percentage point because the simulation is not bit-exact between runs.

## Residual / success-retention fine-tuning

Training resumes from a run directory under `logs/rsl_rl/go2_recovery_frnet_aligned/`, so place the initialization checkpoint there first:

```bash
mkdir -p logs/rsl_rl/go2_recovery_frnet_aligned/residual_baseline_init
cp checkpoints/residual_baseline_init_model_14475.pt \
   logs/rsl_rl/go2_recovery_frnet_aligned/residual_baseline_init/model_14475.pt
```

Configuration used for the per-minibatch retention experiment (frozen baseline teacher, ±0.1 residual, learning rate 1e-5):

```bash
export FRNET_RESIDUAL=1
export FRNET_RESIDUAL_TEACHER=checkpoints/model_14475.pt
export FRNET_RETENTION=1 FRNET_RETENTION_MINIBATCH=1
export FRNET_SUCCESS_TRAJECTORIES=checkpoints/success_trajectories.pt
export PYTHON_FRNET_SUCCESS_REPLAY_PATH=data/residual_baseline_success_replay.json
export PYTHON_FRNET_STATIC_ONLY=1 PYTHON_FRNET_ACCEPTANCE_ALIGNED=1 PYTHON_FRNET_TARGETED_RECOVERY=1
export PYTHON_FRNET_TARGETED_V2=0 FRNET_REPLAY_FRACTION=0.5
export FRNET_LEARNING_RATE=1e-5 FRNET_OVERRIDE_LR=1e-5 FRNET_ENTROPY_COEF=0 FRNET_SCHEDULE=fixed

python scripts/train_aligned.py Unitree-Go2-Recovery-FRNet-Aligned \
  --gpu-ids '[0]' --env.scene.num-envs 1024 \
  --agent.resume True --agent.load-run residual_baseline_init --agent.load-checkpoint model_14475.pt \
  --agent.max-iterations 25 --agent.save-interval 25 \
  --agent.run-name retention_minibatch_25 --agent.logger tensorboard
```

- Plain residual training without the retention loss: set `FRNET_RETENTION=0`.
- Residual checkpoints must be evaluated with `FRNET_RESIDUAL=1` and `FRNET_RESIDUAL_TEACHER` set as above.
- None of the fine-tuned candidates showed a stable improvement over `model_14475.pt` in repeated evaluation (see `public_results/`), so the baseline remains the reference policy.

## Known limitations

- `test_impact_guard.py` / `tests_test_impact_guard.py` and `src/tasks/velocity/config/go2_recovery/test_baseline_*.py` are outdated relative to the current code and fail; they are kept for reference only.
- The stand handoff controller usually engages after recovery but does not track the configured stand joint targets precisely (median max joint error about 0.41 rad); see `public_results/run_logs/handoff_audit_20261020/summary.json`.
- This repository modifies the shared Go2 assets in `src/assets/robots/unitree_go2/` (added head mass and changed default pose/gains). The recovery task itself uses `frnet_aligned_assets/real_go2_head.xml`.

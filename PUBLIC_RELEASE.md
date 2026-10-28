# Go2 Fall Recovery

Public release of the static-fallen-state Go2 fall-recovery project. The project includes aligned FRNet policy code, static reset configuration, recovery training and acceptance evaluation scripts, and selected aggregate reports.

## Scope

- Static fallen initialization for recovery training and evaluation.
- FRNet-aligned recovery environment and action/reset components.
- PPO, retention, residual and acceptance evaluation tooling.
- Aggregate acceptance and handoff audit reports in `public_results/`.

Large checkpoints, raw videos, caches, temporary backups, and machine-specific logs are excluded from this repository. Use the paths and run metadata in the reports to reproduce or select checkpoints from the training server.

## Included checkpoint

- `checkpoints/model_14475.pt`: baseline static-fallen-state recovery policy (iteration 14475). Under the default tiered (relaxed) success criterion it reaches about 90.9% success in simulation (two seeds, 1664 cases each); under the original strict criterion about 61%. These are simulation results only, not real-robot success rates.

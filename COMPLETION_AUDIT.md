# Completion audit

## Requested scope

Adapt `bu-air-lab/guide_dog` to Unitree Go2 and mjlab, using the software already
present on the supplied server and preferring code changes over installation.

## Requirement evidence

| Requirement | Implementation | Authoritative evidence | Result |
|---|---|---|---|
| Unitree Go2 | Loads `Unitree-Go2-Flat`; forces target Go2 `base_link` | `guide_cnn_cfg.py`; server compatibility report | Passed |
| mjlab rather than Isaac Gym | Uses mjlab environment, managers, observations, events, and RSL-RL wrapper | All runtime entry points; server runtime report | Passed |
| Original guide_dog estimator concept | 48-D layout, velocity MLP, 25-step temporal force CNN, asymmetric truth labels | `guide_cnn_cfg.py`, `force_estimator_mjlab.py`, `guide_estimator_wrapper.py` | Passed |
| Trainable pipeline | PPO, both estimators, checkpoint resume, separate matched estimator save | `train_guide_cnn.py`; `evidence/final_v4/verification.json` | Passed |
| Reproducible evaluation | Explicit environment seed, frozen estimators, finite checks, failure/timeout split, history-reset assertion | `eval_guide_cnn.py`; seed 812–814 reports | Passed |
| Testing/play entry point | Loads the matched policy and estimator pair in Viser, native, or finite headless mode | `play_guide_cnn.py`; server headless playback smoke test | Passed |
| Estimator effectiveness | Force estimate compared with zero-force prediction on three fixed seeds | Mean RMSE 14.1051 N versus 16.2681 N baseline | Passed |
| Closed-loop robustness evidence | 32 environments × 1,500 steps × three seeds | One failure termination and 96 normal timeouts; mean return 61.9047 | Passed for tested matrix |
| Full simulation inference export | Velocity estimator, force CNN, actor normalizer, actor, explicit history/reset | `models/controller.onnx`; `evidence/export_final/verification.json` | Passed |
| Export parity | ONNX checker, dynamic batches, reset test, live closed-loop comparison | Maximum action error 2.15e-6 over 150 steps | Passed |
| Existing-server adaptation | Fail-fast runtime checker; local TensorBoard; no new package required | `check_server.py`; `evidence/server_compatibility.json` | Passed |
| Deliverable integrity | Matched model files and SHA-256 manifest | `MANIFEST.json`; local hash verification | Passed |

## Final scope boundary

This completes the requested Go2/mjlab adaptation. It does not claim a
line-for-line reproduction of the original A1 Isaac Gym dynamics and rewards.
The original PyBullet/A1 validation is replaced by Go2/mjlab evaluation. The
ONNX graph is a simulation controller; raw actions still require the exact Go2
mjlab joint order, action scale, offsets, limits, and actuator model. Real-robot
deployment is outside this reproduction.

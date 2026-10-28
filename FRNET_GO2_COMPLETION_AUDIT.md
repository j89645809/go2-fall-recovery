# FR-Net Go2 accurate-model completion audit

Date: 2026-09-13

| Requirement | Evidence | Status |
|---|---|---|
| Real Go2 MJCF/visual meshes | `frnet_aligned_assets/real_go2_head.xml`; 16 meshes, 57 geoms; source `unitree_go2.zip` head model | PASS |
| Mass/inertia/axes/limits/torque | `run_logs/go2_head_model_spec_audit.json`; asset test | PASS |
| 50 Hz control and delays/noise/randomization | `scripts/test_aligned_integrated.py` | PASS |
| Direct `model_3999.pt` warm-start | W&B run `2myqg07w`; final `model_5498.pt` | PASS |
| Four pose x four terrain safety acceptance | `run_logs/go2_head_model5498_full_acceptance.json` | FAIL: 11/16, slope 0/4, hardware gate false |
| Support-stability follow-up | W&B `vfyrfoy8`; `model_6497.pt`; `go2_head_model6497_full_acceptance.json` | FAIL: 11/16, slope 0/4, hardware gate false |

Current Viser uses `model_10794.pt` on `real_go2_head.xml` at port 8082.
The model and test artifacts are retained; no sim2real safety claim is made.

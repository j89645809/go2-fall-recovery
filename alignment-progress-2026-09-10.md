# FR-Net alignment: incomplete, live evidence

Source repository: remote `/home/zy/FR-Net`, clean git status at commit
`85c872d9bef0a74716d8d3fb1e89606c8a42101e`.

## Runtime inspection (not inferred from configuration)

`scripts/probe_frnet_mapping.py` instantiated one mjlab environment on physical
GPU 1 (`CUDA_VISIBLE_DEVICES=1`, logical `cuda:0`). Finished successfully.
Physics timestep .005 s, policy timestep .02 s.
Actual joint order is FL, FR, RL, RR, each hip/thigh/calf.
Actual robot body count is 15: base_link, Head_lower, xiaoli_head, and 12 leg
segments. Feet are merged, not separate bodies. Global body indices are 2..16.
Actual model mass tensor is [1,17], including world and terrain: those entries
MUST NOT be treated as robot masses. Base mass is 7.505 kg; extra head masses
are .001 and 1.6 kg. Existing robot therefore cannot supply 18 independent
non-base masses. Need a separate source-derived asset and verified body mapping;
do not pad or split merged masses arbitrarily. Preserve existing custom asset.

## Source discrepancies requiring explicit handling

The source `rand_push_force` and `rand_push_torque` fields are initialized to zero
and never updated anywhere under legged_gym/envs. `_push_robots` sets base XY
velocity, not force/torque. Therefore earlier statements that all five zero
entries are migration omissions were incorrect. Exact source compatibility and
physically meaningful live force channels must be documented separately; do not
silently substitute an impulse velocity for a force.

The original joint reset independently multiplies [0,1.5,-2.4] for each leg by
U(.5,1.5), zeroes joint velocity, and does not clamp joint limits. Hips remain
zero; this is not an arbitrary full-joint-range distribution.

## Implemented building blocks, not yet integrated into training

- baseline_protocol.py: source reset, Gaussian actor noise, clipped delayed
  actions, PD calculation; four unit tests pass remotely.
- physical_observations.py: named live body-mass access using global indices;
  shared collision-friction access; rejects missing/duplicated mass mappings and
  heterogeneous friction masquerading as a scalar. Three unit tests pass.
- Existing baseline_math.py and baseline_observations.py still need full adapter
  integration and differential testing. Pure-function tests are not end-to-end
  evidence for the six requested requirements.

## Remaining

### Inertia and actor noise follow-up

Replaced mass-only DR with mass/inertia coupled scaling about unchanged COM:
I_new = I_default * mass_new / mass_default. Source mass ranges unchanged.
Live tensor assertions verify positive principal moments and exact ratios;
80-substep PD test still passes. This is physically consistent fixed-shape
density scaling, not proven equivalent to PhysX collision-based recomputeInertia.
The older mass-only status below is superseded by this change.
2048 live actor calls at fixed states verify Gaussian mean/std within 4% of
configured amplitude; commands/actions remain exactly noise-free.
`/tmp/frnet_noise_live.log` records success. Real hardware noise calibration
has not been provided and is not claimed. Overall training integration remains
incomplete; old model and viewer unchanged.

### Applied PD verification

Extended live test runs 20 policy actions / 80 physical substeps. For each step,
expected torque is computed independently from the live joint state, actual
randomized gains and delayed target. Actual `sim.data.ctrl` for every actuator
matches the clipped PD formula at rtol/atol 1e-5. Joint positions and root
velocities remain finite. `/tmp/frnet_pd_live.log` records success. This is a
short physics test, not a policy training run or a sim-to-real safety claim.
`aligned_actions.py` now defaults to source history compatibility: source never
updates last_last_actions, so second-order smoothness is zero. Opt-out is explicit.
`aligned_reward_state.py` groups all 16 terms and applies positive clipping after
the weighted dt-scaled sum; synthetic lifecycle test passes. Runtime integration
is still outstanding.

### Height sampling semantics

`aligned_heights.py` implements 187 points in source meshgrid order, quaternion
x/y-zeroing normalization (not Euler-yaw extraction), truncation-toward-zero
heightfield indexing, edge clamping and minimum of three neighboring samples.
`test_height_source_equivalence.py` directly executes the original `_get_heights`
AST with an independent quaternion-vector rotation implementation. Random tilted
poses, negative/out-of-grid locations, two border sizes and nonconstant grids
match exactly. First test run had a missing field in the test fixture, corrected;
subsequent remote run passes. Not yet connected to generated collision terrain:
the original heightfield must be shared with the terrain builder, not replaced
by arbitrary ray hits. This closes a formula gap, not the integration requirement.

### Parameter randomization and critic adapter

`aligned_randomization.py` now uses startup mass ranges (base additive +/-3 kg,
18 named non-base bodies scaled .5..2), 64 friction buckets in .5..1.25, and one
Kp/Kd factor per environment (.8..1.2), shared across all joints. Live two-env
tests verify actual model values, distinct masses, gain sharing, bucket membership
and mass/friction access. Mass-only randomization currently leaves inertia
unchanged, unlike source recomputeInertia=True; not final fidelity.
`aligned_critic.py` requires explicit privileged state and assembles 260 fields
with live physical masses and shared friction. `/tmp/frnet_critic_live.log`
records successful mapping tests with nonzero diagnostic wrench and synthetic
heights: these are tests, not live terrain/wrench implementation.
Source height measurements take the minimum of three neighboring heightfield
samples, not exact ray intersection; current Viser scanner is not equivalent.
User asked asynchronously whether force/torque should be live physical channels
(documented source deviation) or preserve source constant-zero behavior.

### Direct reward source comparison

`test_reward_source_equivalence.py` extracts the original reward function ASTs
from LeggedRobot, GO2 and GO2RecoveryEnv (without loading Isaac Gym), executes
them against the same synthetic state as the port, and compares all 16 outputs.
All 16 formulas and all 16 config weights passed on the remote Python runtime.
The previously missing acceleration, soft joint limits, force-threshold collision
and foot-stumble formulas are now in baseline_math.py. Feet-stumble uses GO2's
factor 4 (not the differently named base-class stumble implementation).
This does not verify which simulation timestep feeds contacts, previous velocity,
filtered foot contacts, or the PPO iteration schedule: integrated rollout work
remains necessary. No training config or model was replaced by this test.

### Live adapters

`aligned_reset.py`, `aligned_actions.py`, `aligned_actor.py` added. The remote
`/tmp/frnet_actor_live.log` records successful two-environment checks: ten partial
resets; joint ranges/zero velocity/root height; clipped current action history;
0/2-step delay selection; .25 target scaling; all six actor slices read from
live robot state with explicit known commands. Actor is checked without noise
in this test; command manager is replaced only during the assembly check.
These adapters are not yet a registered complete training environment. PD torque
execution, randomization, noisy observations and all reward lifecycle details
still require integrated rollout tests. Old strict training config unchanged.

### Source asset progress

`work/build_frnet_asset.py` now builds a separate collision-only MuJoCo asset
from the reference URDF; remote output is `frnet_aligned_assets/frnet_go2.xml`.
The original retained bodies are base, Head_upper, Head_lower, then
FL/FR/RL/RR hip/thigh/calf/foot (19 total). Additional fixed helper frames
have zero mass and remain in the MuJoCo tree without adding mass observations.
Mapping records declaration order; Isaac runtime body order still unverified.
`aligned_asset.py` creates an independent mjlab Entity with explicit ideal PD,
Kp=30, Kd=.8, hip/thigh torque limit=23.7 and calf=35.55 Nm, zero armature.
The robot remains the FR-Net Go2 URDF-derived body. All 12 position limits now
match Unitree's current official Go2 model. FR-Net's raw joint-reset sampler is
preserved for source comparison, while the runtime adapter clamps invalid samples
to those official stops. Official URDF velocity limits (30.1 rad/s for hip/thigh,
15.70 rad/s for calf) and the current 45.43 Nm calf hardware envelope are
recorded as a validation/deployment contract because
mjlab's ideal-PD actuator has no hard velocity-limit field.
`test_aligned_asset.py` passes remote compilation, 12 actuator, nq/nv=19/18,
retained-body presence and source base/foot mass checks. It is not a rollout or
full inertial/contact-equivalence test. Existing custom Go2 and Viser unchanged.

Create isolated source-compatible robot/config and verify original rigid body
layout. Integrate actor/critic assembly, PD delay, all rewards including temporal
state and PPO iteration target schedule, source reset, and domain randomization.
Run live randomized-value, ordering, action/torque, reset and reward comparisons,
then short PPO smoke test. Do not overwrite model_3999 or claim training/reproduction
completion. Source termination OR condition is always true and must be explicitly
audited, not silently repaired. Physical sensor calibration remains unavailable;
source noise settings are simulation assumptions, not measured real-robot noise.

## Integrated task gate (2026-09-11)

The independent task `Unitree-Go2-Recovery-FRNet-Aligned` is now registered on
the server. It combines the aligned Go2 asset, 45D actor, 260D critic, delayed
0.25-scaled actions, 30/0.8 PD, 5ms physics / 20ms policy period, all 16 source
reward terms with final positive clipping, source reset distribution constrained
by official Go2 position stops, and startup mass/inertia/friction/COM/PD domain
randomization. Actor proprioception has source Gaussian corruption plus a
configurable 0--1 policy-step sensor latency; commands/actions are current.

The integrated live gate passed for eight environments and 32 policy steps. It
checks tensor shapes and finiteness, non-placeholder physical critic masses,
friction and 187 terrain samples, numerical sensor-delay behavior, official
position limits, action delay depth, all reward keys and all 16 reward metrics.
The source's always-true rotation predicate is preserved with its own consecutive
counter, independent of the runner's randomized episode-length initialization.
The target-pose schedule uses `(common_step-1)//24` so its iteration changes at
the same rollout boundary as the source runner.

Two 128-environment PPO updates passed and were synchronized to W&B run
`aia8bapk`. A production-size 4096-environment / 98,304-sample PPO update also
passed on GPU 1 in run directory
`logs/rsl_rl/go2_recovery_frnet_aligned/2026-09-11_02-59-22_fullscale_gate`.
The full 4000-iteration training has deliberately not been launched yet.

Source AST differential tests still pass for all 16 reward formulas and weights;
the 187-point source height sampler test and reward lifecycle test also pass.
The integrated critic uses a live 187-ray terrain measurement at the same grid
spacing/order, rather than a zero or synthetic height placeholder. This is an
engine mapping: MuJoCo ray-surface intersection is not bit-identical to Isaac's
three-neighbor heightfield minimum. Likewise, real Go2 sensor calibration data
has not been supplied; configured Gaussian amplitudes are the FR-Net assumptions,
and the latency model is configurable sim-to-real randomization, not a claim of
measured hardware calibration.

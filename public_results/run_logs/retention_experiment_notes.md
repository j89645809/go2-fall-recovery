# Success retention controlled pilot
GPU1 only. Frozen model14475 teacher; bounded residual +/-0.1 policy action units; fixed learning rate1e-5; no entropy bonus; 1024 environments. Control versus retention-loss arm, 25+25 updates each. Both restart identical composite checkpoint; same training seed/config except retention optimization step. All reset root/joint velocities zero; static fallen reset invariant retained.

The earlier v1 rollout comparison was not reproducible even for baseline versus itself (1000 vs1003 successes,385 outcome flips of1664 cases). Initial observations, pose, joint positions and first actions matched baseline/zero exactly; state differed after first physics step. Exact source of rollout variability is not yet established. Do not interpret gained/lost cases as causal evidence of training effects.

V2 explicitly reseeds reset, clears action and observation histories after static placement, fixes per-step noise seed. Strict static1664-case suite excludes angular-velocity and push tests. Not directly comparable with full1688-case suite. Original formal script unchanged.

Before training: evaluate baseline and zero at seeds1234,2345,3456; confirm teacher weights exact and residual output layer zero; all setups valid; each zero-vs-base rate gap within3pp as a screening safeguard, not proof of equivalence. Successful baseline1234 observations feed actual retention gradients. Seeds2345/3456 reserved for validation.

At25 updates stop arm for overall loss>3pp or any pose loss>5pp. At50 evaluate three seeds. Do not auto-promote either checkpoint; examine validation gain, worst-pose/terrain rates, recovery time, torque and oscillation. A small one-off improvement is insufficient. No claim of convergence.

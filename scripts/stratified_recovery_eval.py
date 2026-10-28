#!/usr/bin/env python3
"""Strict 4-terrain x 4-pose x 8-birth-mode recovery evaluation."""
import argparse, json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import torch
import mjlab.tasks
import src.tasks.velocity.config.go2_recovery_aligned
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TERRAINS = ("flat", "stairs", "slope", "rough")
POSES = ("front", "back", "left", "right")
MODES = ("independent_random", "single_FL", "single_FR", "single_RL", "single_RR",
         "diagonal_FL_RR", "diagonal_FR_RL", "all_folded")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True); p.add_argument("--output", required=True)
    p.add_argument("--terrains", nargs="+", choices=TERRAINS, default=list(TERRAINS))
    p.add_argument("--poses", nargs="+", choices=POSES, default=list(POSES))
    p.add_argument("--per-stratum", type=int, default=20); p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args(); torch.manual_seed(20260915)
    cfg = load_env_cfg("Unitree-Go2-Recovery-FRNet-Aligned", play=True); cfg.scene.num_envs = 1; cfg.viewer = None
    env = ManagerBasedRlEnv(cfg=cfg, device=a.device); wrapped = RslRlVecEnvWrapper(env)
    ac = load_rl_cfg("Unitree-Go2-Recovery-FRNet-Aligned")
    cls = load_runner_cls("Unitree-Go2-Recovery-FRNet-Aligned") or MjlabOnPolicyRunner
    runner = cls(wrapped, asdict(ac), device=a.device)
    runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=a.device)
    policy = runner.get_inference_policy(device=a.device)
    rows=[]; episode=0
    for terrain in a.terrains:
      for pose in a.poses:
        for mode in MODES:
          for _ in range(a.per_stratum):
            env.settled_reset_terrain_family=terrain; env.settled_reset_fall_pose=pose; env.settled_reset_birth_mode=mode
            env.reset(); obs=wrapped.get_observations(); hold=0; success_step=None
            for step in range(a.horizon):
              with torch.inference_mode(): act=policy(obs)
              obs,_,_,_=wrapped.step(act)
              robot=env.scene["robot"]
              upright=bool((robot.data.projected_gravity_b[0,2] < -0.90).item())
              height_ok=bool((robot.data.root_link_pos_w[0,2] > 0.30).item())
              hold=hold+1 if upright and height_ok else 0
              if hold>=50: success_step=step-49; break
            meta=dict(env.last_settled_reset); meta.update(episode=episode, success=success_step is not None,
              recovery_time_s=None if success_step is None else success_step*env.step_dt)
            rows.append(meta); episode+=1
            if episode % 100 == 0: print(json.dumps({"completed":episode,"total":len(a.terrains)*len(a.poses)*8*a.per_stratum}),flush=True)
    stats=defaultdict(lambda:[0,0])
    for r in rows:
      k=(r["terrain_family"],r["fall_pose"],r["birth_mode"]); stats[k][0]+=1; stats[k][1]+=int(r["success"])
    summary={"/".join(k):{"episodes":n,"success":s,"success_rate":s/n} for k,(n,s) in stats.items()}
    report={"checkpoint":a.checkpoint,"episodes":len(rows),"per_stratum":a.per_stratum,
      "strata":len(summary),"success_rate":sum(int(r["success"]) for r in rows)/len(rows),
      "stratum_summary":summary,"results":rows}
    Path(a.output).write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps({k:report[k] for k in ("episodes","strata","success_rate")}),flush=True); env.close()

if __name__ == "__main__": main()

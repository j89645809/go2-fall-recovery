import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from collections import Counter, defaultdict

import mjlab.tasks
import src.tasks.velocity.config.go2_recovery_aligned
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.manual_seed(20260914)
    cfg = load_env_cfg("Unitree-Go2-Recovery-FRNet-Aligned", play=True)
    cfg.scene.num_envs = 1
    cfg.viewer = None
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env)
    agent_cfg = load_rl_cfg("Unitree-Go2-Recovery-FRNet-Aligned")
    runner_cls = load_runner_cls("Unitree-Go2-Recovery-FRNet-Aligned") or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    rows = []
    for episode in range(args.episodes):
        env.reset()
        obs = wrapped.get_observations()
        hold = 0
        success_step = None
        for step in range(args.horizon):
            with torch.no_grad():
                action = policy(obs)
            obs, _, _, _ = wrapped.step(action)
            robot = env.scene["robot"]
            upright = bool((robot.data.projected_gravity_b[0, 2] < -0.90).item())
            height_ok = bool((robot.data.root_link_pos_w[0, 2] > 0.30).item())
            hold = hold + 1 if upright and height_ok else 0
            if hold >= 50:
                success_step = step - 49
                break
        meta = dict(env.last_settled_reset)
        meta.update(episode=episode, success=success_step is not None,
                    recovery_time_s=None if success_step is None else success_step * env.step_dt)
        rows.append(meta)
        print(json.dumps(meta), flush=True)
    mode_stats = defaultdict(lambda: {"success": 0, "episodes": 0})
    clamp_stats = Counter()
    joint_clamp_counts = Counter()
    for row in rows:
        mode = row["birth_mode"]
        mode_stats[mode]["episodes"] += 1
        mode_stats[mode]["success"] += int(row["success"])
        clamp_stats["any_clamped" if row["any_joint_clamped"] else "none_clamped"] += 1
        for name, was_clamped in row["joint_clamp_mask"].items():
            if was_clamped:
                joint_clamp_counts[name] += 1
    for value in mode_stats.values():
        value["success_rate"] = value["success"] / value["episodes"]
    report = {"checkpoint": args.checkpoint, "episodes": args.episodes,
              "success_rate": sum(r["success"] for r in rows) / len(rows),
              "birth_mode_summary": dict(mode_stats),
              "joint_clamp_summary": {
                  "episodes_with_any_clamp": clamp_stats["any_clamped"],
                  "episodes_without_clamp": clamp_stats["none_clamped"],
                  "clamp_count_by_joint": dict(joint_clamp_counts),
              },
              "results": rows}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("episodes", "success_rate")}), flush=True)
    env.close()


if __name__ == "__main__":
    main()

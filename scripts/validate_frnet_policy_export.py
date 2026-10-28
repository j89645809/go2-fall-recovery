"""Export and validate the final FR-Net-aligned recovery policy.

The validation compares deterministic PyTorch and ONNX actions on the exact
actor observations produced by mjlab across three fixed-seed rollouts.  The
rollout is driven with the PyTorch action so both backends always see identical
inputs.  No DDS or motor interface is imported or used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mjlab.tasks  # noqa: F401
import src.tasks.velocity.config.go2_recovery_aligned  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=(1234, 2234, 3234))
    args = parser.parse_args()

    task = "Unitree-Go2-Recovery-FRNet-Aligned"
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = args.num_envs
    cfg.viewer = None
    cfg.curriculum = {}
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env)
    agent_cfg = load_rl_cfg(task)
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
    runner.load(
        str(args.checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=args.device,
    )
    policy = runner.get_inference_policy(device=args.device)
    exported = runner.alg.get_policy().as_onnx(verbose=False).cpu().eval()
    dummy = exported.get_dummy_inputs()
    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        exported,
        dummy,
        str(args.onnx),
        export_params=True,
        opset_version=18,
        input_names=exported.input_names,
        output_names=exported.output_names,
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        dynamo=False,
    )

    import onnx
    from onnx.reference import ReferenceEvaluator

    model = onnx.load(str(args.onnx))
    onnx.checker.check_model(model)
    session = ReferenceEvaluator(model)
    per_seed = []
    global_worst = 0.0
    global_mean_sum = 0.0
    global_samples = 0
    all_finite = True
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env.reset(seed=seed)
        obs = wrapped.get_observations()
        seed_worst = 0.0
        seed_sum = 0.0
        seed_samples = 0
        for _ in range(args.steps):
            actor_obs = obs["actor"]
            with torch.inference_mode():
                torch_action = policy(obs)
            onnx_action = session.run(
                None, {"obs": actor_obs.detach().cpu().numpy().astype(np.float32)}
            )[0]
            reference = torch_action.detach().cpu().numpy()
            error = np.abs(onnx_action - reference)
            seed_worst = max(seed_worst, float(error.max()))
            seed_sum += float(error.sum())
            seed_samples += int(error.size)
            all_finite &= bool(np.isfinite(onnx_action).all())
            obs, _, _, _ = wrapped.step(torch_action)
        global_worst = max(global_worst, seed_worst)
        global_mean_sum += seed_sum
        global_samples += seed_samples
        per_seed.append(
            {
                "seed": seed,
                "steps": args.steps,
                "num_envs": args.num_envs,
                "max_action_abs_error": seed_worst,
                "mean_action_abs_error": seed_sum / seed_samples,
            }
        )

    dynamic_batches = {}
    rng = np.random.default_rng(9917)
    for batch in (1, 7, 32):
        sample = rng.normal(size=(batch, 45)).astype(np.float32)
        result = session.run(None, {"obs": sample})[0]
        dynamic_batches[str(batch)] = {
            "shape": list(result.shape),
            "finite": bool(np.isfinite(result).all()),
        }
        all_finite &= bool(np.isfinite(result).all()) and result.shape == (batch, 12)

    report = {
        "suite": "frnet_policy_pytorch_onnx_consistency",
        "task": task,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "onnx": str(args.onnx),
        "onnx_sha256": sha256(args.onnx),
        "onnx_checker_passed": True,
        "validation_backend": "onnx.reference.ReferenceEvaluator",
        "actor_observation_dim": 45,
        "action_dim": 12,
        "per_seed": per_seed,
        "dynamic_batch_tests": dynamic_batches,
        "max_action_abs_error": global_worst,
        "mean_action_abs_error": global_mean_sum / global_samples,
        "finite": all_finite,
        "tolerance": 2e-4,
        "passed": all_finite and global_worst <= 2e-4,
        "dds_or_motor_commands_sent": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    env.close()
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

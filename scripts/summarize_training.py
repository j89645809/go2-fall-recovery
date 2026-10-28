"""Summarize a completed RSL-RL TensorBoard run without W&B dependency."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def linear_slope(points: list[tuple[int, float]]) -> float | None:
    if len(points) < 2:
        return None
    xs = [float(x) for x, _ in points]
    ys = [float(y) for _, y in points]
    xm = sum(xs) / len(xs)
    ym = sum(ys) / len(ys)
    denom = sum((x - xm) ** 2 for x in xs)
    if denom == 0:
        return None
    return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / denom


def stats(events, window: int) -> dict:
    values = [(int(e.step), float(e.value)) for e in events]
    recent = values[-window:]
    previous = values[-2 * window : -window]
    rv = [v for _, v in recent]
    pv = [v for _, v in previous]
    return {
        "count": len(values),
        "first_step": values[0][0],
        "last_step": values[-1][0],
        "first": values[0][1],
        "last": values[-1][1],
        "min": min(v for _, v in values),
        "max": max(v for _, v in values),
        "recent_mean": sum(rv) / len(rv),
        "recent_min": min(rv),
        "recent_max": max(rv),
        "recent_slope_per_iteration": linear_slope(recent),
        "previous_mean": None if not pv else sum(pv) / len(pv),
        "recent_minus_previous": None if not pv else sum(rv) / len(rv) - sum(pv) / len(pv),
        "all_finite": all(math.isfinite(v) for _, v in values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("event_file")
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    accumulator = EventAccumulator(args.event_file, size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags()["scalars"]
    result = {
        "event_file": args.event_file,
        "window": args.window,
        "tags": {tag: stats(accumulator.Scalars(tag), args.window) for tag in tags},
    }
    mean_reward = result["tags"]["Train/mean_reward"]
    reward_delta = mean_reward["recent_minus_previous"]
    reward_slope = mean_reward["recent_slope_per_iteration"]
    all_finite = all(entry["all_finite"] for entry in result["tags"].values())
    if not all_finite:
        trend = "diverged"
    elif reward_delta is not None and reward_delta < -3.0:
        trend = "degrading"
    elif reward_delta is not None and reward_delta > 1.0:
        trend = "still_improving"
    elif reward_slope is not None and abs(reward_slope) < 0.005:
        trend = "reward_plateau"
    else:
        trend = "near_plateau"
    result["trend"] = {
        "classification": trend,
        "all_metrics_finite": all_finite,
        "note": "Reward plateau alone is not a recovery-success convergence criterion.",
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

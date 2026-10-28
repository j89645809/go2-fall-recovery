#!/usr/bin/env python3
"""Rebuild acceptance aggregates from per-case results without rerunning physics."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


# mjlab TerrainGenerator curriculum allocation uses:
#   first cumulative proportion greater than col / num_cols + 0.001
# The aligned task inherits proportions 0.2/0.2/0.2/0.1/0.1/0.1/0.1 and
# sets num_cols=8. Consequently wave_terrain receives no column.
COLUMN_TERRAIN = {
    0: "flat",
    1: "flat",
    2: "pyramid_stairs",
    3: "pyramid_stairs",
    4: "pyramid_stairs_inv",
    5: "hf_pyramid_slope",
    6: "hf_pyramid_slope_inv",
    7: "random_rough",
}


def rate(rows: list[dict], field: str = "success") -> float | None:
    if not rows:
        return None
    return sum(bool(row[field]) for row in rows) / len(rows)


def grouped(rows: list[dict], key: str) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[str(row[key])].append(row)
    result = {}
    for name, bucket in sorted(buckets.items()):
        successful_times = [row["recovery_time_s"] for row in bucket if row["success"]]
        result[name] = {
            "cases": len(bucket),
            "successes": sum(bool(row["success"]) for row in bucket),
            "success_rate": rate(bucket),
            "mean_recovery_time_s": mean(successful_times) if successful_times else None,
            "prone_oscillation_cases": sum(bool(row["prone_oscillation"]) for row in bucket),
            "post_recovery_oscillation_cases": sum(
                bool(row["post_recovery_oscillation"]) for row in bucket
            ),
            "in_place_spin_cases": sum(bool(row["in_place_spin"]) for row in bucket),
        }
    return result


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - index) + values[hi] * (index - lo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = json.loads(args.input.read_text())
    rows = source["cases"]
    for row in rows:
        row["terrain_name_corrected"] = COLUMN_TERRAIN[int(row["terrain_col"])]

    by_group = {name: [row for row in rows if row["group"] == name] for name in {
        "orientation_angle", "leg_fold", "terrain", "friction",
        "initial_angular_velocity", "disturbance",
    }}
    terrain_rows = by_group["terrain"]
    successful_times = [row["recovery_time_s"] for row in rows if row["success"]]
    torque_sat = [float(row["torque_saturation_fraction"]) for row in rows]
    clip = [float(row["action_clip_fraction"]) for row in rows]

    rates = {
        "group": grouped(rows, "group"),
        "pose_all_cases": grouped(rows, "pose"),
        "orientation_angle_only": grouped(by_group["orientation_angle"], "angle_deg"),
        "leg_fold_only": grouped(by_group["leg_fold"], "fold"),
        "terrain_type_only": grouped(terrain_rows, "terrain_name_corrected"),
        "terrain_column_only": grouped(terrain_rows, "terrain_col"),
        "terrain_row_only": grouped(terrain_rows, "terrain_row"),
        "terrain_position_only": grouped(terrain_rows, "terrain_position"),
        "friction_only": grouped(by_group["friction"], "friction"),
        "initial_angular_velocity_only": grouped(
            by_group["initial_angular_velocity"], "initial_ang_vel"
        ),
        "disturbance_only": grouped(by_group["disturbance"], "disturbance"),
    }
    pose_rates = [item["success_rate"] for item in rates["pose_all_cases"].values()]
    terrain_type_rates = [
        item["success_rate"] for item in rates["terrain_type_only"].values()
    ]
    terrain_patch_rates = [
        rate([row for row in terrain_rows if row["terrain_row"] == r and row["terrain_col"] == c])
        for r in range(10)
        for c in range(8)
    ]

    summary = {
        "source_result": str(args.input),
        "checkpoint": source["summary"].get("checkpoint"),
        "seed": source["summary"].get("seed"),
        "cases": len(rows),
        "valid_static_setups": sum(bool(row["valid_setup"]) for row in rows),
        "successes": sum(bool(row["success"]) for row in rows),
        "success_rate": rate(rows),
        "mean_recovery_time_s_successes": mean(successful_times),
        "peak_torque_nm": max(float(row["peak_torque_nm"]) for row in rows),
        "torque_saturation_fraction": {
            "mean": mean(torque_sat),
            "p95": quantile(torque_sat, 0.95),
            "max": max(torque_sat),
        },
        "action_clip_fraction": {
            "mean": mean(clip),
            "p95": quantile(clip, 0.95),
            "max": max(clip),
        },
        "behavior_flags": {
            "prone_oscillation_cases": sum(bool(row["prone_oscillation"]) for row in rows),
            "post_recovery_oscillation_cases": sum(
                bool(row["post_recovery_oscillation"]) for row in rows
            ),
            "in_place_spin_cases": sum(bool(row["in_place_spin"]) for row in rows),
        },
        "worst_pose_success_rate": min(pose_rates),
        "worst_terrain_type_success_rate": min(terrain_type_rates),
        "worst_terrain_patch_success_rate": min(terrain_patch_rates),
        "rates": rates,
        "terrain_allocation": {
            "num_rows": 10,
            "num_cols": 8,
            "configured_order": [
                "flat", "pyramid_stairs", "pyramid_stairs_inv",
                "hf_pyramid_slope", "hf_pyramid_slope_inv", "random_rough",
                "wave_terrain",
            ],
            "configured_proportions": [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1],
            "column_mapping": {str(key): value for key, value in COLUMN_TERRAIN.items()},
            "unallocated_types": ["wave_terrain"],
            "allocation_rule": "first cumulative proportion > col/8 + 0.001",
        },
    }
    gates = {
        "all_setups_valid": summary["valid_static_setups"] == len(rows),
        "overall_success_ge_0_95": summary["success_rate"] >= 0.95,
        "worst_pose_ge_0_90": summary["worst_pose_success_rate"] >= 0.90,
        "worst_terrain_type_ge_0_80": summary["worst_terrain_type_success_rate"] >= 0.80,
        "mean_torque_saturation_lt_0_05": summary["torque_saturation_fraction"]["mean"] < 0.05,
        "no_prone_oscillation": summary["behavior_flags"]["prone_oscillation_cases"] == 0,
        "no_post_recovery_oscillation": summary["behavior_flags"]["post_recovery_oscillation_cases"] == 0,
        "no_in_place_spin": summary["behavior_flags"]["in_place_spin_cases"] == 0,
    }
    summary["engineering_gate"] = gates
    summary["engineering_gate_pass"] = all(gates.values())
    summary["failure_counts_by_group"] = dict(Counter(
        row["group"] for row in rows if not row["success"]
    ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

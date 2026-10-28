#!/usr/bin/env python3
"""Merge four pose shards and evaluate the requested acceptance gates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

TERRAINS = ("flat", "stairs", "slope", "rough")
POSES = ("front", "back", "left", "right")
MODES = (
    "independent_random", "single_FL", "single_FR", "single_RL",
    "single_RR", "diagonal_FL_RR", "diagonal_FR_RL", "all_folded",
)


def stats(rows):
    success = sum(int(row["success"]) for row in rows)
    return {
        "episodes": len(rows),
        "success": success,
        "success_rate": success / len(rows),
        "hardware_safety_pass_rate": sum(
            int(row["hardware_safety_pass"]) for row in rows
        ) / len(rows),
        "reset_detected": sum(int(row["reset_detected"]) for row in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shards = []
    rows = []
    for pose in POSES:
        path = args.input_dir / f"{pose}.json"
        report = json.loads(path.read_text())
        shards.append(report)
        rows.extend(report["results"])

    counts = Counter(
        (row["terrain_family"], row["fall_pose"], row["birth_mode"])
        for row in rows
    )
    expected = {
        (terrain, pose, mode)
        for terrain in TERRAINS
        for pose in POSES
        for mode in MODES
    }
    assert len(rows) == 2560, len(rows)
    assert set(counts) == expected
    assert set(counts.values()) == {20}, Counter(counts.values())
    # Randomized source poses may be safely projected into official joint
    # limits; this is not an invalid initial state. Verify that every row
    # records the zero-velocity/contact checks and remains hardware-safe.
    initial_state_check = all(
        row["initial_contact"]
        and row["initial_root_speed"] <= 1.0e-8
        and row["initial_joint_speed"] <= 1.0e-8
        for row in rows
    )

    overall = stats(rows)
    by_terrain = {
        terrain: stats([row for row in rows if row["terrain_family"] == terrain])
        for terrain in TERRAINS
    }
    by_pose = {
        pose: stats([row for row in rows if row["fall_pose"] == pose])
        for pose in POSES
    }
    by_mode = {
        mode: stats([row for row in rows if row["birth_mode"] == mode])
        for mode in MODES
    }
    stratum_summary = {}
    for key in sorted(expected):
        stratum_summary["/".join(key)] = stats([
            row for row in rows
            if (row["terrain_family"], row["fall_pose"], row["birth_mode"]) == key
        ])

    gates = {
        "overall_at_least_90pct": overall["success_rate"] >= 0.90,
        "every_terrain_at_least_85pct": all(
            item["success_rate"] >= 0.85 for item in by_terrain.values()
        ),
        "all_folded_at_least_75pct": by_mode["all_folded"]["success_rate"] >= 0.75,
        "right_fall_at_least_85pct": by_pose["right"]["success_rate"] >= 0.85,
        "every_stratum_has_20": set(counts.values()) == {20},
        "all_initial_states_verified": initial_state_check,
        "no_mid_episode_resets": not any(row["reset_detected"] for row in rows),
    }
    report = {
        "checkpoint": shards[0]["checkpoint"],
        "suite": "native_terrain_decoupled_random_position_recovery_2560",
        "episodes": len(rows),
        "strata": len(counts),
        "per_stratum": 20,
        "horizon_s": shards[0]["horizon_s"],
        "overall": overall,
        "by_terrain": by_terrain,
        "by_pose": by_pose,
        "by_birth_mode": by_mode,
        "acceptance_gates": gates,
        "acceptance_pass": all(gates.values()),
        "training_required": not all(gates.values()),
        "stratum_summary": stratum_summary,
        "results": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "checkpoint": report["checkpoint"],
        "overall": overall,
        "by_terrain": by_terrain,
        "by_pose": by_pose,
        "all_folded": by_mode["all_folded"],
        "acceptance_gates": gates,
        "acceptance_pass": report["acceptance_pass"],
        "training_required": report["training_required"],
    }, indent=2))


if __name__ == "__main__":
    main()

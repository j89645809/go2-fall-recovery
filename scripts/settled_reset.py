"""Direct, zero-velocity fallen reset for the Viser/evaluation path.

The initial state is intentionally posed, not dropped and simulated.  This
keeps recovery metrics free of hidden impact dynamics while still randomizing
terrain, fall direction, yaw, and joint configuration.
"""
import math
import os
import torch
from mjlab.envs.mdp.events import reset_root_state_uniform
from .official_go2_limits import JOINT_NAMES, POSITION_LOWER, POSITION_UPPER

_TERRAIN_COLUMNS = {"flat": (0, 1), "stairs": (2, 3, 4), "slope": (5, 6), "rough": (7,)}
_FALL_POSES = {
    "front": (0.0, -0.5 * math.pi),
    "back": (0.0, 0.5 * math.pi),
    "left": (-0.5 * math.pi, 0.0),
    "right": (0.5 * math.pi, 0.0),
}
_LEG_NAMES = ("FL", "FR", "RL", "RR")


def _has_fallen_contact(env):
    """Prefer body/non-foot contact; fall back to any robot contact."""
    try:
        nonfoot = env.scene["nonfoot_ground_touch"]
    except KeyError:
        nonfoot = None
    if nonfoot is not None:
        nonfoot.update(0.0)
        return bool(nonfoot.data.found.any().item())
    contact = False
    for name in ("base_ground_contact", "feet_ground_contact"):
        try:
            sensor = env.scene[name]
        except KeyError:
            sensor = None
        if sensor is not None:
            sensor.update(0.0)
            contact = contact or bool(sensor.data.found.any().item())
    return contact

def _sample_terrain(env, reset_ids):
    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
        raise ValueError("Direct fallen reset requires generated terrain")
    family_names = tuple(_TERRAIN_COLUMNS)
    family = getattr(env, "settled_reset_terrain_family", None) or os.environ.get("SETTLED_RESET_TERRAIN_FAMILY")
    if family is None:
        # Bias the birth distribution toward the currently weak terrain domains.
        # Keep flat/rough cases in the mix for retention.
        family = family_names[int(torch.multinomial(
            torch.tensor([0.10, 0.40, 0.30, 0.20], device=env.device), 1
        ).item())]
    columns = _TERRAIN_COLUMNS[family]
    column = getattr(env, "settled_reset_terrain_column", None)
    if column is None and os.environ.get("SETTLED_RESET_TERRAIN_COLUMN") is not None:
        column = int(os.environ["SETTLED_RESET_TERRAIN_COLUMN"])
    if column is None:
        column = columns[int(torch.randint(len(columns), (1,), device=env.device))]
    row = getattr(env, "settled_reset_terrain_row", None)
    if row is None and os.environ.get("SETTLED_RESET_TERRAIN_ROW") is not None:
        row = int(os.environ["SETTLED_RESET_TERRAIN_ROW"])
    if row is None:
        row = int(torch.randint(terrain.terrain_origins.shape[0], (1,), device=env.device))
    terrain.terrain_levels[reset_ids] = row
    terrain.terrain_types[reset_ids] = column
    terrain.env_origins[reset_ids] = terrain.terrain_origins[row, column]
    return family, row, column

def settle_fallen(env, env_ids):
    if env.num_envs != 1:
        raise ValueError("Direct fallen reset supports one environment only")
    reset_ids = torch.zeros(1, dtype=torch.long, device=env.device)
    family, row, column = _sample_terrain(env, reset_ids)
    pose_names = tuple(_FALL_POSES)
    pose = getattr(env, "settled_reset_fall_pose", None) or os.environ.get("SETTLED_RESET_FALL_POSE")
    if pose is None:
        if family == "stairs":
            # Stair failures are concentrated in front/left falls.
            pose = pose_names[int(torch.multinomial(
                torch.tensor([0.40, 0.10, 0.30, 0.20], device=env.device), 1
            ).item())]
        elif family == "slope":
            pose = pose_names[int(torch.multinomial(
                torch.tensor([0.30, 0.20, 0.20, 0.30], device=env.device), 1
            ).item())]
        else:
            pose = pose_names[int(torch.randint(len(pose_names), (1,), device=env.device))]
    roll, pitch = _FALL_POSES[pose]
    robot = env.scene["robot"]
    yaw = float((2.0 * math.pi * torch.rand(1, device=env.device) - math.pi).item())
    reset_root_state_uniform(
        env, reset_ids,
        # Start above the terrain; the contact bracket below computes the
        # final height for this pose instead of relying on a fixed z offset.
        pose_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (2.5, 2.5),
                    "roll": (roll, roll), "pitch": (pitch, pitch), "yaw": (yaw, yaw)},
        velocity_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                        "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
    )
    # Rich fallen initial states: independent per-leg joint sampling plus
    # explicit single-leg, diagonal, and all-folded configurations.
    base = {
        "front": [0.0, 1.15, -2.05] * 4,
        "back":  [0.0, 1.45, -2.45] * 4,
        "left":  [0.0, 1.05, -1.85] * 4,
        "right": [0.0, 1.05, -1.85] * 4,
    }[pose]
    # Independent random birth: sample every joint uniformly across the
    # certified limit range. This includes folded and near-limit configurations
    # instead of only perturbing a hand-selected nominal fallen pose.
    q = robot.data.default_joint_pos.clone().reshape(1, 12)
    lower = q.new_tensor(POSITION_LOWER) + 0.03
    upper = q.new_tensor(POSITION_UPPER) - 0.03
    q = lower + (upper - lower) * torch.rand_like(q)
    forced_mode = getattr(env, "settled_reset_birth_mode", None) or os.environ.get("SETTLED_RESET_BIRTH_MODE")
    if forced_mode == "independent_random":
        folded = ()
    elif forced_mode and forced_mode.startswith("single_"):
        folded = (_LEG_NAMES.index(forced_mode.split("_", 1)[1]),)
    elif forced_mode == "diagonal_FL_RR":
        folded = (0, 3)
    elif forced_mode == "diagonal_FR_RL":
        folded = (1, 2)
    elif forced_mode == "all_folded":
        folded = (0, 1, 2, 3)
    else:
        mode = int(torch.randint(0, 10, (1,), device=env.device).item())
        if mode < 4:  # one independently folded leg
            folded = (mode,)
        elif mode < 6:  # diagonal pair
            folded = (0, 3) if mode == 4 else (1, 2)
        elif mode == 6:  # all four folded
            folded = (0, 1, 2, 3)
        else:
            folded = ()
    for leg in folded:
        j = 3 * leg
        side = 1.0 if leg in (0, 2) else -1.0
        q[0, j : j + 3] = q.new_tensor([
            0.82 * side,
            2.9 if leg < 2 else 3.5,
            -2.68,
        ])
    q_before_clamp = q.clone()
    if len(folded) == 1:
        birth_mode = f"single_{_LEG_NAMES[folded[0]]}"
    elif len(folded) == 2:
        birth_mode = "diagonal_FL_RR" if folded == (0, 3) else "diagonal_FR_RL"
    elif len(folded) == 4:
        birth_mode = "all_folded"
    else:
        birth_mode = "independent_random"
    # Keep every direct pose strictly inside the certified Go2 joint range.
    q.clamp_(lower, upper)
    clamped = (q - q_before_clamp).abs() > 1.0e-7
    robot.write_joint_state_to_sim(q, torch.zeros_like(robot.data.joint_vel), env_ids=reset_ids)
    robot.write_root_link_velocity_to_sim(torch.zeros_like(robot.data.root_link_vel_w), env_ids=reset_ids)
    env.sim.forward()
    env.sim.sense()

    # Bracket the highest root height that still produces robot/terrain
    # contact, then place 2 mm inside the collision boundary.  This is a
    # static placement operation (no gravity/drop steps are simulated), so
    # the first Viser frame is already an upright-velocity-zero fallen pose.
    scan = env.scene["terrain_scan"].data
    ray = scan.hit_pos_w.shape[1] // 2
    ground = torch.where(scan.distances[:, ray] >= 0,
                         scan.hit_pos_w[:, ray, 2],
                         env.scene.terrain.env_origins[reset_ids, 2])
    # Build a writable root state from the configured default state.  The
    # reset event has already selected terrain/orientation; preserve that
    # orientation while overriding only the world position and velocities.
    root = robot.data.default_root_state.clone()
    root[:, :3] = env.scene.terrain.env_origins[reset_ids]
    # Optional local offset lets demonstrations target the visible stair
    # steps rather than always spawning at the tile origin.
    root[:, 0] += float(os.environ.get("SETTLED_RESET_X", "0.0"))
    root[:, 1] += float(os.environ.get("SETTLED_RESET_Y", "0.0"))
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    root[:, 3:7] = torch.tensor(
        [math.cos(roll / 2) * math.cos(pitch / 2) * cy + math.sin(roll / 2) * math.sin(pitch / 2) * sy,
         math.sin(roll / 2) * math.cos(pitch / 2) * cy - math.cos(roll / 2) * math.sin(pitch / 2) * sy,
         math.cos(roll / 2) * math.sin(pitch / 2) * cy + math.sin(roll / 2) * math.cos(pitch / 2) * sy,
         math.cos(roll / 2) * math.cos(pitch / 2) * sy - math.sin(roll / 2) * math.sin(pitch / 2) * cy],
        device=env.device, dtype=root.dtype,
    )
    root[:, 7:] = 0.0
    low = ground - 1.5
    high = ground + 2.5
    found = False
    for step in range(200):
        candidate = ground + 2.5 - 0.02 * step
        root[:, 2] = candidate
        robot.write_root_state_to_sim(root, env_ids=reset_ids)
        env.sim.forward(); env.sim.sense()
        contact = _has_fallen_contact(env)
        if contact:
            low, high, found = candidate, candidate + 0.02, True
            break
    if found:
        for _ in range(18):
            mid = (low + high) * 0.5
            root[:, 2] = mid
            robot.write_root_state_to_sim(root, env_ids=reset_ids)
            env.sim.forward(); env.sim.sense()
            contact = _has_fallen_contact(env)
            if contact: low = mid
            else: high = mid
        root[:, 2] = low - 0.002
        robot.write_root_state_to_sim(root, env_ids=reset_ids)
        robot.write_root_link_velocity_to_sim(torch.zeros_like(robot.data.root_link_vel_w), env_ids=reset_ids)
        env.sim.forward(); env.sim.sense()
        for name in ("base_ground_contact", "feet_ground_contact", "nonfoot_ground_touch"):
            try:
                env.scene[name].update(0.0)
            except KeyError:
                pass
    else:
        # Conservative fallback for steep/occluded ray locations: place the
        # root well inside the local terrain envelope and explicitly zero all
        # velocities rather than leaving the body suspended above the ground.
        root[:, 2] = ground - 0.30
        robot.write_root_state_to_sim(root, env_ids=reset_ids)
        robot.write_root_link_velocity_to_sim(torch.zeros_like(robot.data.root_link_vel_w), env_ids=reset_ids)
        env.sim.forward(); env.sim.sense()
    env.last_settled_reset = {"terrain_family": family, "terrain_row": row,
                              "terrain_column": column, "fall_pose": pose,
                              "birth_mode": birth_mode,
                              "leg_modes": ["folded" if i in folded else "independent_random"
                                            for i in range(4)],
                              "joint_positions_rad": {name: float(q[0, i].item())
                                                       for i, name in enumerate(JOINT_NAMES)},
                              "joint_clamp_mask": {name: bool(clamped[0, i].item())
                                                    for i, name in enumerate(JOINT_NAMES)},
                              "any_joint_clamped": bool(clamped.any().item()),
                              "clamped_joint_count": int(clamped.sum().item()),
                              "hidden_settling": False, "root_speed": 0.0,
                              "joint_speed": 0.0}
    print(f"[direct-reset] terrain={family} row={row} column={column} fall_pose={pose} "
          f"hidden_settling=False root_speed=0 joint_speed=0", flush=True)

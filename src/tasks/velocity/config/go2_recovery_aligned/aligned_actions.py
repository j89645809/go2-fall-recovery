"""FR-Net policy-rate delay with explicit clipped action history."""
from dataclasses import dataclass
import os
import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from .baseline_protocol import ActionDelay, JOINT_NAMES
from .official_go2_limits import POSITION_LOWER, POSITION_UPPER, VELOCITY_LIMIT
from .recovery_acceptance import STAND_JOINT_TARGET, onboard_stand_ready


HIP_INDICES = (0, 3, 6, 9)
THIGH_INDICES = (1, 4, 7, 10)
CALF_INDICES = (2, 5, 8, 11)


def update_predictive_brake_latch(
    previous, requested, velocity, velocity_limit, *, release_ratio=0.20,
):
    """Hold reverse braking until the hazardous joint motion has slowed.

    Latch values are motion directions: +1 brakes positive motion and -1
    brakes negative motion.  A fresh request always wins.  Without a fresh
    request, braking persists only while the joint is still moving in the
    latched direction above the release speed.
    """
    limits = velocity_limit.expand_as(velocity)
    moving_in_latched_direction = velocity * previous > 0.0
    still_fast = velocity.abs() > limits * float(release_ratio)
    retained = torch.where(
        (previous != 0.0) & moving_in_latched_direction & still_fast,
        previous,
        torch.zeros_like(previous),
    )
    return torch.where(requested != 0.0, requested, retained)


def detect_deep_fold(q, *, calf_threshold=-2.45, thigh_threshold=2.50,
                     minimum_legs=3):
    """Identify a three/four-leg deep fold from onboard joint encoders."""
    thighs = q[..., THIGH_INDICES]
    calves = q[..., CALF_INDICES]
    folded = (thighs > thigh_threshold) & (calves < calf_threshold)
    return folded.sum(dim=-1) >= minimum_legs


def staged_unfold_command(
    q, velocity, *, thigh_goal=2.25, calf_goal=-2.25,
    thigh_step=0.012, calf_step=0.018, ready_speed=1.5,
):
    """Move a deeply folded robot toward a neutral intermediate pose slowly.

    Hip targets stay at the measured angle so the deterministic phase cannot
    introduce a lateral bias.  Thigh/calf targets sit only a small distance
    ahead of their measured positions, limiting stored PD energy even if a
    leg is blocked by terrain contact.
    """
    desired = q.clone()
    desired[..., THIGH_INDICES] = torch.maximum(
        q[..., THIGH_INDICES] - thigh_step,
        q.new_full(q[..., THIGH_INDICES].shape, thigh_goal),
    )
    desired[..., CALF_INDICES] = torch.minimum(
        q[..., CALF_INDICES] + calf_step,
        q.new_full(q[..., CALF_INDICES].shape, calf_goal),
    )
    relevant = THIGH_INDICES + CALF_INDICES
    pose_ready = (
        (q[..., THIGH_INDICES] <= thigh_goal + 0.05).all(dim=-1)
        & (q[..., CALF_INDICES] >= calf_goal - 0.05).all(dim=-1)
    )
    speed_ready = velocity[..., relevant].abs().amax(dim=-1) <= ready_speed
    return desired, pose_ready & speed_ready


def blend_staged_policy(unfold_desired, policy_desired, step, *, handoff_steps=50):
    """Blend from the low-energy unfold target into the learned policy."""
    progress=(step.to(unfold_desired.dtype)/float(max(handoff_steps,1))).clamp(0.0,1.0)
    return torch.lerp(unfold_desired,policy_desired,progress.unsqueeze(-1))


def staged_initial_hold(previous_target, unfold_desired, step, *, hold_steps=10):
    """Keep the reset-synchronized PD target before beginning to unfold."""
    hold=step <= hold_steps
    return torch.where(hold.unsqueeze(-1),previous_target,unfold_desired)


def hold_reset_synchronized_target(previous_target, proposed_target, step, hold_steps):
    """Keep the zero-jump reset target for a short contact acquisition window."""
    if not torch.is_tensor(hold_steps) and hold_steps <= 0:
        return proposed_target
    step_tensor=torch.as_tensor(step,device=proposed_target.device)
    if step_tensor.ndim == 0:
        step_tensor=step_tensor.expand(proposed_target.shape[:-1])
    # hold_steps may be a scalar or one value per environment.  The latter
    # lets deeply folded states keep their measured contact target longer
    # without freezing ordinary fallen states.
    if torch.is_tensor(hold_steps):
        hold_limit=hold_steps.to(device=step_tensor.device,dtype=step_tensor.dtype)
        if hold_limit.ndim == 0:
            hold_limit=hold_limit.expand_as(step_tensor)
    else:
        hold_limit=step_tensor.new_full(step_tensor.shape,int(hold_steps))
    hold=step_tensor < hold_limit
    return torch.where(hold.unsqueeze(-1),previous_target,proposed_target)


def reset_contact_force_history(previous_force, ids):
    """Clear contact-force deltas for environments that have just reset."""
    if previous_force is not None and previous_force.numel() > 0:
        previous_force[ids] = 0.0


def filter_dynamic_contact_rise(
    rising, force, armed, age, *, threshold, release_threshold,
    warmup_steps=20, enabled=False,
):
    """Separate reset support contact from a later dynamic re-contact.

    ``rising`` is the raw sensor edge.  When enabled, a sensor that starts
    loaded remains disarmed until it unloads below ``release_threshold`` and
    the reset settling window has elapsed.  The default is deliberately
    disabled for the release action path; callers can enable it for causal
    source diagnostics without changing the conservative safety fallback.
    """
    armed_next = armed | (force < float(release_threshold))
    if not enabled:
        return rising, armed_next
    eligible = age.unsqueeze(-1) >= int(warmup_steps)
    return rising & armed_next & eligible, armed_next


def interpolate_reset_target(start_target, reference_target, step, steps):
    """Move a reset PD target from measured q to its release target."""
    count = max(int(steps), 1)
    progress = torch.as_tensor(step, device=start_target.device, dtype=start_target.dtype)
    progress = (progress / float(count)).clamp(0.0, 1.0)
    while progress.ndim < start_target.ndim:
        progress = progress.unsqueeze(-1)
    return torch.lerp(start_target, reference_target, progress)


def limit_target_slew(q, desired, max_step, *, calf_indices=(2, 5, 8, 11), calf_max_step=None):
    """Limit one policy target update without reversing the requested motion."""
    if max_step <= 0.0 and (calf_max_step is None or calf_max_step <= 0.0):
        return desired
    step = torch.full_like(q, float(max_step))
    if calf_max_step is not None and calf_max_step > 0.0:
        for index in calf_indices:
            step[..., index] = float(calf_max_step)
    delta = (desired - q).clamp(min=-step, max=step)
    return q + delta


def apply_predictive_target_brake(
    q, velocity, desired, lower, upper, velocity_limit, brake_accel, dt,
    *, margin=0.0, trigger_ratio=0.30, gain=0.50,
    calf_indices=(2, 5, 8, 11), calf_gain=None,
):
    """Shorten only targets whose predicted stopping point crosses a bound."""
    if gain <= 0.0:
        return desired
    margin_tensor = (
        margin.expand_as(q) if torch.is_tensor(margin)
        else torch.full_like(q, float(margin))
    )
    lo = lower.expand_as(q) + margin_tensor
    hi = upper.expand_as(q) - margin_tensor
    accel = brake_accel.expand_as(q).clamp_min(1e-3)
    stop_distance = velocity.abs() * float(dt) + velocity.square() / (2.0 * accel)
    distance = torch.where(
        velocity >= 0.0, (hi - q).clamp_min(0.0), (q - lo).clamp_min(0.0)
    )
    risk = ((stop_distance - distance) / stop_distance.clamp_min(1e-6)).clamp(0.0, 1.0)
    limit = velocity_limit.expand_as(q).clamp_min(1e-6)
    driving = velocity * (desired - q) > 0.0
    active = driving & (velocity.abs() >= limit * float(trigger_ratio)) & (risk > 0.0)
    gains = torch.full_like(q, float(gain))
    if calf_gain is not None and calf_gain > 0.0:
        for index in calf_indices:
            gains[..., index] = float(calf_gain)
    scale = torch.where(active, (1.0 - gains * risk).clamp_min(0.0), torch.ones_like(q))
    return q + (desired - q) * scale


def apply_source_leg_coordination(
    q, velocity, desired, source_leg, source_joint, trigger, *,
    calf_scale=0.35, thigh_scale=0.65, hip_scale=1.0,
):
    """Dampen only source-leg calf/thigh continuation after confirmation.

    The source is a leg-level event, while ``source_joint`` records the joint
    that supplied the strongest evidence.  We damp the source leg's calf and
    thigh only when they are still moving in the commanded direction.  The
    hip and all other legs keep their original target.
    """
    joint_index = torch.arange(12, device=q.device).view(1, 12)
    leg_index = joint_index // 3
    source_active = (
        trigger.unsqueeze(-1)
        & (source_leg.unsqueeze(-1) >= 0)
        & (leg_index == source_leg.unsqueeze(-1))
    )
    # A valid source joint is useful for diagnostics, but a leg event still
    # coordinates the whole source leg.  Invalid source joints never activate
    # a protection event by themselves.
    source_active = source_active & (
        (source_joint.unsqueeze(-1) >= 0) | (source_leg.unsqueeze(-1) >= 0)
    )
    calf_mask = source_active & ((joint_index % 3) == 2)
    thigh_mask = source_active & ((joint_index % 3) == 1)
    hip_mask = source_active & ((joint_index % 3) == 0)
    driving = velocity * (desired - q) > 0.0
    scale = torch.ones_like(q)
    scale = torch.where(hip_mask, torch.full_like(q, float(hip_scale)), scale)
    scale = torch.where(calf_mask, torch.full_like(q, float(calf_scale)), scale)
    scale = torch.where(thigh_mask, torch.full_like(q, float(thigh_scale)), scale)
    # Hip coordination is intentionally enabled only when the caller asks
    # for a hip scale below 1.0.  The previous implementation computed the
    # hip mask but omitted it from this final predicate, so hip_scale had no
    # effect and an adjacent hip could remain the first overspeed joint.
    return torch.where(
        driving & (calf_mask | thigh_mask | (hip_mask & (float(hip_scale) < 1.0))),
        q + (desired - q) * scale,
        desired,
    )


def classify_impact_source(
    q, velocity, proposed, lower, upper, velocity_limit, velocity_delta,
    contact_rise_leg, previous_streak, *, boundary_margin=0.10,
    min_speed_ratio=0.15, acceleration_ratio=0.05, high_speed_ratio=0.60,
    min_consecutive=2, detection_eligible=None, deep_folded=None,
):
    """Classify a leg only when motion and impact evidence coincide.

    This is deliberately a pure diagnostic primitive.  Static contact alone
    cannot trigger it: the leg must be moving toward its target and show a
    contact-force rise, an outward velocity change, a boundary approach, or
    high speed.  The consecutive-frame streak rejects one-frame sensor
    chatter and makes the source leg distinct from a later peak-speed leg.
    """
    limits = velocity_limit.expand_as(velocity).clamp_min(1e-6)
    target_delta = proposed - q
    driving = velocity * target_delta > 0.0
    speed_ratio = velocity.abs() / limits
    moving = driving & (speed_ratio >= float(min_speed_ratio))
    sudden = (
        driving
        & (velocity * velocity_delta > 0.0)
        & (velocity_delta.abs() / limits >= float(acceleration_ratio))
    )
    near_boundary = (
        ((q >= upper - float(boundary_margin)) & (velocity > 0.0))
        | ((q <= lower + float(boundary_margin)) & (velocity < 0.0))
    ) & driving
    high_speed = moving & (speed_ratio >= float(high_speed_ratio))
    evidence = contact_rise_leg.to(dtype=torch.bool)
    strict_source = os.environ.get("FRNET_IMPACT_SOURCE_STRICT", "0") == "1"
    if strict_source:
        # Contact alone is static support, not an impact.  In strict mode a
        # contact rise must coincide with outward acceleration or boundary
        # approach; acceleration/boundary evidence can also qualify without a
        # contact sensor.  This cuts false source events while preserving
        # genuine contact transfers.
        contact_motion = evidence.repeat_interleave(3, dim=-1) & (
            sudden | near_boundary
        )
        dynamic_motion = sudden & near_boundary
        high_speed_motion = high_speed & (sudden | near_boundary)
        strict_joint = moving & (contact_motion | dynamic_motion | high_speed_motion)
        joint_evidence = torch.stack(
            [strict_joint[..., 0:3].any(dim=-1),
             strict_joint[..., 3:6].any(dim=-1),
             strict_joint[..., 6:9].any(dim=-1),
             strict_joint[..., 9:12].any(dim=-1)], dim=-1,
        )
    else:
        joint_evidence = torch.stack(
            [
                (moving[..., 0:3] & (evidence[..., 0:1] | sudden[..., 0:3] | near_boundary[..., 0:3])).any(dim=-1),
                (moving[..., 3:6] & (evidence[..., 1:2] | sudden[..., 3:6] | near_boundary[..., 3:6])).any(dim=-1),
                (moving[..., 6:9] & (evidence[..., 2:3] | sudden[..., 6:9] | near_boundary[..., 6:9])).any(dim=-1),
                (moving[..., 9:12] & (evidence[..., 3:4] | sudden[..., 9:12] | near_boundary[..., 9:12])).any(dim=-1),
            ],
            dim=-1,
        )
    # A reset starts from a static fallen pose.  During the contact-acquisition
    # window no motion/contact signal is allowed to create an impact source;
    # the window only establishes the static support baseline.
    if detection_eligible is not None:
        joint_evidence = joint_evidence & detection_eligible.to(dtype=torch.bool).unsqueeze(-1)
    raw_event = joint_evidence
    streak = torch.where(
        raw_event,
        previous_streak + 1,
        torch.zeros_like(previous_streak),
    )
    trigger = streak >= int(min_consecutive)
    leg_scores = torch.stack(
        [
            (moving[..., 0:3] & evidence[..., 0:1]).any(dim=-1).to(q.dtype) * 2.0
            + sudden[..., 0:3].any(dim=-1).to(q.dtype)
            + near_boundary[..., 0:3].any(dim=-1).to(q.dtype)
            + high_speed[..., 0:3].any(dim=-1).to(q.dtype),
            (moving[..., 3:6] & evidence[..., 1:2]).any(dim=-1).to(q.dtype) * 2.0
            + sudden[..., 3:6].any(dim=-1).to(q.dtype)
            + near_boundary[..., 3:6].any(dim=-1).to(q.dtype)
            + high_speed[..., 3:6].any(dim=-1).to(q.dtype),
            (moving[..., 6:9] & evidence[..., 2:3]).any(dim=-1).to(q.dtype) * 2.0
            + sudden[..., 6:9].any(dim=-1).to(q.dtype)
            + near_boundary[..., 6:9].any(dim=-1).to(q.dtype)
            + high_speed[..., 6:9].any(dim=-1).to(q.dtype),
            (moving[..., 9:12] & evidence[..., 3:4]).any(dim=-1).to(q.dtype) * 2.0
            + sudden[..., 9:12].any(dim=-1).to(q.dtype)
            + near_boundary[..., 9:12].any(dim=-1).to(q.dtype)
            + high_speed[..., 9:12].any(dim=-1).to(q.dtype),
        ],
        dim=-1,
    )
    # Source scoring keeps contact rise as a strong signal, but in strict
    # mode acceleration, boundary approach, and high speed remain valid
    # dynamic evidence even when the contact sensor has not risen yet.
    # Static support alone still cannot score because ``moving`` is required.
    joint_scores = (
        moving.to(q.dtype) * (
            evidence.repeat_interleave(3, dim=-1).to(q.dtype) * 2.0
            + sudden.to(q.dtype)
            + near_boundary.to(q.dtype)
            + high_speed.to(q.dtype)
        )
    )
    if strict_source:
        joint_scores = moving.to(q.dtype) * (
            evidence.repeat_interleave(3, dim=-1).to(q.dtype) * 2.0
            + sudden.to(q.dtype)
            + near_boundary.to(q.dtype)
            + high_speed.to(q.dtype)
        )
        leg_scores = torch.stack(
            [joint_scores[..., 0:3].amax(dim=-1),
             joint_scores[..., 3:6].amax(dim=-1),
             joint_scores[..., 6:9].amax(dim=-1),
             joint_scores[..., 9:12].amax(dim=-1)], dim=-1,
        )
    source_score = torch.where(trigger, leg_scores, torch.zeros_like(leg_scores))
    source_value, source_leg = source_score.max(dim=-1)
    source_leg = torch.where(source_value > 0.0, source_leg, torch.full_like(source_leg, -1))
    joint_leg = torch.arange(12, device=q.device).div(3, rounding_mode="floor")
    same_source_leg = joint_leg.view(1, 12) == source_leg.unsqueeze(-1)
    masked_joint_scores = torch.where(
        same_source_leg, joint_scores, torch.full_like(joint_scores, -1.0)
    )
    joint_value, source_joint = masked_joint_scores.max(dim=-1)
    source_joint = torch.where(
        trigger.repeat_interleave(3, dim=-1).any(dim=-1) & (joint_value > 0.0),
        source_joint,
        torch.full_like(source_joint, -1),
    )
    return {
        "moving": moving,
        "sudden": sudden,
        "near_boundary": near_boundary,
        "high_speed": high_speed,
        "raw_event": raw_event,
        "streak": streak,
        "trigger": trigger,
        "leg_scores": leg_scores,
        "joint_scores": joint_scores,
        "source_leg": source_leg,
        "source_joint": source_joint,
    }


def compute_safety_target(
    q, velocity, desired, lower, upper, velocity_limit, margin,
    brake_accel, dt, safety_step, *, calf_indices=(2, 5, 8, 11),
    startup_start=0.50, startup_end=0.80, startup_steps=50,
    brake_trigger_ratio=1.00, predictive_all_joints=False,
    general_trigger_ratio=0.50, calf_trigger_ratio=0.30,
    general_reverse_offset=0.015, calf_reverse_offset=0.030,
    velocity_brake_enabled=False, velocity_brake_general_trigger=0.80,
    velocity_brake_calf_trigger=0.70, velocity_brake_hip_trigger=None,
):
    """Limit a policy target using safe position and stopping-distance rules."""
    safe_lower = lower + margin
    safe_upper = upper - margin
    accel = brake_accel.clamp_min(1e-3)
    desired = torch.maximum(torch.minimum(desired, safe_upper), safe_lower)
    brake_distance = velocity.abs() * dt + velocity.square() / (2.0 * accel)
    stop_position = q + torch.sign(velocity) * brake_distance
    distance_up = (safe_upper - q).clamp_min(0.0)
    distance_down = (q - safe_lower).clamp_min(0.0)
    allowed_up = velocity_limit.expand_as(q).clone()
    allowed_down = velocity_limit.expand_as(q).clone()
    dynamic_mask = torch.ones_like(q,dtype=torch.bool) if predictive_all_joints else torch.zeros_like(q,dtype=torch.bool)
    trigger_ratio=torch.full_like(q,general_trigger_ratio if predictive_all_joints else brake_trigger_ratio)
    for index in calf_indices:
        dynamic_mask[..., index] = True
        if predictive_all_joints:
            trigger_ratio[...,index]=calf_trigger_ratio
    fast_enough = velocity.abs() >= velocity_limit.expand_as(q) * trigger_ratio
    overshoot_up = (stop_position > safe_upper) & (velocity > 0.0) & dynamic_mask & fast_enough
    overshoot_down = (stop_position < safe_lower) & (velocity < 0.0) & dynamic_mask & fast_enough
    brake_speed_up = torch.sqrt(2.0 * accel * distance_up)
    brake_speed_down = torch.sqrt(2.0 * accel * distance_down)
    allowed_up = torch.where(overshoot_up, torch.minimum(allowed_up, brake_speed_up), allowed_up)
    allowed_down = torch.where(overshoot_down, torch.minimum(allowed_down, brake_speed_down), allowed_down)
    step = torch.as_tensor(safety_step, device=q.device, dtype=q.dtype)
    if step.ndim == 0:
        step = step.expand(q.shape[:-1])
    progress = (step / float(max(startup_steps, 1))).clamp(0.0, 1.0)
    progress = progress.unsqueeze(-1)
    calf_scale = startup_start + (startup_end - startup_start) * progress
    calf_scale_for_limit = calf_scale if velocity_limit.ndim > 1 else calf_scale.squeeze(-1)
    for index in calf_indices:
        allowed_up[..., index] = torch.minimum(allowed_up[..., index], velocity_limit[..., index] * calf_scale_for_limit)
        allowed_down[..., index] = torch.minimum(allowed_down[..., index], velocity_limit[..., index] * calf_scale_for_limit)
    desired_velocity = (desired - q) / max(float(dt), 1e-6)
    brake_up = overshoot_up & (desired_velocity > 0.0)
    brake_down = overshoot_down & (desired_velocity < 0.0)
    velocity_brake_trigger = torch.full_like(q, velocity_brake_general_trigger)
    if velocity_brake_hip_trigger is not None:
        for index in (0, 3, 6, 9):
            velocity_brake_trigger[..., index] = float(velocity_brake_hip_trigger)
    for index in calf_indices:
        velocity_brake_trigger[..., index] = velocity_brake_calf_trigger
    velocity_brake = (
        bool(velocity_brake_enabled)
        & dynamic_mask
        & (velocity.abs() >= velocity_limit.expand_as(q) * velocity_brake_trigger)
        & (velocity * desired_velocity > 0.0)
    )
    if predictive_all_joints:
        reverse_offset=torch.full_like(q,general_reverse_offset)
        for index in calf_indices:
            reverse_offset[...,index]=calf_reverse_offset
        reverse_target=q-torch.sign(velocity)*reverse_offset
        target=torch.where(brake_up | brake_down | velocity_brake,reverse_target,desired)
    else:
        upward_limit = q + allowed_up * dt
        downward_limit = q - allowed_down * dt
        target = torch.where(brake_up, torch.minimum(desired, upward_limit), desired)
        target = torch.where(brake_down, torch.maximum(target, downward_limit), target)
    # The startup calf rule is a brake trigger, not a permanent target slew.
    startup_active = step < float(startup_steps)
    calf_speed_limit = velocity_limit * calf_scale
    for index in calf_indices:
        same_direction = velocity[..., index] * desired_velocity[..., index] > 0.0
        over_startup = velocity[..., index].abs() > calf_speed_limit[..., index]
        target[..., index] = torch.where(
            startup_active & same_direction & over_startup,
            q[..., index], target[..., index],
        )
    target = torch.maximum(torch.minimum(target, safe_upper), safe_lower)
    modified = (target - desired).abs() > 1e-7
    brake_direction = torch.where(
        brake_up | (velocity_brake & (velocity > 0.0)),
        torch.ones_like(q),
        torch.where(
            brake_down | (velocity_brake & (velocity < 0.0)),
            -torch.ones_like(q),
            torch.zeros_like(q),
        ),
    )
    return target, {
        "brake_distance": brake_distance,
        "stop_position": stop_position,
        "allowed_velocity": torch.where(desired_velocity >= 0.0, allowed_up, allowed_down),
        "modified": modified,
        "brake_direction": brake_direction,
        "velocity_brake": velocity_brake,
    }


def apply_global_impact_guard(
    q, velocity, proposed, lower, upper, velocity_limit,
    guard_counter, *, trigger_ratio=0.75, boundary_margin=0.10,
    cooldown_steps=3, cross_joint_scale=0.35, target_error_threshold=0.25,
    external_trigger=None,
    velocity_delta=None, acceleration_threshold=0.40, contact_leg_trigger=None,
    prearm_counter=None, transfer_counter=None, source_leg_mask=None,
):
    """Dissipate impact locally, per leg, without commanding a reversal."""
    limits = velocity_limit.expand_as(velocity)
    trigger = q.new_full(q.shape[:-1], float(trigger_ratio))
    folded_trigger = float(os.environ.get("FRNET_GLOBAL_IMPACT_GUARD_FOLDED_TRIGGER", str(trigger_ratio)))
    folded = (q[..., THIGH_INDICES] > 2.50) & (q[..., CALF_INDICES] < -2.45)
    folded_min_legs=int(os.environ.get("FRNET_GLOBAL_IMPACT_GUARD_FOLDED_MIN_LEGS", "3"))
    deeply_folded = folded.sum(dim=-1) >= folded_min_legs
    pair=os.environ.get("FRNET_GLOBAL_IMPACT_GUARD_FOLDED_PAIR", "").upper()
    pair_map={"FL_RR":(0,3),"FR_RL":(1,2),"FL_FR":(0,1),"RL_RR":(2,3)}
    if pair in pair_map:
        a,b=pair_map[pair]
        deeply_folded=folded[...,a] & folded[...,b]
    trigger = torch.where(deeply_folded, q.new_full(trigger.shape, folded_trigger), trigger)
    desired_velocity = proposed - q
    driving = velocity * desired_velocity > 0.0
    fast = velocity.abs() >= limits * trigger.unsqueeze(-1)
    near_upper = (q >= upper - float(boundary_margin)) & (velocity > 0.50)
    near_lower = (q <= lower + float(boundary_margin)) & (velocity < -0.50)
    # A large policy target is common during unfolding and is not, by itself,
    # evidence of an impact.  In event mode, identify the source joint from
    # several signals: it must be moving in the commanded direction and show
    # either a sudden acceleration, boundary approach, or genuinely high
    # speed.  This avoids treating static support or a neighboring leg's
    # rebound as the impact source.
    sudden_outward = torch.zeros_like(fast)
    if velocity_delta is not None:
        sudden_outward = (
            (velocity * velocity_delta > 0.0)
            & (velocity_delta.abs() >= float(acceleration_threshold))
        )
    event_mode = os.environ.get("FRNET_GLOBAL_IMPACT_EVENT_MODE", "0") == "1"
    if event_mode:
        speed_ratio = velocity.abs() / limits.clamp_min(1e-6)
        min_ratio = float(os.environ.get("FRNET_GLOBAL_IMPACT_EVENT_MIN_RATIO", "0.15"))
        high_ratio = float(os.environ.get("FRNET_GLOBAL_IMPACT_EVENT_HIGH_RATIO", "0.60"))
        acceleration_event = sudden_outward & (speed_ratio >= min_ratio)
        boundary_event = (near_upper | near_lower) & driving
        high_speed_event = fast & driving & (speed_ratio >= high_ratio)
        trigger_joint = (acceleration_event | boundary_event | high_speed_event) & driving
        # Event mode has its own evidence-qualified trigger.  Keep the
        # downstream dissipation path well-defined instead of referring to
        # the legacy hard trigger that is only built in the other branch.
        hard_trigger_joint = trigger_joint
    else:
        hard_trigger_joint = fast | near_upper | near_lower
    causal_mode = os.environ.get("FRNET_GLOBAL_IMPACT_CAUSAL_MODE", "0") == "1"
    if causal_mode and contact_leg_trigger is not None:
        # In causal mode, a fast neighboring joint is only eligible for the
        # hard guard after the source-leg classifier has qualified that leg.
        # This prevents an early generic speed threshold from freezing the
        # wrong leg before the actual impact chain is observed.
        causal_joint = contact_leg_trigger.to(dtype=torch.bool).repeat_interleave(3, dim=-1)
        hard_trigger_joint = hard_trigger_joint & causal_joint
    predictive_enabled = os.environ.get(
        "FRNET_GLOBAL_IMPACT_PREDICTIVE_STOP", "0"
    ) == "1"
    if predictive_enabled and velocity_delta is not None and contact_leg_trigger is not None:
        # Predict the first position reached while braking.  This is used only
        # after the source leg has been confirmed, so ordinary static support
        # cannot activate it.  The estimate is intentionally conservative:
        # measured acceleration is bounded below by a configurable braking
        # capability instead of assuming an ideal instantaneous stop.
        prediction_dt = float(os.environ.get(
            "FRNET_GLOBAL_IMPACT_PREDICTIVE_DT", "0.02"
        ))
        brake_accel = float(os.environ.get(
            "FRNET_GLOBAL_IMPACT_PREDICTIVE_ACCEL", "1.5"
        ))
        brake_accel = max(brake_accel, 1e-3)
        projected = q + velocity * prediction_dt + (
            velocity.square() / (2.0 * brake_accel)
        ) * torch.sign(velocity)
        projected_outward = (
            ((velocity > 0.0) & (projected > upper - float(boundary_margin)))
            | ((velocity < 0.0) & (projected < lower + float(boundary_margin)))
        )
        predictive_joint = (
            projected_outward
            & driving
            & contact_leg_trigger.to(dtype=torch.bool).repeat_interleave(3, dim=-1)
        )
        if os.environ.get("FRNET_GLOBAL_IMPACT_PREDICTIVE_CALF_ONLY", "0") == "1":
            calf_only = torch.zeros_like(predictive_joint)
            calf_only[..., CALF_INDICES] = True
            predictive_joint = predictive_joint & calf_only
        hard_trigger_joint = hard_trigger_joint | predictive_joint
    soft_accel_enabled = os.environ.get("FRNET_GLOBAL_IMPACT_SOFT_ACCEL", "0") == "1"
    soft_accel_joint = sudden_outward & (~hard_trigger_joint)
    if os.environ.get("FRNET_GLOBAL_IMPACT_SOFT_ACCEL_CALF_ONLY", "0") == "1":
        calf_only = torch.zeros_like(soft_accel_joint)
        calf_only[..., CALF_INDICES] = True
        soft_accel_joint = soft_accel_joint & calf_only
    # Optional source-leg filter keeps a stronger early guard local.  This is
    # useful when one source leg is delay-sensitive while another leg's soft
    # guard would interfere with a valid recovery transfer.
    soft_legs = os.environ.get("FRNET_GLOBAL_IMPACT_SOFT_ACCEL_LEGS", "").upper()
    if soft_legs:
        leg_index = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}
        mask = torch.zeros_like(soft_accel_joint)
        for name in soft_legs.split(","):
            idx = leg_index.get(name.strip())
            if idx is not None:
                mask[..., idx * 3:(idx + 1) * 3] = True
        soft_accel_joint = soft_accel_joint & mask
    soft_trigger_joint = soft_accel_joint if soft_accel_enabled else torch.zeros_like(soft_accel_joint)
    trigger_joint = hard_trigger_joint | soft_trigger_joint
    leg_trigger = torch.stack(
        [trigger_joint[..., 0:3].any(dim=-1),
         trigger_joint[..., 3:6].any(dim=-1),
         trigger_joint[..., 6:9].any(dim=-1),
         trigger_joint[..., 9:12].any(dim=-1)], dim=-1,
    )
    if contact_leg_trigger is not None:
        leg_trigger = leg_trigger | contact_leg_trigger.to(dtype=torch.bool)
    # Contact-triggered activation is already leg-local; joint evidence above
    # remains a fallback when a contact sensor is unavailable.
    next_counter = torch.where(
        leg_trigger,
        torch.full_like(guard_counter, int(cooldown_steps)),
        torch.clamp(guard_counter - 1, min=0),
    )
    # Hysteresis: a fixed cooldown can release while a guarded leg is still
    # moving into the same target.  Keep that leg latched until its speed has
    # fallen below a release ratio or its target no longer drives motion.
    release_ratio = float(os.environ.get(
        "FRNET_GLOBAL_IMPACT_GUARD_RELEASE_RATIO", "0.0"
    ))
    active_previous = guard_counter > 0
    release_speed = limits * max(0.0, release_ratio)
    release_hold_joint = (
        active_previous.repeat_interleave(3, dim=-1)
        & driving
        & (velocity.abs() > release_speed)
    )
    release_hold_leg = torch.stack(
        [release_hold_joint[..., 0:3].any(dim=-1),
         release_hold_joint[..., 3:6].any(dim=-1),
         release_hold_joint[..., 6:9].any(dim=-1),
         release_hold_joint[..., 9:12].any(dim=-1)], dim=-1,
    )
    next_counter = torch.where(
        release_hold_leg,
        torch.maximum(next_counter, torch.full_like(next_counter, int(cooldown_steps))),
        next_counter,
    )
    # Optional source-to-receiver latch: a confirmed source impact can keep
    # a specific neighboring leg in a short dissipative window even before
    # that receiver exceeds its own hard threshold.  This is intentionally
    # pair-specific so it does not freeze every leg during static support.
    latch_pairs = os.environ.get(
        "FRNET_GLOBAL_IMPACT_TRANSFER_LATCH_PAIRS", ""
    ).upper()
    latch_leg = torch.zeros_like(next_counter, dtype=torch.bool)
    if latch_pairs and os.environ.get("FRNET_GLOBAL_IMPACT_TRANSFER_GUARD", "0") == "1":
        source_latch = (guard_counter > 0)
        if contact_leg_trigger is not None:
            source_latch = source_latch | contact_leg_trigger.to(dtype=torch.bool)
        leg_index = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}
        for edge in latch_pairs.split(","):
            try:
                source_name, receiver_name = edge.split(">", 1)
                source_idx = leg_index[source_name.strip()]
                receiver_idx = leg_index[receiver_name.strip()]
            except (KeyError, ValueError):
                continue
            latch_leg[..., receiver_idx] = latch_leg[..., receiver_idx] | source_latch[..., source_idx]
            latch_hold_steps = int(os.environ.get(
                "FRNET_GLOBAL_IMPACT_TRANSFER_LATCH_HOLD_STEPS", str(cooldown_steps)
            ))
            next_counter[..., receiver_idx] = torch.where(
                source_latch[..., source_idx],
                torch.maximum(
                    next_counter[..., receiver_idx],
                    torch.full_like(next_counter[..., receiver_idx], latch_hold_steps),
                ),
                next_counter[..., receiver_idx],
            )
    active = next_counter > 0
    # Keep a separate, receiver-only latch.  The source leg should be
    # released on the short local cooldown, while a delayed load transfer to
    # a neighboring leg may still need a longer prediction window.
    if transfer_counter is not None:
        transfer_hold_steps = int(os.environ.get(
            "FRNET_GLOBAL_IMPACT_TRANSFER_HOLD_STEPS", str(cooldown_steps)
        ))
        transfer_source = leg_trigger
        if contact_leg_trigger is not None:
            transfer_source = transfer_source | contact_leg_trigger.to(dtype=torch.bool)
        next_transfer_counter = torch.where(
            transfer_source,
            torch.full_like(transfer_counter, transfer_hold_steps),
            torch.clamp(transfer_counter - 1, min=0),
        )
        transfer_counter.copy_(next_transfer_counter)
    active_joint = torch.cat(
        [active[..., 0:1].expand(-1, 3), active[..., 1:2].expand(-1, 3),
         active[..., 2:3].expand(-1, 3), active[..., 3:4].expand(-1, 3)], dim=-1,
    )
    # The triggering joint is fully dissipative.  Other joints on that same
    # leg only shrink their same-direction target step; other legs are free.
    scale = q.new_full(q.shape, float(cross_joint_scale))
    scale = torch.where(hard_trigger_joint, torch.zeros_like(q), scale)
    if os.environ.get("FRNET_GLOBAL_IMPACT_PRESERVE_HIP", "0") == "1":
        hip_mask = torch.zeros_like(scale, dtype=torch.bool)
        hip_mask[..., HIP_INDICES] = True
        scale = torch.where(
            hip_mask & (~hard_trigger_joint),
            torch.ones_like(scale),
            scale,
        )
    if soft_accel_enabled:
        soft_scale=float(os.environ.get("FRNET_GLOBAL_IMPACT_SOFT_ACCEL_SCALE", "0.50"))
        soft_scale_vec = q.new_full(q.shape, soft_scale)
        leg_names = ("FL", "FR", "RL", "RR")
        for leg_idx, leg_name in enumerate(leg_names):
            override = os.environ.get(
                f"FRNET_GLOBAL_IMPACT_SOFT_ACCEL_SCALE_{leg_name}", ""
            )
            if override:
                soft_scale_vec[..., leg_idx * 3:(leg_idx + 1) * 3] = float(override)
        scale = torch.where(soft_accel_joint, soft_scale_vec, scale)
    else:
        scale = torch.where(soft_accel_joint, torch.zeros_like(q), scale)
    guarded_delta = (proposed - q) * scale
    guarded = torch.where(active_joint & driving, q + guarded_delta, proposed)
    # A latched receiver uses joint-specific dissipation: calf is stopped,
    # thigh follows only a small fraction, and hip retains posture authority.
    if latch_leg.any():
        latch_scale = q.new_full(q.shape, float(os.environ.get(
            "FRNET_GLOBAL_IMPACT_TRANSFER_LATCH_SCALE", "0.25"
        )))
        latch_scale[..., HIP_INDICES] = float(os.environ.get(
            "FRNET_GLOBAL_IMPACT_TRANSFER_LATCH_HIP_SCALE", "0.75"
        ))
        latch_scale[..., CALF_INDICES] = float(os.environ.get(
            "FRNET_GLOBAL_IMPACT_TRANSFER_LATCH_CALF_SCALE", "0.0"
        ))
        latch_joint = latch_leg.repeat_interleave(3, dim=-1)
        guarded = torch.where(
            latch_joint & driving, q + (proposed - q) * latch_scale, guarded
        )
    # Optional pre-impact damping.  The hard causal guard intentionally waits
    # for a qualified source leg, but that can be several policy steps after
    # the first acceleration spike.  A small target reduction gives the joint
    # braking distance time to act without freezing the leg or changing the
    # diagnostic source label.
    prearm_enabled = os.environ.get(
        "FRNET_GLOBAL_IMPACT_PREARM", "0"
    ) == "1"
    if prearm_enabled:
        prearm_min_ratio = float(os.environ.get(
            "FRNET_GLOBAL_IMPACT_PREARM_MIN_RATIO", "0.15"
        ))
        prearm_accel = float(os.environ.get(
            "FRNET_GLOBAL_IMPACT_PREARM_ACCEL_THRESHOLD", "0.25"
        ))
        prearm_scale = float(os.environ.get(
            "FRNET_GLOBAL_IMPACT_PREARM_SCALE", "0.85"
        ))
        prearm_hip_scale = float(os.environ.get(
            "FRNET_GLOBAL_IMPACT_PREARM_HIP_SCALE", str(prearm_scale)
        ))
        speed_ratio = velocity.abs() / limits.clamp_min(1e-6)
        prearm_motion = (
            driving
            & (speed_ratio >= prearm_min_ratio)
            & sudden_outward
            & (velocity_delta.abs() >= prearm_accel)
        )
        prearm_distance = velocity.abs() * float(os.environ.get(
            "FRNET_GLOBAL_IMPACT_PREARM_DT", "0.04"
        )) + velocity.square() / (
            2.0 * float(os.environ.get("FRNET_GLOBAL_IMPACT_PREARM_BRAKE", "1.5"))
        )
        prearm_outward = (
            ((velocity > 0.0) & (q + prearm_distance > upper - float(boundary_margin)))
            | ((velocity < 0.0) & (q - prearm_distance < lower + float(boundary_margin)))
        )
        prearm_joint = prearm_motion & prearm_outward & (~active_joint)
        # Deep folded starts already use the static-library equilibrium and
        # their first-step dynamics are different.  Keep the aggressive
        # predictive setting for ordinary transfers, while allowing a caller
        # to retain the conservative deep-folded path.
        if os.environ.get("FRNET_GLOBAL_IMPACT_PREARM_EXCLUDE_DEEP_FOLDED", "0") == "1":
            prearm_joint = prearm_joint & (~deeply_folded.unsqueeze(-1))
            if os.environ.get("FRNET_GLOBAL_IMPACT_PREARM_INCLUDE_CALF", "0") != "1":
                prearm_joint[..., CALF_INDICES] = False
            else:
                calf_legs = os.environ.get(
                    "FRNET_GLOBAL_IMPACT_PREARM_INCLUDE_CALF_LEGS", "FL,FR,RL,RR"
                ).upper()
                allowed = torch.zeros_like(prearm_joint)
                for leg_idx, leg_name in enumerate(("FL", "FR", "RL", "RR")):
                    if leg_name in {name.strip() for name in calf_legs.split(",")}: 
                        allowed[..., leg_idx * 3 + 2] = True
                prearm_joint[..., CALF_INDICES] = prearm_joint[..., CALF_INDICES] & allowed[..., CALF_INDICES]
        if prearm_counter is not None:
            prearm_hold_steps = int(os.environ.get(
                "FRNET_GLOBAL_IMPACT_PREARM_HOLD_STEPS", "3"
            ))
            prearm_release_ratio = float(os.environ.get(
                "FRNET_GLOBAL_IMPACT_PREARM_RELEASE_RATIO", "0.50"
            ))
            release_ratio_vec = q.new_full(q.shape, prearm_release_ratio)
            release_ratio_vec[..., CALF_INDICES] = float(os.environ.get(
                "FRNET_GLOBAL_IMPACT_PREARM_RELEASE_CALF_RATIO",
                str(prearm_release_ratio),
            ))
            release_ratio_vec[..., THIGH_INDICES] = float(os.environ.get(
                "FRNET_GLOBAL_IMPACT_PREARM_RELEASE_THIGH_RATIO",
                str(prearm_release_ratio),
            ))
            release_ratio_vec[..., HIP_INDICES] = float(os.environ.get(
                "FRNET_GLOBAL_IMPACT_PREARM_RELEASE_HIP_RATIO",
                str(prearm_release_ratio),
            ))
            previous_prearm = prearm_counter > 0
            release = (~driving) | (
                velocity.abs() <= limits * release_ratio_vec
            )
            next_prearm = torch.where(
                prearm_joint,
                torch.full_like(prearm_counter, prearm_hold_steps),
                torch.clamp(prearm_counter - 1, min=0),
            )
            next_prearm = torch.where(
                previous_prearm & (~release),
                torch.maximum(
                    next_prearm,
                    torch.full_like(prearm_counter, prearm_hold_steps),
                ),
                next_prearm,
            )
            prearm_counter.copy_(next_prearm)
            prearm_joint = prearm_joint | (next_prearm > 0)
        prearm_joint_scale = q.new_full(q.shape, prearm_scale)
        prearm_joint_scale[..., HIP_INDICES] = prearm_hip_scale
        if os.environ.get("FRNET_GLOBAL_IMPACT_PREARM_ADAPTIVE", "0") == "1":
            safe_up = (upper - float(boundary_margin) - q).clamp_min(1e-4)
            safe_down = (q - (lower + float(boundary_margin))).clamp_min(1e-4)
            safe_distance = torch.where(velocity >= 0.0, safe_up, safe_down)
            stop_distance = velocity.abs() * float(os.environ.get(
                "FRNET_GLOBAL_IMPACT_PREARM_DT", "0.04"
            )) + velocity.square() / (
                2.0 * float(os.environ.get("FRNET_GLOBAL_IMPACT_PREARM_BRAKE", "1.5"))
            )
            overshoot_fraction = (
                (stop_distance - safe_distance) / safe_distance
            ).clamp_min(0.0).clamp_max(1.0)
            adaptive_gain = float(os.environ.get(
                "FRNET_GLOBAL_IMPACT_PREARM_ADAPTIVE_GAIN", "0.50"
            ))
            adaptive_min = float(os.environ.get(
                "FRNET_GLOBAL_IMPACT_PREARM_ADAPTIVE_MIN_SCALE", "0.65"
            ))
            adaptive_scale = (
                prearm_joint_scale - adaptive_gain * overshoot_fraction
            ).clamp_min(adaptive_min).clamp_max(1.0)
            prearm_joint_scale = torch.where(
                prearm_joint.unsqueeze(-1) if prearm_joint.ndim == q.ndim - 1 else prearm_joint,
                adaptive_scale,
                prearm_joint_scale,
            )
        prearm_target = q + (proposed - q) * prearm_joint_scale
        guarded = torch.where(prearm_joint, prearm_target, guarded)
    # A source leg can transfer a rebound through the body before the
    # receiving leg reaches the hard speed threshold.  Treat that receiving
    # leg as a short-lived, target-rate-only intervention.  Static contact
    # cannot trigger it because it requires an already active source leg,
    # same-direction acceleration, and a configurable multi-frame threshold.
    transfer_enabled = os.environ.get("FRNET_GLOBAL_IMPACT_TRANSFER_GUARD", "0") == "1"
    # Returned to the action term so the actuator substep interlock can apply
    # the same source-to-receiver decision before the next policy frame.
    transfer_leg = torch.zeros_like(next_counter, dtype=torch.bool)
    if transfer_enabled and velocity_delta is not None:
        transfer_threshold=float(os.environ.get("FRNET_GLOBAL_IMPACT_TRANSFER_ACCEL_THRESHOLD", "0.40"))
        require_previous=os.environ.get("FRNET_GLOBAL_IMPACT_TRANSFER_REQUIRE_PREVIOUS", "1") == "1"
        source_state = (
            transfer_counter > 0 if transfer_counter is not None else guard_counter > 0
        )
        source_present=(source_state.any(dim=-1, keepdim=True)
                        if require_previous else active.any(dim=-1, keepdim=True))
        transfer_speed_ratio = float(os.environ.get(
            "FRNET_GLOBAL_IMPACT_TRANSFER_SPEED_RATIO", "0.0"
        ))
        limits_safe = limits.clamp_min(1e-6)
        speed_transfer = (
            (transfer_speed_ratio > 0.0)
            & (velocity.abs() >= limits_safe * transfer_speed_ratio)
        )
        transfer_accel = (
            (velocity * velocity_delta > 0.0)
            & (velocity_delta.abs() >= transfer_threshold)
        )
        transfer_driving = velocity * (proposed - q) > 0.0
        transfer_joint=(
            source_present
            & transfer_driving
            & (transfer_accel | speed_transfer)
            & (~active_joint)
        )
        # Preserve hip posture authority on the receiving leg.  Only its
        # thigh/calf continuation is rate-limited by transferred impact.
        include_transfer_hip = os.environ.get(
            "FRNET_GLOBAL_IMPACT_TRANSFER_INCLUDE_HIP", "0"
        ) == "1"
        if not include_transfer_hip:
            hip_mask=torch.zeros_like(transfer_joint)
            hip_mask[..., HIP_INDICES]=True
            transfer_joint=transfer_joint & (~hip_mask)
        transfer_leg=torch.stack(
            [transfer_joint[...,0:3].any(dim=-1),transfer_joint[...,3:6].any(dim=-1),
             transfer_joint[...,6:9].any(dim=-1),transfer_joint[...,9:12].any(dim=-1)],dim=-1,
        )
        # Body coupling is strongest across the same lateral pair.  Do not
        # damp the two unrelated legs merely because they are accelerating.
        source_active = active | (guard_counter > 0)
        if os.environ.get("FRNET_GLOBAL_IMPACT_TRANSFER_USE_CLASSIFIED_SOURCE", "0") == "1" and source_leg_mask is not None:
            # Use only the temporally qualified source-leg classifier.  This
            # prevents a static folded support leg from becoming a transfer
            # source merely because another guard is active.
            source_active = source_leg_mask.to(dtype=torch.bool)
        elif os.environ.get("FRNET_GLOBAL_IMPACT_TRANSFER_USE_SOURCE", "0") == "1":
            if contact_leg_trigger is not None:
                source_active = source_active | contact_leg_trigger.to(dtype=torch.bool)
        lateral_source=torch.stack(
            [source_active[...,1],source_active[...,0],source_active[...,3],source_active[...,2]],dim=-1,
        )
        longitudinal_source=torch.stack(
            [source_active[...,2],source_active[...,3],source_active[...,0],source_active[...,1]],dim=-1,
        )
        neighbor_mode=os.environ.get(
            "FRNET_GLOBAL_IMPACT_TRANSFER_NEIGHBOR_MODE", "lateral"
        ).lower()
        if neighbor_mode == "longitudinal":
            pair_source=longitudinal_source
        elif neighbor_mode == "all_adjacent":
            pair_source=lateral_source | longitudinal_source
        else:
            pair_source=lateral_source
        # Allow a small, evidence-driven set of extra body-coupling edges
        # without enabling every geometric neighbor.  Syntax is a comma
        # separated list such as ``RR>FL,RR>FR`` (source>receiving).
        extra_edges=os.environ.get(
            "FRNET_GLOBAL_IMPACT_TRANSFER_EXTRA_PAIRS", ""
        ).upper()
        if extra_edges:
            leg_index={"FL":0,"FR":1,"RL":2,"RR":3}
            for edge in extra_edges.split(","):
                try:
                    source_name, receiving_name=edge.split(">",1)
                    source_idx=leg_index[source_name.strip()]
                    receiving_idx=leg_index[receiving_name.strip()]
                except (KeyError, ValueError):
                    continue
                folded_extra = os.environ.get(
                    "FRNET_GLOBAL_IMPACT_TRANSFER_FOLDED_ONLY", "0"
                ) == "1"
                edge_source = source_active[..., source_idx]
                if folded_extra:
                    edge_source = edge_source & deeply_folded
                pair_source[..., receiving_idx] = (
                    pair_source[..., receiving_idx] | edge_source
                )
        transfer_leg=transfer_leg & pair_source
        pair_source_joint = pair_source.repeat_interleave(3, dim=-1)
        transfer_joint = transfer_joint & pair_source_joint
        # Keep the directly evidenced receiving joint causal.  Coordinate
        # only thigh/calf on that same leg when requested; do not blindly
        # damp the hip or all three joints because one neighbor accelerated.
        transfer_joint_all = transfer_joint.clone()
        if os.environ.get("FRNET_GLOBAL_IMPACT_TRANSFER_COORDINATE", "1") == "1":
            for indices in (THIGH_INDICES, CALF_INDICES):
                leg_has_event = torch.stack(
                    [transfer_joint[...,0:3].any(dim=-1),
                     transfer_joint[...,3:6].any(dim=-1),
                     transfer_joint[...,6:9].any(dim=-1),
                     transfer_joint[...,9:12].any(dim=-1)], dim=-1,
                )
                transfer_joint_all[..., indices] = transfer_joint_all[..., indices] | torch.stack(
                    [leg_has_event[...,0], leg_has_event[...,1],
                     leg_has_event[...,2], leg_has_event[...,3]], dim=-1
                ).repeat_interleave(1, dim=-1).reshape(transfer_joint_all[..., indices].shape)
        transfer_scale=float(os.environ.get("FRNET_GLOBAL_IMPACT_TRANSFER_SCALE", "0.50"))
        # Optional source-to-receiver scale overrides keep the intervention
        # local.  A smaller scale means a stronger dissipative target reduction.
        transfer_scale_vec = q.new_full(q.shape, transfer_scale)
        leg_index = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}
        for edge in extra_edges.split(",") if extra_edges else ():
            try:
                source_name, receiving_name = edge.split(">", 1)
                source_idx = leg_index[source_name.strip()]
                receiving_idx = leg_index[receiving_name.strip()]
                override = os.environ.get(
                    f"FRNET_GLOBAL_IMPACT_TRANSFER_SCALE_{source_name.strip()}_{receiving_name.strip()}",
                    "",
                )
                if not override:
                    continue
                edge_source = source_active[..., source_idx]
                receiver_joint = torch.zeros_like(transfer_joint_all)
                receiver_joint[..., receiving_idx * 3:(receiving_idx + 1) * 3] = edge_source.unsqueeze(-1)
                transfer_scale_vec = torch.where(
                    receiver_joint,
                    q.new_full(q.shape, float(override)),
                    transfer_scale_vec,
                )
            except (KeyError, ValueError):
                continue
        # Optional per-joint edge overrides let a confirmed source leg brake
        # only the receiver hip, thigh, or calf.  This avoids reducing the
        # whole receiving leg when one joint needs extra damping.
        for source_name, source_idx in leg_index.items():
            for receiving_name, receiving_idx in leg_index.items():
                edge_source = source_active[..., source_idx]
                for joint_name, joint_offset in (("HIP", 0), ("THIGH", 1), ("CALF", 2)):
                    key = f"FRNET_GLOBAL_IMPACT_TRANSFER_SCALE_{source_name}_{receiving_name}_{joint_name}"
                    override = os.environ.get(key, "")
                    if not override:
                        continue
                    receiver_joint = torch.zeros_like(transfer_joint_all)
                    receiver_joint[..., receiving_idx * 3 + joint_offset] = edge_source
                    transfer_scale_vec = torch.where(
                        receiver_joint,
                        q.new_full(q.shape, float(override)),
                        transfer_scale_vec,
                    )
        fallback_hip_scale = float(os.environ.get(
            "FRNET_GLOBAL_IMPACT_TRANSFER_FALLBACK_HIP_SCALE", "0.0"
        ))
        if fallback_hip_scale > 0.0:
            fallback_ratio = float(os.environ.get(
                "FRNET_GLOBAL_IMPACT_TRANSFER_FALLBACK_HIP_RATIO", "0.80"
            ))
            # This is a last-resort receiver guard: it requires an already
            # qualified impact window, then only slows a near-limit hip.  It
            # does not infer a new source from static contact.
            fallback_receiver_mode = os.environ.get(
                "FRNET_GLOBAL_IMPACT_TRANSFER_FALLBACK_HIP_RECEIVERS", "0"
            ) == "1"
            fallback_leg = pair_source if fallback_receiver_mode else (active | (guard_counter > 0))
            fallback_joint = fallback_leg.repeat_interleave(3, dim=-1)
            fallback_joint[..., THIGH_INDICES] = False
            fallback_joint[..., CALF_INDICES] = False
            fallback_ignore_direction = os.environ.get(
                "FRNET_GLOBAL_IMPACT_TRANSFER_FALLBACK_HIP_IGNORE_DIRECTION", "0"
            ) == "1"
            if not fallback_ignore_direction:
                fallback_joint = fallback_joint & driving
            fallback_joint = fallback_joint & (~active_joint)
            fallback_joint = fallback_joint & (
                velocity.abs() >= limits * fallback_ratio
            )
            transfer_scale_vec = torch.where(
                fallback_joint,
                q.new_full(q.shape, fallback_hip_scale),
                transfer_scale_vec,
            )
        fallback_calf_scale = float(os.environ.get(
            "FRNET_GLOBAL_IMPACT_TRANSFER_FALLBACK_CALF_SCALE", "0.0"
        ))
        if fallback_calf_scale > 0.0:
            fallback_calf_ratio = float(os.environ.get(
                "FRNET_GLOBAL_IMPACT_TRANSFER_FALLBACK_CALF_RATIO", "0.90"
            ))
            fallback_calf_receivers = os.environ.get(
                "FRNET_GLOBAL_IMPACT_TRANSFER_FALB_RECEIVERS", "1"
            ) == "1"
            calf_leg = pair_source if fallback_calf_receivers else (active | (guard_counter > 0))
            calf_joint = calf_leg.repeat_interleave(3, dim=-1)
            calf_joint[..., HIP_INDICES] = False
            calf_joint[..., THIGH_INDICES] = False
            calf_joint = calf_joint & (~active_joint)
            calf_joint = calf_joint & (velocity.abs() >= limits * fallback_calf_ratio)
            if os.environ.get(
                "FRNET_GLOBAL_IMPACT_TRANSFER_FALLBACK_CALF_IGNORE_DIRECTION", "1"
            ) != "1":
                calf_joint = calf_joint & driving
            transfer_scale_vec = torch.where(
                calf_joint,
                q.new_full(q.shape, fallback_calf_scale),
                transfer_scale_vec,
            )
        transfer_delta=(proposed-q)*transfer_scale_vec
        guarded=torch.where(transfer_joint_all & driving,q+transfer_delta,guarded)
    guarded = torch.maximum(torch.minimum(guarded, upper), lower)
    return guarded, next_counter, active, leg_trigger, transfer_counter, transfer_leg

@dataclass(kw_only=True)
class RecoveryActionCfg(ActionTermCfg):
    source_history_compatibility: bool = True
    stand_hold_enabled: bool = False
    staged_unfold_enabled: bool = False
    predictive_all_joints: bool = False
    predictive_general_trigger: float = 0.50
    predictive_calf_trigger: float = 0.30
    predictive_general_reverse_offset: float = 0.015
    predictive_calf_reverse_offset: float = 0.030
    predictive_brake_release_ratio: float = 0.20
    predictive_latch_enabled: bool = True
    velocity_brake_enabled: bool = False
    velocity_brake_general_trigger: float = 0.80
    velocity_brake_calf_trigger: float = 0.70
    safety_general_margin: float = 0.020
    safety_calf_margin: float = 0.040
    initial_target_hold_steps: int = 0
    global_impact_guard_enabled: bool = False
    global_impact_guard_trigger: float = 0.75
    global_impact_guard_margin: float = 0.10
    global_impact_guard_cooldown_steps: int = 3
    global_impact_guard_cross_joint_scale: float = 1.0
    global_impact_guard_target_error: float = 0.25

    def build(self, env):
        return RecoveryAction(self, env)

class RecoveryAction(ActionTerm):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        if tuple(self._entity.joint_names) != JOINT_NAMES:
            raise ValueError('Recovery action joint ordering mismatch')
        self.delay=ActionDelay(env.num_envs,env.device)
        self.current=torch.zeros((env.num_envs,12),device=env.device)
        self.filtered_action=torch.zeros_like(self.current)
        self.contact_state=torch.zeros((env.num_envs,4),dtype=torch.bool,device=env.device)
        self.nonfoot_contact_force_previous = None
        self.nonfoot_contact_leg_map = None
        try:
            sensor = env.scene["nonfoot_ground_touch"]
            force_names = [slot.primary_name for slot in sensor._slots if slot.field_name == "force"]
            leg_map = torch.zeros((len(force_names), 4), dtype=torch.bool, device=env.device)
            for i, name in enumerate(force_names):
                for leg_idx, leg in enumerate(("FL", "FR", "RL", "RR")):
                    if name.startswith(leg + "_"):
                        leg_map[i, leg_idx] = True
                        break
            self.nonfoot_contact_leg_map = leg_map
            self.nonfoot_contact_force_previous = torch.zeros(
                (env.num_envs, len(force_names)), dtype=torch.float32, device=env.device
            )
            # A reset may begin with a body part already pressing on the
            # ground.  Such contact is static support, not a collision.  A
            # sensor is allowed to report a new impact only after it has
            # first unloaded below the release threshold.
            self.nonfoot_contact_armed = torch.zeros(
                (env.num_envs, len(force_names)), dtype=torch.bool, device=env.device
            )
        except Exception:
            self.nonfoot_contact_force_previous = None
            self.nonfoot_contact_leg_map = None
            self.nonfoot_contact_armed = None
        self.stable_counter=torch.zeros(env.num_envs,dtype=torch.int32,device=env.device)
        self.destabilized_counter=torch.zeros(env.num_envs,dtype=torch.int32,device=env.device)
        self.hold_mode=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
        self.recovery_complete=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
        # 0 = classify, 1 = slow unfold, 2 = blended handoff, 3 = policy.
        self.recovery_phase=torch.zeros(env.num_envs,dtype=torch.int8,device=env.device)
        self.unfold_ready_counter=torch.zeros(env.num_envs,dtype=torch.int32,device=env.device)
        self.unfold_steps=torch.zeros(env.num_envs,dtype=torch.int32,device=env.device)
        self.handoff_steps=torch.zeros(env.num_envs,dtype=torch.int32,device=env.device)
        self.previous=self.current.clone()
        self.previous_previous=self.current.clone()
        self.target=self._entity.data.default_joint_pos.clone()
        self.position_lower=torch.tensor(
            POSITION_LOWER,device=env.device,dtype=self.target.dtype
        )
        self.position_upper=torch.tensor(
            POSITION_UPPER,device=env.device,dtype=self.target.dtype
        )
        target_rate_scale=float(os.environ.get('FRNET_ACTION_TARGET_RATE_SCALE','0.15'))
        if not 0.01 <= target_rate_scale <= 0.50:
            raise ValueError('FRNET_ACTION_TARGET_RATE_SCALE must be in [0.01, 0.50]')
        self.max_target_delta=torch.tensor(
            VELOCITY_LIMIT,device=env.device,dtype=self.target.dtype
        )*(target_rate_scale*env.step_dt)
        self.stand_target=self.target.new_tensor(STAND_JOINT_TARGET)
        self.hold_stand_target=self.stand_target.clone()
        self.hold_stand_target[[2,5,8,11]]+=0.05
        brake_accel_scale=float(os.environ.get("FRNET_BRAKE_ACCEL_SCALE", "1.0"))
        if not 0.10 <= brake_accel_scale <= 1.50:
            raise ValueError("FRNET_BRAKE_ACCEL_SCALE must be in [0.10, 1.50]")
        self.brake_accel=self.target.new_tensor((35.,35.,18.,35.,35.,18.,35.,35.,18.,35.,35.,18.)) * brake_accel_scale
        self.safety_margin=self.target.new_tensor((
            cfg.safety_general_margin,cfg.safety_general_margin,cfg.safety_calf_margin,
            cfg.safety_general_margin,cfg.safety_general_margin,cfg.safety_calf_margin,
            cfg.safety_general_margin,cfg.safety_general_margin,cfg.safety_calf_margin,
            cfg.safety_general_margin,cfg.safety_general_margin,cfg.safety_calf_margin,
        ))
        self.safety_step=torch.zeros(env.num_envs,dtype=torch.int32,device=env.device)
        self.initial_target_hold_steps=torch.full(
            (env.num_envs,), int(cfg.initial_target_hold_steps),
            dtype=torch.int32, device=env.device,
        )
        self.safety_modified=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
        self.safety_brake_distance=torch.zeros_like(self.target)
        self.safety_allowed_velocity=torch.zeros_like(self.target)
        self.predictive_brake_latch=torch.zeros_like(self.target)
        self.global_guard_counter=torch.zeros((env.num_envs,4),dtype=torch.int32,device=env.device)
        self.global_transfer_counter=torch.zeros_like(self.global_guard_counter)
        self.prearm_counter=torch.zeros_like(self.target,dtype=torch.int32)
        self.reset_start_target=self.target.clone()
        self.reset_reference_target=self.target.clone()
        self.impact_source_streak=torch.zeros((env.num_envs,4),dtype=torch.int32,device=env.device)
        self.impact_source_leg=torch.full((env.num_envs,),-1,dtype=torch.int32,device=env.device)
        self.impact_source_joint=torch.full((env.num_envs,),-1,dtype=torch.int32,device=env.device)
        self.impact_source_score=torch.zeros((env.num_envs,),device=env.device)
        self.impact_source_trigger=torch.zeros((env.num_envs,),dtype=torch.bool,device=env.device)
        self.impact_source_latched_leg=torch.full((env.num_envs,),-1,dtype=torch.int32,device=env.device)
        self.impact_source_latch_counter=torch.zeros((env.num_envs,),dtype=torch.int32,device=env.device)
        self.impact_detection_age=torch.zeros(env.num_envs,dtype=torch.int32,device=env.device)
        self.previous_joint_velocity=torch.zeros_like(self.target)
        # Keep a short leg-local emergency latch so a detected impact is not
        # released on the very next frame while inertia is still reversing.
        self.emergency_leg_latch=torch.zeros((env.num_envs,4),dtype=torch.int32,device=env.device)

    @property
    def action_dim(self): return 12

    @property
    def raw_action(self): return self.current

    def process_actions(self,actions):
        # Reference initializes last_last_actions but never advances it.
        # Keep that behavior explicit; opt-out is a documented algorithm change.
        if not self.cfg.source_history_compatibility:
            self.previous_previous.copy_(self.previous)
        self.previous.copy_(self.current)
        self.current.copy_(actions.clamp(-10.,10.))
        # Low-pass the policy command and apply a small dead-zone to prevent
        # contact chatter once the robot is close to standing.
        delayed=self.delay.push(self.current)
        self.filtered_action.mul_(0.65).add_(0.35*delayed)
        self.filtered_action.masked_fill_(self.filtered_action.abs() < 0.03, 0.0)
        if self.cfg.staged_unfold_enabled:
            unclassified=self.recovery_phase == 0
            deeply_folded=detect_deep_fold(self._entity.data.joint_pos)
            self.recovery_phase=torch.where(
                unclassified & deeply_folded,
                torch.ones_like(self.recovery_phase),
                torch.where(
                    unclassified,
                    torch.full_like(self.recovery_phase,3),
                    self.recovery_phase,
                ),
            )
        if self.cfg.stand_hold_enabled:
            force=self._env.scene['feet_ground_contact'].data.force
            if force is not None and force.shape[-2:] == (4,3):
                normal=force.norm(dim=-1)
                # Hysteresis: engage at 8 N, release only below 3 N.
                self.contact_state=(self.contact_state & (normal > 3.0)) | (normal > 8.0)
            gravity=self._entity.data.projected_gravity_b
            ang=self._entity.data.root_link_ang_vel_b
            stable=onboard_stand_ready(self._env, contact_mask=self.contact_state)
            if self.cfg.staged_unfold_enabled:
                stable &= self.recovery_phase != 1
            self.stable_counter=torch.where(stable,self.stable_counter+1,torch.zeros_like(self.stable_counter))
            # Latch hold through the short motion transient caused by moving
            # from the recovered posture to the canonical stand target.  The
            # previous one-step angular-rate release fired immediately after
            # every 10-step engagement and produced a deterministic chatter
            # loop.  Release only after sustained severe loss of posture or
            # support, both measurable from onboard sensors.
            severe=(gravity[...,2] > -0.70) | (self.contact_state.sum(dim=-1) < 2)
            self.destabilized_counter=torch.where(
                self.hold_mode & severe,
                self.destabilized_counter+1,
                torch.zeros_like(self.destabilized_counter),
            )
            engaged=self.hold_mode | (self.stable_counter >= 10)
            self.hold_mode=engaged & (self.destabilized_counter < 10)
            if self.cfg.staged_unfold_enabled:
                self.hold_mode &= self.recovery_phase != 1
            self.recovery_complete=self.stable_counter >= 50
            self._entity._recovery_hold_mode=self.hold_mode
        action_scale=torch.where(self.hold_mode, torch.zeros_like(self.stable_counter,dtype=self.current.dtype), torch.ones_like(self.stable_counter,dtype=self.current.dtype))
        recovery_desired=self._entity.data.default_joint_pos+.20*self.filtered_action
        desired=torch.where(
            self.hold_mode.unsqueeze(-1),
            self.hold_stand_target.unsqueeze(0),
            recovery_desired,
        )
        if self.cfg.staged_unfold_enabled:
            unfold_desired,unfold_ready=staged_unfold_command(
                self._entity.data.joint_pos,self._entity.data.joint_vel,
            )
            unfold_active=self.recovery_phase == 1
            self.unfold_ready_counter=torch.where(
                unfold_active & unfold_ready,
                self.unfold_ready_counter+1,
                torch.zeros_like(self.unfold_ready_counter),
            )
            self.unfold_steps+=unfold_active.to(self.unfold_steps.dtype)
            unfold_desired=staged_initial_hold(
                self.target,unfold_desired,self.unfold_steps,hold_steps=10,
            )
            calm=self._entity.data.joint_vel[...,THIGH_INDICES+CALF_INDICES].abs().amax(dim=-1) <= 1.5
            release=unfold_active & (
                (self.unfold_ready_counter >= 5)
                | ((self.unfold_steps >= 150) & calm)
            )
            self.recovery_phase=torch.where(
                release,torch.full_like(self.recovery_phase,2),self.recovery_phase
            )
            unfold_active=self.recovery_phase == 1
            desired=torch.where(unfold_active.unsqueeze(-1),unfold_desired,desired)
            handoff_active=self.recovery_phase == 2
            self.handoff_steps+=handoff_active.to(self.handoff_steps.dtype)
            blended=blend_staged_policy(
                unfold_desired,desired,self.handoff_steps,handoff_steps=50,
            )
            desired=torch.where(handoff_active.unsqueeze(-1),blended,desired)
            handoff_done=handoff_active & (self.handoff_steps >= 50)
            self.recovery_phase=torch.where(
                handoff_done,torch.full_like(self.recovery_phase,3),self.recovery_phase
            )
            self._entity._frnet_recovery_phase=self.recovery_phase.clone()
            self._entity._frnet_unfold_steps=self.unfold_steps.clone()
            self._entity._frnet_handoff_steps=self.handoff_steps.clone()
        # Deployment safety envelope: never command a hard-stop target and do
        # not change a position target faster than the official Go2 joint-speed
        # rating at the 50 Hz policy rate. A small margin prevents the PD loop
        # from deliberately driving into MuJoCo's joint-limit constraint.
        margin=0.02
        desired=torch.maximum(
            torch.minimum(desired,self.position_upper-margin),
            self.position_lower+margin,
        )
        target_slew=float(os.environ.get("FRNET_TARGET_SLEW_MAX_RAD", "0.0"))
        calf_target_slew=float(os.environ.get(
            "FRNET_TARGET_SLEW_CALF_MAX_RAD", "0.0"
        ))
        slew_limited=limit_target_slew(
            self._entity.data.joint_pos,
            desired,
            target_slew,
            calf_max_step=calf_target_slew,
        )
        # Keep the extra startup slew for partial folds, where it prevents a
        # receiving hip/thigh impulse. Fully folded starts use the dedicated
        # static-contact hold and must not receive a second slew layer, which
        # can create a step-0 calf impulse in symmetric contact.
        if os.environ.get("FRNET_TARGET_SLEW_SKIP_ALL_FOLDED", "0") == "1":
            folded_all=(
                (self._entity.data.joint_pos[..., THIGH_INDICES] > 2.50)
                & (self._entity.data.joint_pos[..., CALF_INDICES] < -2.45)
            ).all(dim=-1)
            desired=torch.where(folded_all.unsqueeze(-1),desired,slew_limited)
        else:
            desired=slew_limited
        predictive_gain=float(os.environ.get(
            "FRNET_PREDICTIVE_TARGET_BRAKE_GAIN", "0.0"
        ))
        predictive_calf_gain=float(os.environ.get(
            "FRNET_PREDICTIVE_TARGET_BRAKE_CALF_GAIN", "0.0"
        ))
        if predictive_gain > 0.0 or predictive_calf_gain > 0.0:
            desired=apply_predictive_target_brake(
                self._entity.data.joint_pos,
                self._entity.data.joint_vel,
                desired,
                self.position_lower,
                self.position_upper,
                self.target.new_tensor(VELOCITY_LIMIT),
                self.brake_accel,
                self._env.step_dt,
                margin=self.safety_margin,
                trigger_ratio=float(os.environ.get(
                    "FRNET_PREDICTIVE_TARGET_BRAKE_TRIGGER", "0.30"
                )),
                gain=predictive_gain,
                calf_gain=predictive_calf_gain,
            )
        safe_target,diag=compute_safety_target(
            self._entity.data.joint_pos,self._entity.data.joint_vel,desired,
            self.position_lower,self.position_upper,
            self.target.new_tensor(VELOCITY_LIMIT),self.safety_margin,
            self.brake_accel,self._env.step_dt,self.safety_step,
            predictive_all_joints=self.cfg.predictive_all_joints,
            general_trigger_ratio=self.cfg.predictive_general_trigger,
            calf_trigger_ratio=self.cfg.predictive_calf_trigger,
            general_reverse_offset=self.cfg.predictive_general_reverse_offset,
            calf_reverse_offset=self.cfg.predictive_calf_reverse_offset,
            velocity_brake_enabled=self.cfg.velocity_brake_enabled,
            velocity_brake_general_trigger=self.cfg.velocity_brake_general_trigger,
            velocity_brake_calf_trigger=self.cfg.velocity_brake_calf_trigger,
            velocity_brake_hip_trigger=float(os.environ.get(
                "FRNET_ACTUATOR_HIP_BRAKE_TRIGGER",
                str(self.cfg.velocity_brake_general_trigger),
            )),
        )
        if self.cfg.predictive_all_joints and self.cfg.predictive_latch_enabled:
            velocity_limit=self.target.new_tensor(VELOCITY_LIMIT)
            self.predictive_brake_latch=update_predictive_brake_latch(
                self.predictive_brake_latch,
                diag["brake_direction"],
                self._entity.data.joint_vel,
                velocity_limit,
                release_ratio=self.cfg.predictive_brake_release_ratio,
            )
            reverse_offset=torch.full_like(
                safe_target,self.cfg.predictive_general_reverse_offset
            )
            reverse_offset[...,CALF_INDICES]=self.cfg.predictive_calf_reverse_offset
            latched_target=(
                self._entity.data.joint_pos
                - self.predictive_brake_latch * reverse_offset
            )
            safe_target=torch.where(
                self.predictive_brake_latch != 0.0,latched_target,safe_target
            )
            safe_target=torch.maximum(
                torch.minimum(safe_target,self.position_upper-self.safety_margin),
                self.position_lower+self.safety_margin,
            )
            diag["modified"]=(safe_target-desired).abs() > 1e-7
        elif self.cfg.predictive_all_joints:
            self.predictive_brake_latch.zero_()
        velocity_delta = self._entity.data.joint_vel - self.previous_joint_velocity
        contact_rise_leg = torch.zeros(
            (self.num_envs, 4), dtype=torch.bool, device=self.device
        )
        raw_contact_rise_leg = torch.zeros_like(contact_rise_leg)
        contact_force_leg = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        contact_force_delta_leg = torch.zeros_like(contact_force_leg)
        if self.nonfoot_contact_leg_map is not None:
            try:
                contact_sensor = self._env.scene["nonfoot_ground_touch"]
                force = contact_sensor.data.force.norm(dim=-1)
                previous_force = self.nonfoot_contact_force_previous
                threshold = float(os.environ.get("FRNET_CONTACT_IMPACT_FORCE_N", "25.0"))
                delta_threshold = float(os.environ.get("FRNET_CONTACT_IMPACT_DELTA_N", "10.0"))
                release_threshold = float(os.environ.get(
                    "FRNET_CONTACT_IMPACT_RELEASE_N", str(max(threshold * 0.40, 3.0))
                ))
                warmup_steps = int(os.environ.get(
                    "FRNET_IMPACT_DETECTION_WARMUP_STEPS", "20"
                ))
                rising = (force >= threshold) & (
                    (previous_force < threshold)
                    | ((force - previous_force) >= delta_threshold)
                )
                # During the reset settling window we only learn the static
                # contact baseline.  Initially loaded sensors stay disarmed
                # until they unload, so their persistent support force cannot
                # be mistaken for an impact on the first action.
                sensor_leg_map = self.nonfoot_contact_leg_map.to(torch.float32)
                raw_contact_rise_leg = (
                    rising.to(torch.float32) @ sensor_leg_map
                ) > 0.0
                contact_force_leg = force @ sensor_leg_map
                contact_force_delta_leg = (force - previous_force).clamp_min(0.0) @ sensor_leg_map
                dynamic_rising, armed = filter_dynamic_contact_rise(
                    rising, force, self.nonfoot_contact_armed,
                    self.impact_detection_age,
                    threshold=threshold,
                    release_threshold=release_threshold,
                    warmup_steps=warmup_steps,
                    enabled=os.environ.get(
                        # Reset starts from static support contacts.  A raw
                        # contact rise is not an impact until the sensor has
                        # unloaded and then rises again after the warmup.
                        "FRNET_DYNAMIC_CONTACT_FILTER", "1"
                    ) == "1",
                )
                contact_rise_leg = (
                    dynamic_rising.to(torch.float32)
                    @ self.nonfoot_contact_leg_map.to(torch.float32)
                ) > 0.0
                self.nonfoot_contact_armed.copy_(armed)
                self.nonfoot_contact_force_previous.copy_(force)
            except Exception:
                contact_rise_leg.zero_()
        impact_diag = classify_impact_source(
            self._entity.data.joint_pos,
            self._entity.data.joint_vel,
            safe_target,
            self.position_lower,
            self.position_upper,
            self.target.new_tensor(VELOCITY_LIMIT),
            velocity_delta,
            contact_rise_leg,
            self.impact_source_streak,
            boundary_margin=float(os.environ.get("FRNET_IMPACT_SOURCE_BOUNDARY_MARGIN", "0.10")),
            min_speed_ratio=float(os.environ.get("FRNET_IMPACT_SOURCE_MIN_SPEED_RATIO", "0.15")),
            acceleration_ratio=float(os.environ.get("FRNET_IMPACT_SOURCE_ACCEL_RATIO", "0.05")),
            high_speed_ratio=float(os.environ.get("FRNET_IMPACT_SOURCE_HIGH_SPEED_RATIO", "0.60")),
            min_consecutive=int(os.environ.get("FRNET_IMPACT_SOURCE_MIN_CONSECUTIVE", "2")),
            detection_eligible=(
                self.impact_detection_age
                >= int(os.environ.get("FRNET_IMPACT_DETECTION_WARMUP_STEPS", "20"))
            ),
            deep_folded=getattr(self._entity, "_frnet_deep_folded", None),
        )
        source_latch_enabled = os.environ.get('FRNET_GLOBAL_IMPACT_SOURCE_LATCH', '0') == '1'
        if source_latch_enabled:
            latch_steps = int(os.environ.get('FRNET_GLOBAL_IMPACT_SOURCE_LATCH_STEPS', '8'))
            source_now = impact_diag['trigger'].any(dim=-1) & (impact_diag['source_leg'] >= 0)
            source_leg_index = impact_diag['source_leg'].clamp_min(0).unsqueeze(-1)
            source_contact = contact_rise_leg.gather(1, source_leg_index).squeeze(-1)
            source_force_delta = contact_force_delta_leg.gather(1, source_leg_index).squeeze(-1)
            force_delta_threshold = float(os.environ.get(
                'FRNET_GLOBAL_IMPACT_SOURCE_FORCE_DELTA_N', '0.0'
            ))
            source_now = source_now & source_contact & (
                source_force_delta >= force_delta_threshold
            )
            active_latch = self.impact_source_latch_counter > 0
            start_latch = source_now & (~active_latch)
            self.impact_source_latched_leg.copy_(torch.where(start_latch, impact_diag['source_leg'], self.impact_source_latched_leg))
            self.impact_source_latch_counter.copy_(torch.where(start_latch, torch.full_like(self.impact_source_latch_counter, latch_steps), torch.clamp(self.impact_source_latch_counter - 1, min=0)))
            latch_active = self.impact_source_latch_counter > 0
            effective_leg = torch.where(latch_active, self.impact_source_latched_leg, impact_diag['source_leg'])
            effective_leg = torch.where(latch_active | (impact_diag['source_leg'] >= 0), effective_leg, torch.full_like(effective_leg, -1))
            effective_trigger = impact_diag['trigger'].clone()
            valid = latch_active & (effective_leg >= 0)
            allowed = torch.zeros_like(effective_trigger)
            allowed.scatter_(1, effective_leg.clamp_min(0).unsqueeze(-1), valid.unsqueeze(-1))
            effective_trigger = effective_trigger & allowed
            impact_diag['source_leg'] = effective_leg
            impact_diag['trigger'] = effective_trigger
        self.impact_source_streak.copy_(impact_diag["streak"])
        self.impact_source_leg.copy_(impact_diag["source_leg"])
        self.impact_source_joint.copy_(impact_diag["source_joint"])
        source_value, _ = impact_diag["leg_scores"].max(dim=-1)
        self.impact_source_score.copy_(source_value)
        self.impact_source_trigger.copy_(impact_diag["trigger"].any(dim=-1))
        self._entity._frnet_impact_source_leg = self.impact_source_leg.clone()
        self._entity._frnet_impact_source_joint = self.impact_source_joint.clone()
        self._entity._frnet_impact_source_score = self.impact_source_score.clone()
        self._entity._frnet_impact_source_trigger = self.impact_source_trigger.clone()
        self._entity._frnet_impact_source_contact_rise = contact_rise_leg.clone()
        self._entity._frnet_raw_contact_rise_leg = raw_contact_rise_leg.clone()
        self._entity._frnet_contact_force_leg = contact_force_leg.clone()
        self._entity._frnet_contact_force_delta_leg = contact_force_delta_leg.clone()
        self._entity._frnet_joint_velocity_delta = velocity_delta.clone()
        if self.cfg.global_impact_guard_enabled:
            pre_local_guard_target = safe_target
            contact_leg_trigger = None
            if os.environ.get("FRNET_GLOBAL_IMPACT_CAUSAL_MODE", "0") == "1":
                # Use the temporally qualified source-leg classifier as an
                # additional local trigger.  This is opt-in so the existing
                # release behavior remains the comparison baseline.
                contact_leg_trigger = impact_diag["trigger"]
            if os.environ.get("PYTHON_FRNET_CONTACT_IMPACT_GUARD", "0") == "1":
                # The action guard keeps the existing conservative contact
                # filter, while the diagnostic source above remains causal.
                contact_leg_trigger = (
                    raw_contact_rise_leg.clone()
                    if contact_leg_trigger is None
                    else contact_leg_trigger | raw_contact_rise_leg
                )
                contact_leg_trigger = contact_leg_trigger & (
                    self.safety_step.unsqueeze(-1)
                    >= int(os.environ.get("FRNET_CONTACT_IMPACT_MIN_STEPS", "20"))
                )
                driving = self._entity.data.joint_vel * (
                    safe_target - self._entity.data.joint_pos
                ) > 0.0
                moving = self._entity.data.joint_vel.abs() > float(
                    os.environ.get("FRNET_CONTACT_IMPACT_MIN_SPEED", "0.20")
                )
                motion = driving & moving
                motion_leg = torch.stack(
                    [
                        motion[..., 0:3].any(dim=-1),
                        motion[..., 3:6].any(dim=-1),
                        motion[..., 6:9].any(dim=-1),
                        motion[..., 9:12].any(dim=-1),
                    ],
                    dim=-1,
                )
                contact_leg_trigger = contact_leg_trigger & motion_leg
            safe_target, self.global_guard_counter, guard_active, guard_trigger, self.global_transfer_counter, dynamic_transfer_receiver = apply_global_impact_guard(
                self._entity.data.joint_pos,
                self._entity.data.joint_vel,
                safe_target,
                self.position_lower,
                self.position_upper,
                self.target.new_tensor(VELOCITY_LIMIT),
                self.global_guard_counter,
                prearm_counter=self.prearm_counter,
                transfer_counter=self.global_transfer_counter,
                trigger_ratio=self.cfg.global_impact_guard_trigger,
                boundary_margin=self.cfg.global_impact_guard_margin,
                cooldown_steps=self.cfg.global_impact_guard_cooldown_steps,
                cross_joint_scale=self.cfg.global_impact_guard_cross_joint_scale,
                target_error_threshold=self.cfg.global_impact_guard_target_error,
                velocity_delta=velocity_delta,
                acceleration_threshold=float(os.environ.get(
                    'FRNET_GLOBAL_IMPACT_ACCEL_THRESHOLD','1000000000.0'
                )),
                contact_leg_trigger=contact_leg_trigger,
                source_leg_mask=(
                    torch.nn.functional.one_hot(
                        impact_diag["source_leg"].clamp_min(0), num_classes=4
                    ).to(torch.bool)
                    & (impact_diag["source_leg"].unsqueeze(-1) >= 0)
                    if os.environ.get("FRNET_GLOBAL_IMPACT_TRANSFER_USE_CLASSIFIED_SOURCE", "0") == "1"
                    else None
                ),
            )
            if os.environ.get('PYTHON_FRNET_LOCAL_TARGET_GUARD','1') != '1':
                safe_target = pre_local_guard_target
            self._entity._frnet_global_impact_guard_active=guard_active
            self._entity._frnet_global_impact_guard_trigger=guard_trigger
            self._entity._frnet_dynamic_transfer_receiver=dynamic_transfer_receiver
            diag["modified"]=(safe_target-desired).abs() > 1e-7
        else:
            self.global_guard_counter.zero_()
            self.global_transfer_counter.zero_()
            self.prearm_counter.zero_()
            self._entity._frnet_global_impact_guard_active = torch.zeros(
                (self.num_envs,4), dtype=torch.bool, device=self.device
            )
            self._entity._frnet_global_impact_guard_trigger = torch.zeros(
                (self.num_envs,4), dtype=torch.bool, device=self.device
            )
            self._entity._frnet_dynamic_transfer_receiver = torch.zeros(
                (self.num_envs,4), dtype=torch.bool, device=self.device
            )
        # Optional exact birth-mode receiver guard.  The mode is supplied by
        # reset/evaluation metadata, so this never guesses from incidental
        # static-library joint values.  It is intentionally narrow: only the
        # two previously observed 256-case failures are protected.
        mode_scale = float(os.environ.get(
            "FRNET_EXACT_MODE_GUARD_SCALE", "0.0"
        ))
        mode_code = getattr(self._entity, "_frnet_birth_mode_code", None)
        if mode_scale > 0.0 and mode_code is not None:
            mode_trigger = float(os.environ.get(
                "FRNET_EXACT_MODE_GUARD_TRIGGER", "0.80"
            ))
            mode_joint = torch.zeros_like(safe_target, dtype=torch.bool)
            mode_joint[..., 8] = mode_code == 3  # all_folded: RL calf
            mode_joint[..., 10] = mode_code == 1  # diagonal_FL_RR: RR thigh
            mode_joint[..., 11] = (mode_code == 4) | (mode_code == 2)  # single_RR/diagonal_FR_RL: RR calf
            mode_scale_vec = self.target.new_full(safe_target.shape, mode_scale)
            mode_scale_vec[..., 11] = float(os.environ.get(
                "FRNET_EXACT_MODE_SINGLE_RR_SCALE", str(mode_scale)
            ))
            # diagonal_FR_RL has a distinct early RR calf failure mode; give
            # this receiver its own predictive threshold/scale instead of
            # changing protection for the other birth modes.
            diag_fr_rl_scale = float(os.environ.get(
                "FRNET_EXACT_MODE_DIAG_FR_RL_SCALE", str(mode_scale)
            ))
            mode_scale_vec[..., 11] = torch.where(
                mode_code == 2,
                self.target.new_tensor(diag_fr_rl_scale),
                mode_scale_vec[..., 11],
            )
            diag_fr_rl_trigger = float(os.environ.get(
                "FRNET_EXACT_MODE_DIAG_FR_RL_TRIGGER", str(mode_trigger)
            ))
            mode_trigger_vec = self.target.new_full(mode_code.shape, mode_trigger)
            mode_trigger_vec = torch.where(
                mode_code == 2,
                self.target.new_tensor(diag_fr_rl_trigger),
                mode_trigger_vec,
            )
            mode_joint = mode_joint & (
                self._entity.data.joint_vel.abs()
                >= self.target.new_tensor(VELOCITY_LIMIT) * mode_trigger_vec.unsqueeze(-1)
            ) & (
                self._entity.data.joint_vel
                * (safe_target - self._entity.data.joint_pos) > 0.0
            )
            safe_target = torch.where(
                mode_joint,
                self._entity.data.joint_pos
                + (safe_target - self._entity.data.joint_pos) * mode_scale_vec,
                safe_target,
            )
            diag["modified"] = (safe_target - desired).abs() > 1e-7
        # Run source-leg coordination after the global guard so the guard
        # still sees the unmodified policy target and can record the causal
        # event.  This layer only shortens the already-safe target for a brief
        # source-leg window.
        if os.environ.get("FRNET_SOURCE_JOINT_COORDINATION", "0") == "1":
            source_trigger = impact_diag["trigger"].any(dim=-1)
            source_leg_for_streak = impact_diag["source_leg"].clamp_min(0)
            source_streak = impact_diag["streak"].gather(
                1, source_leg_for_streak.unsqueeze(-1)
            ).squeeze(-1)
            source_window = int(os.environ.get(
                "FRNET_SOURCE_JOINT_COORDINATION_MAX_STEPS", "4"
            ))
            if source_window > 0:
                source_trigger = source_trigger & (source_streak <= source_window)
            if os.environ.get("FRNET_SOURCE_JOINT_REQUIRE_NONHIP", "0") == "1":
                source_trigger = source_trigger & (
                    (impact_diag["source_joint"] >= 0)
                    & ((impact_diag["source_joint"] % 3) != 0)
                )
            safe_target = apply_source_leg_coordination(
                self._entity.data.joint_pos,
                self._entity.data.joint_vel,
                safe_target,
                impact_diag["source_leg"],
                impact_diag["source_joint"],
                source_trigger,
                calf_scale=float(os.environ.get("FRNET_SOURCE_JOINT_CALF_SCALE", "0.85")),
                thigh_scale=float(os.environ.get("FRNET_SOURCE_JOINT_THIGH_SCALE", "0.95")),
                hip_scale=float(os.environ.get("FRNET_SOURCE_JOINT_HIP_SCALE", "1.0")),
            )
            diag["modified"] = (safe_target - desired).abs() > 1e-7
        reset_ramp_steps=int(os.environ.get("FRNET_RESET_TARGET_RAMP_STEPS", "20"))
        if reset_ramp_steps > 0:
            ramp_active=self.safety_step < reset_ramp_steps
            ramp_target=interpolate_reset_target(
                self.reset_start_target,
                self.reset_reference_target,
                self.safety_step,
                reset_ramp_steps,
            )
            ramp_target=torch.maximum(
                torch.minimum(ramp_target,self.position_upper-self.safety_margin),
                self.position_lower+self.safety_margin,
            )
            safe_target=torch.where(ramp_active.unsqueeze(-1),ramp_target,safe_target)
        adaptive_hold=int(os.environ.get("FRNET_ADAPTIVE_FOLDED_HOLD_STEPS", "0"))
        hold_steps=self.initial_target_hold_steps
        if adaptive_hold > 0:
            folded=((self._entity.data.joint_pos[..., THIGH_INDICES] > 2.50)
                    & (self._entity.data.joint_pos[..., CALF_INDICES] < -2.45)).sum(dim=-1)
            min_folded_legs = int(os.environ.get("FRNET_ADAPTIVE_FOLDED_MIN_LEGS", "2"))
            hold_steps=torch.where(
                folded >= max(1, min_folded_legs),
                torch.maximum(hold_steps, hold_steps.new_full(hold_steps.shape, adaptive_hold)),
                hold_steps,
            )
        if os.environ.get("PYTHON_FRNET_DEBUG_HOLD", "0") == "1":
            watch = (self.safety_step == 0) | (self.safety_step == 100) | (self.safety_step == 101)
            if bool(watch.any().item()):
                print("FRNET_HOLD_DEBUG", self.safety_step.detach().cpu().tolist(), "adaptive", adaptive_hold, "folded", folded.detach().cpu().tolist() if adaptive_hold > 0 else None, "hold", hold_steps.detach().cpu().tolist() if torch.is_tensor(hold_steps) else hold_steps, flush=True)
        safe_target=hold_reset_synchronized_target(
            self.target,safe_target,self.safety_step,hold_steps,
        )
        # Do not hand a static-contact equilibrium target directly to the
        # policy on the first frame after the initial hold.  Blend the policy
        # target in over a short release window so a body-load transfer cannot
        # create a deterministic step-24 calf/hip spike.
        release_ramp_steps=int(os.environ.get("FRNET_RESET_RELEASE_RAMP_STEPS", "0"))
        if release_ramp_steps > 0:
            hold_tensor = hold_steps if torch.is_tensor(hold_steps) else self.safety_step.new_full(self.safety_step.shape, int(hold_steps))
            release_age=self.safety_step-hold_tensor
            release_active=(release_age >= 0) & (release_age < release_ramp_steps)
            release_alpha=((release_age.to(safe_target.dtype)+1.0)/float(release_ramp_steps)).clamp(0.0,1.0)
            release_target=torch.lerp(self.target,safe_target,release_alpha.unsqueeze(-1))
            safe_target=torch.where(release_active.unsqueeze(-1),release_target,safe_target)
        diag["modified"]=(safe_target-desired).abs() > 1e-7
        self.safety_modified=diag["modified"].any(dim=-1)
        self.safety_brake_distance=diag["brake_distance"]
        self.safety_allowed_velocity=diag["allowed_velocity"]
        self._entity._frnet_safety_modified=self.safety_modified
        self._entity._frnet_safety_brake_distance=self.safety_brake_distance
        self._entity._frnet_safety_raw_target=desired
        self._entity._frnet_safety_target=safe_target
        self._entity._frnet_safety_target_delta=(safe_target-desired).abs()
        self._entity._frnet_safety_step=self.safety_step.clone()
        self._entity._frnet_predictive_brake_latch=self.predictive_brake_latch.clone()
        delta=(safe_target-self.target).clamp(
            -self.max_target_delta,self.max_target_delta
        )
        # During contact acquisition use a smaller target slew, then restore
        # the normal recovery rate.  This limits the first impact without
        # freezing the policy for the rest of the episode.
        startup_rate=float(os.environ.get("FRNET_STARTUP_TARGET_RATE_SCALE", "1.0"))
        if startup_rate < 1.0:
            startup_steps=int(os.environ.get("FRNET_STARTUP_TARGET_RATE_STEPS", "50"))
            startup_mask=(self.safety_step < startup_steps).unsqueeze(-1)
            delta=torch.where(startup_mask, delta*startup_rate, delta)
        self.target=self.target+delta
        # Optional local emergency stop for a driven calf that has already
        # crossed a configured speed ratio.  This is deliberately per-joint:
        # it retracts only the offending calf target while preserving all
        # other legs and joints.  The default is disabled for compatibility.
        emergency_ratio=float(os.environ.get("FRNET_EMERGENCY_CALF_STOP_RATIO", "0.0"))
        if emergency_ratio > 0.0:
            limits=self.target.new_tensor(VELOCITY_LIMIT).expand_as(self.target)
            q_now=self._entity.data.joint_pos
            v_now=self._entity.data.joint_vel
            driven=v_now * (self.target - q_now) > 0.0
            emergency=torch.zeros_like(driven)
            # Acceleration catches a sudden inertial reversal before the
            # velocity ratio itself reaches the limit.  The ratio is the
            # one-step velocity change divided by that joint's allowed speed.
            accel_stop_ratio=float(os.environ.get("FRNET_EMERGENCY_ACCEL_STOP_RATIO", "0.0"))
            accel_ratio=(v_now-self.previous_joint_velocity).abs() / limits.clamp_min(1e-6)
            accel_trigger=(accel_stop_ratio > 0.0) & (accel_ratio >= accel_stop_ratio)
            calf_speed_ratio = v_now[..., CALF_INDICES].abs() / limits[..., CALF_INDICES].clamp_min(1e-6)
            emergency[..., CALF_INDICES] = (
                driven[..., CALF_INDICES]
                & (calf_speed_ratio >= emergency_ratio)
            )
            # If inertia has already reversed the motion, ``driven`` is false
            # and the old emergency stop missed the event.  A higher,
            # direction-independent threshold catches that late phase while
            # leaving ordinary low-speed/static support untouched.
            undriven_ratio = float(os.environ.get(
                "FRNET_EMERGENCY_UNDRIVEN_STOP_RATIO", "0.0"
            ))
            if undriven_ratio > 0.0:
                emergency[..., CALF_INDICES] |= (
                    calf_speed_ratio >= undriven_ratio
                )
            if accel_stop_ratio > 0.0:
                emergency[..., CALF_INDICES] |= accel_trigger[..., CALF_INDICES]
            stop_offset=float(os.environ.get("FRNET_EMERGENCY_CALF_STOP_OFFSET_RAD", "0.0"))
            stop_target=q_now - torch.sign(v_now) * stop_offset
            # A calf emergency is a leg-level event: freeze the hip and
            # thigh of that same leg for this action as well, while leaving
            # the other three legs untouched.
            leg_emergency=torch.zeros_like(emergency)
            for calf_index, leg_start in zip(CALF_INDICES, (0, 3, 6, 9)):
                leg_emergency[..., leg_start:leg_start + 3] = emergency[..., calf_index].unsqueeze(-1)
            # A calf trigger protects the whole source leg for a few frames.
            # The latch is local to that leg, so the other three legs retain
            # their recovery targets.
            hold_steps=int(os.environ.get("FRNET_EMERGENCY_HOLD_STEPS", "4"))
            leg_trigger=leg_emergency.reshape(leg_emergency.shape[0],4,3).any(dim=-1)
            if hold_steps > 0:
                self.emergency_leg_latch=torch.where(
                    leg_trigger,
                    self.emergency_leg_latch.new_full(self.emergency_leg_latch.shape, hold_steps),
                    (self.emergency_leg_latch-1).clamp_min(0),
                )
                latched_leg=self.emergency_leg_latch > 0
                leg_emergency = leg_emergency | latched_leg.unsqueeze(-1).expand(-1,-1,3).reshape_as(leg_emergency)
            # A thigh can exceed its limit before the calf signal appears.
            # Keep this trigger separate so it can be tuned without changing
            # the calf threshold, and freeze only the affected leg.
            thigh_ratio=float(os.environ.get("FRNET_EMERGENCY_THIGH_STOP_RATIO", "0.0"))
            if thigh_ratio > 0.0:
                thigh_speed_ratio = v_now[..., THIGH_INDICES].abs() / limits[..., THIGH_INDICES].clamp_min(1e-6)
                thigh_emergency=(
                    driven[..., THIGH_INDICES]
                    & (thigh_speed_ratio >= thigh_ratio)
                )
                if undriven_ratio > 0.0:
                    thigh_emergency |= (thigh_speed_ratio >= undriven_ratio)
                if accel_stop_ratio > 0.0:
                    thigh_emergency |= accel_trigger[..., THIGH_INDICES]
                thigh_offset=float(os.environ.get(
                    "FRNET_EMERGENCY_THIGH_STOP_OFFSET_RAD", str(stop_offset)
                ))
                thigh_stop_target=q_now - torch.sign(v_now) * thigh_offset
                thigh_leg_emergency=torch.zeros_like(emergency)
                for thigh_slot, leg_start in enumerate((0, 3, 6, 9)):
                    thigh_leg_emergency[..., leg_start:leg_start + 3] |= thigh_emergency[..., thigh_slot].unsqueeze(-1)
                leg_emergency |= thigh_leg_emergency
                stop_target=torch.where(thigh_leg_emergency, thigh_stop_target, stop_target)
            # Hip protection is joint-local: preserve the hip's ability to
            # hold posture while stopping only a driven hip that is already
            # approaching its velocity limit.  The other two joints and all
            # other legs keep their recovery targets.
            hip_ratio=float(os.environ.get("FRNET_EMERGENCY_HIP_STOP_RATIO", "0.0"))
            if hip_ratio > 0.0:
                hip_speed_ratio = v_now[..., HIP_INDICES].abs() / limits[..., HIP_INDICES].clamp_min(1e-6)
                hip_emergency=(
                    driven[..., HIP_INDICES]
                    & (hip_speed_ratio >= hip_ratio)
                )
                if undriven_ratio > 0.0:
                    hip_emergency |= (hip_speed_ratio >= undriven_ratio)
                if accel_stop_ratio > 0.0:
                    hip_emergency |= accel_trigger[..., HIP_INDICES]
                hip_offset=float(os.environ.get(
                    "FRNET_EMERGENCY_HIP_STOP_OFFSET_RAD", str(stop_offset)
                ))
                hip_stop_target=q_now - torch.sign(v_now) * hip_offset
                hip_joint_stop=torch.zeros_like(emergency)
                hip_joint_stop[..., HIP_INDICES] = hip_emergency
                self.target=torch.where(hip_joint_stop, hip_stop_target, self.target)
            self.target=torch.where(leg_emergency, stop_target, self.target)
        # Update the velocity history only after every acceleration-based guard
        # has consumed the previous frame.  Updating it before the emergency
        # check makes accel_ratio identically zero and disables early braking.
        self.previous_joint_velocity.copy_(self._entity.data.joint_vel)
        self.safety_step+=1
        self.impact_detection_age+=1

    def apply_actions(self):
        self._entity.set_joint_position_target(self.target)
        self._entity.set_joint_velocity_target(torch.zeros_like(self.target))

    def reset(self,env_ids=None):
        ids=torch.arange(self.num_envs,device=self.device)
        if env_ids is not None: ids=ids[env_ids]
        self.delay.reset(ids)
        for tensor in (self.current,self.previous,self.previous_previous,self.filtered_action): tensor[ids]=0
        reset_contact_force_history(self.nonfoot_contact_force_previous, ids)
        self.contact_state[ids]=False
        self.stable_counter[ids]=0
        self.destabilized_counter[ids]=0
        self.hold_mode[ids]=False
        self.recovery_complete[ids]=False
        self.recovery_phase[ids]=0
        self.unfold_ready_counter[ids]=0
        self.unfold_steps[ids]=0
        self.handoff_steps[ids]=0
        self.safety_step[ids]=0
        self.initial_target_hold_steps[ids]=int(self.cfg.initial_target_hold_steps)
        self.safety_modified[ids]=False
        self.safety_brake_distance[ids]=0
        self.safety_allowed_velocity[ids]=0
        self.predictive_brake_latch[ids]=0
        self.global_guard_counter[ids]=0
        self.global_transfer_counter[ids]=0
        self.prearm_counter[ids]=0
        self.impact_source_streak[ids]=0
        self.impact_source_leg[ids]=-1
        self.impact_source_joint[ids]=-1
        self.impact_source_score[ids]=0
        self.impact_source_trigger[ids]=False
        self.impact_source_latched_leg[ids]=-1
        self.impact_source_latch_counter[ids]=0
        self.impact_detection_age[ids]=0
        self.emergency_leg_latch[ids]=0
        if self.nonfoot_contact_armed is not None:
            self.nonfoot_contact_armed[ids]=False
        self.previous_joint_velocity[ids]=0
        self.reset_start_target[ids]=self._entity.data.joint_pos[ids]
        self.reset_reference_target[ids]=self._entity.data.joint_pos[ids]
        # Avoid a large first-step PD impulse when reset starts deeply folded.
        self.target[ids]=self._entity.data.joint_pos[ids]

    def sync_target_to_joint_state(self, ids, joint_pos=None, reference_target=None):
        """Synchronize q and optionally retain a smooth release target."""
        if joint_pos is None:
            joint_pos=self._entity.data.joint_pos[ids]
        if reference_target is None:
            reference_target=joint_pos
        self.reset_start_target[ids]=joint_pos
        self.reset_reference_target[ids]=reference_target
        # Optional reset equilibrium target: a bounded target computed from
        # static contact acceleration is safe to apply immediately because it
        # is already close to measured q.  The default retains legacy q sync.
        if os.environ.get("PYTHON_FRNET_STATIC_ACCEL_TARGET", "0") == "1":
            self.target[ids]=reference_target
        else:
            self.target[ids]=joint_pos
        self.safety_step[ids]=0

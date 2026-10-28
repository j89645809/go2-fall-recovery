"""Isolated FR-Net source-derived robot; never mutates custom Go2 constants."""
from pathlib import Path
from dataclasses import dataclass, replace
import os
import mujoco
import torch
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg
from mjlab.actuator.actuator import ActuatorCmd
from .official_go2_limits import FRNET_EFFORT_LIMIT, JOINT_NAMES, POSITION_LOWER, POSITION_UPPER, VELOCITY_LIMIT

BODY_NAMES = ('base', 'Head_upper', 'Head_lower') + tuple(
    f'{leg}_{part}' for leg in ('FL','FR','RL','RR') for part in ('hip','thigh','calf','foot'))


def configure_joint_limit_constraint(
    joint, *, margin=0.02, timeconst=0.01, dampratio=1.0,
    impedance_min=0.95, impedance_max=0.99,
):
    """Configure a firm, damped MuJoCo joint stop for the 5 ms physics step."""
    if timeconst < 0.01:
        raise ValueError("joint-limit time constant must be at least two physics steps")
    joint.margin=float(margin)
    joint.solref_limit=(float(timeconst),float(dampratio))
    joint.solimp_limit=(
        float(impedance_min),float(impedance_max),0.001,0.5,2.0,
    )


def actuator_damping_for_part(part):
    """Return configurable motor D gain while preserving the 0.8 default."""
    key=(
        "FRNET_ACTUATOR_CALF_DAMPING"
        if part == "calf" else "FRNET_ACTUATOR_GENERAL_DAMPING"
    )
    value=float(os.environ.get(key,"0.8"))
    if not 0.0 <= value <= 8.0:
        raise ValueError(f"{key} must be in [0, 8] N m s/rad")
    return value


def actuator_armature_for_part(part, official_passive_dynamics):
    """Return the reflected motor inertia used by the joint dynamics."""
    default="0.01" if official_passive_dynamics else "0.0"
    key=(
        "FRNET_ACTUATOR_CALF_ARMATURE"
        if part == "calf" else "FRNET_ACTUATOR_GENERAL_ARMATURE"
    )
    value=float(os.environ.get(key,default))
    if not 0.0 <= value <= 0.20:
        raise ValueError(f"{key} must be in [0, 0.20] kg m^2")
    return value


def remaining_brake_safety_factor_for_part(part, default):
    """Read a per-joint-group stopping-distance safety factor."""
    key=f"FRNET_REMAINING_BRAKE_{part.upper()}_SAFETY_FACTOR"
    value=float(os.environ.get(key, str(default)))
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{key} must be in (0, 1]")
    return value


def apply_remaining_braking_distance_limit(
    position, velocity, desired, lower, upper, max_torque, velocity_limit,
    *, dt, margin, inertia, safety_factor=0.10, max_target_step=0.015,
):
    """Limit an outward target using the torque-limited stopping distance.

    ``max_torque / inertia`` is the actuator's ideal maximum angular
    acceleration.  ``safety_factor`` deliberately uses only a fraction of
    it because contacts and payload reduce the acceleration available for
    braking.  The target is changed only when the current motion and the
    requested target point toward the same joint limit; it is never reversed.
    """
    q = position
    v = velocity
    target = desired
    lo = lower.expand_as(q) + float(margin)
    hi = upper.expand_as(q) - float(margin)
    effort = max_torque.expand_as(q)
    speed_limit = velocity_limit.expand_as(q)
    effective_inertia = torch.as_tensor(
        inertia, device=q.device, dtype=q.dtype,
    ).clamp_min(1e-4)
    accel = (effort / effective_inertia) * float(safety_factor)
    accel = accel.clamp_min(1e-3)
    dist_up = (hi - q).clamp_min(0.0)
    dist_down = (q - lo).clamp_min(0.0)
    allowed_up = torch.minimum(speed_limit, torch.sqrt(2.0 * accel * dist_up))
    allowed_down = torch.minimum(speed_limit, torch.sqrt(2.0 * accel * dist_down))
    outward_up = (v > 0.0) & (target > q)
    outward_down = (v < 0.0) & (target < q)
    allowed = torch.where(v >= 0.0, allowed_up, allowed_down)
    max_delta = allowed * float(dt)
    delta = target - q
    capped_delta = torch.where(
        outward_up, torch.minimum(delta, max_delta),
        torch.where(outward_down, torch.maximum(delta, -max_delta), delta),
    )
    # Limit only the safety correction, not the policy's whole target step.
    # This preserves useful recovery motion while making every intervention
    # small enough to avoid a forced reverse rebound.
    correction = delta - capped_delta
    capped_delta = torch.where(
        outward_up | outward_down,
        delta - torch.sign(delta) * torch.minimum(
            correction.abs(), torch.as_tensor(
                max_target_step, device=q.device, dtype=q.dtype
            )
        ),
        capped_delta,
    )
    safe_target = torch.maximum(torch.minimum(q + capped_delta, hi), lo)
    modified = (safe_target - target).abs() > 1e-7
    return safe_target, allowed, modified


def apply_velocity_brake_torque(
    torque, velocity, force_limit, velocity_limit, *, trigger_ratio,
):
    """Guarantee increasing reverse torque as rated speed is approached."""
    trigger=float(trigger_ratio)
    if not 0.0 < trigger < 1.0:
        raise ValueError("trigger_ratio must be in (0, 1)")
    limit=velocity_limit.expand_as(velocity)
    effort=force_limit.expand_as(torque)
    blend=((velocity.abs()/limit-trigger)/(1.0-trigger)).clamp(0.0,1.0)
    required_brake=-torch.sign(velocity)*effort*blend
    candidate=torch.where(
        velocity > 0.0,
        torch.minimum(torque,required_brake),
        torch.where(velocity < 0.0,torch.maximum(torque,required_brake),torque),
    )
    safe=torch.where(blend > 0.0,candidate,torque)
    return torch.maximum(torch.minimum(safe,effort),-effort)


def apply_remaining_braking_torque(
    torque, velocity, allowed_velocity, force_limit, velocity_limit, *, gain=1.0,
    base_gain=0.25,
):
    """Add bounded, dissipative torque only above the safe stopping speed."""
    effort = force_limit.expand_as(torque)
    speed_limit = velocity_limit.expand_as(velocity)
    allowed = allowed_velocity.expand_as(velocity)
    excess = (velocity.abs() - allowed).clamp_min(0.0)
    active = excess > 0.0
    severity = (excess / speed_limit.clamp_min(1e-6)).clamp(0.0, 1.0)
    speed_ratio = velocity.abs() / speed_limit.clamp_min(1e-6)
    high_speed = ((speed_ratio - 1.0) / 0.20).clamp(0.0, 1.0)
    allowed_ratio = allowed / speed_limit.clamp_min(1e-6)
    boundary_risk = ((0.25 - allowed_ratio) / 0.25).clamp(0.0, 1.0)
    hazard = torch.maximum(high_speed, boundary_risk)
    effective_gain = float(base_gain) + (float(gain) - float(base_gain)) * hazard
    correction = -torch.sign(velocity) * effort * severity * effective_gain
    candidate = torque + correction
    driving = torque * velocity >= 0.0
    would_reverse = candidate * velocity < 0.0
    candidate = torch.where(active & driving & would_reverse, torch.zeros_like(candidate), candidate)
    safe = torch.where(active, candidate, torque)
    return torch.maximum(torch.minimum(safe, effort), -effort), active


def apply_position_barrier_torque(
    torque, position, velocity, force_limit, velocity_limit, lower, upper,
    brake_accel, *, margin=0.05, brake_gain=1.0,
):
    """Brake at 200 Hz when the predicted stop point crosses a safe bound."""
    effort=force_limit.expand_as(torque)
    del velocity_limit
    lo=lower.expand_as(position)+float(margin)
    hi=upper.expand_as(position)-float(margin)
    accel=brake_accel.expand_as(velocity).clamp_min(1e-3)
    stop_distance=velocity.square()/(2.0*accel)
    stop=position+torch.sign(velocity)*stop_distance
    upper_risk=(velocity>0.0) & (stop>hi)
    lower_risk=(velocity<0.0) & (stop<lo)
    distance=torch.where(
        velocity>0.0,(hi-position).clamp_min(0.0),
        (position-lo).clamp_min(0.0),
    )
    severity=((stop_distance-distance)/stop_distance.clamp_min(1e-6)).clamp(0.0,1.0)
    requested=-torch.sign(velocity)*effort*float(brake_gain)*severity
    braking=torch.where(
        velocity>0.0,
        torch.minimum(torque,requested),
        torch.where(velocity<0.0,torch.maximum(torque,requested),torque),
    )
    active=upper_risk | lower_risk
    safe=torch.where(active,braking,torque)
    return torch.maximum(torch.minimum(safe,effort),-effort),active


def apply_predictive_position_damping(
    torque, position, velocity, force_limit, lower, upper, brake_accel,
    *, margin=0.12, damping_gain=2.0,
):
    """Dissipate outward joint motion before its predicted stop crosses a limit.

    The correction never adds mechanical energy: it first fades any torque in
    the direction of the hazardous motion, then adds viscous damping.  Unlike
    the hard position barrier, it does not command a fixed reverse effort that
    can launch the joint toward the opposite limit after contact releases.
    """
    effort=force_limit.expand_as(torque)
    lo=lower.expand_as(position)+float(margin)
    hi=upper.expand_as(position)-float(margin)
    accel=brake_accel.expand_as(velocity).clamp_min(1e-3)
    stop_distance=velocity.square()/(2.0*accel)
    stop=position+torch.sign(velocity)*stop_distance
    upper_risk=(velocity>0.0) & (stop>hi)
    lower_risk=(velocity<0.0) & (stop<lo)
    active=upper_risk | lower_risk
    distance=torch.where(
        velocity>0.0,(hi-position).clamp_min(0.0),
        (position-lo).clamp_min(0.0),
    )
    severity=((stop_distance-distance)/stop_distance.clamp_min(1e-6)).clamp(0.0,1.0)
    severity=torch.where(active,severity,torch.zeros_like(severity))
    outward=(torque*velocity)>0.0
    faded=torch.where(outward,torque*(1.0-severity),torque)
    damped=faded-float(damping_gain)*velocity*severity
    safe=torch.where(active,damped,torque)
    return torch.maximum(torch.minimum(safe,effort),-effort),active


def apply_folded_calf_preload(
    torque, position, force_limit, safety_step, *, preload_nm, preload_steps,
    folded_threshold=-2.45,
):
    """Add bounded flexion torque briefly while a folded calf takes contact."""
    if preload_steps <= 0 or preload_nm <= 0.0:
        return torque
    effort=force_limit.expand_as(torque)
    step=torch.as_tensor(safety_step,device=torque.device)
    if step.ndim == 0:
        step=step.expand(torque.shape[:-1])
    active=(step < int(preload_steps)).unsqueeze(-1) & (
        position < float(folded_threshold)
    )
    preloaded=torch.where(active,torque-float(preload_nm),torque)
    return torch.maximum(torch.minimum(preloaded,effort),-effort)


def accumulate_barrier_diagnostics(count, max_delta, active, torque_delta):
    """Accumulate per-joint barrier interventions without affecting control."""
    count = count + active.to(dtype=count.dtype)
    active_delta = torch.where(active, torque_delta.abs(), torch.zeros_like(torque_delta))
    return count, torch.maximum(max_delta, active_delta)


class SafeIdealPdActuator(IdealPdActuator):
    def __init__(self, cfg, entity, target_ids, target_names, velocity_limit,
                 velocity_brake_enabled, velocity_brake_trigger_ratio,
                 position_barrier_enabled, position_lower, position_upper,
                 brake_accel, position_barrier_margin,position_barrier_brake_gain,
                 position_damping_enabled,position_damping_margin,
                 position_damping_gain,folded_calf_preload_enabled,
                 folded_calf_preload_nm,folded_calf_preload_steps,
                 remaining_brake_enabled, remaining_brake_inertia,
                 remaining_brake_safety_factor, remaining_brake_dt,
                 remaining_brake_max_target_step, remaining_brake_torque_gain,
                 remaining_brake_torque_base_gain):
        super().__init__(cfg, entity, target_ids, target_names)
        self._frnet_target_ids = torch.as_tensor(target_ids, dtype=torch.long)
        self.velocity_limit = velocity_limit
        self.velocity_brake_enabled=velocity_brake_enabled
        self.velocity_brake_trigger_ratio=velocity_brake_trigger_ratio
        self.position_barrier_enabled=position_barrier_enabled
        self.position_lower=position_lower
        self.position_upper=position_upper
        self.brake_accel=brake_accel
        self.position_barrier_margin=position_barrier_margin
        self.position_barrier_brake_gain=position_barrier_brake_gain
        self.position_damping_enabled=position_damping_enabled
        self.position_damping_margin=position_damping_margin
        self.position_damping_gain=position_damping_gain
        self.folded_calf_preload_enabled=folded_calf_preload_enabled
        self.folded_calf_preload_nm=folded_calf_preload_nm
        self.folded_calf_preload_steps=folded_calf_preload_steps
        self.remaining_brake_enabled=remaining_brake_enabled
        self.remaining_brake_inertia=remaining_brake_inertia
        self.remaining_brake_safety_factor=remaining_brake_safety_factor
        self.remaining_brake_dt=remaining_brake_dt
        self.remaining_brake_max_target_step=remaining_brake_max_target_step
        self.remaining_brake_torque_gain=remaining_brake_torque_gain
        self.remaining_brake_torque_base_gain=remaining_brake_torque_base_gain
        self.barrier_active_counts=None
        self.barrier_max_torque_delta=None

    def reset_barrier_diagnostics(self, env_ids=None):
        if self.barrier_active_counts is None:
            return
        if env_ids is None:
            self.barrier_active_counts.zero_()
            self.barrier_max_torque_delta.zero_()
        else:
            self.barrier_active_counts[env_ids]=0
            self.barrier_max_torque_delta[env_ids]=0

    def compute(self, cmd: ActuatorCmd):
        # ActuatorCmd.vel is the requested velocity target.  FR-Net sends a
        # zero velocity target, so using it for safety checks hides the actual
        # motion.  Every interlock below must use measured joint velocity from
        # the articulation instead.
        full_velocity = getattr(self.entity.data, "joint_vel", None)
        measured_velocity = None
        if full_velocity is not None and full_velocity.shape[-1] == len(VELOCITY_LIMIT):
            ids = self._frnet_target_ids.to(device=full_velocity.device)
            measured_velocity = full_velocity.index_select(-1, ids)
        safety_velocity = measured_velocity if measured_velocity is not None else cmd.vel
        remaining_allowed_velocity=None
        if self.remaining_brake_enabled:
            assert self.force_limit is not None
            safe_target, allowed_velocity, modified = apply_remaining_braking_distance_limit(
                cmd.pos, safety_velocity, cmd.position_target,
                cmd.pos.new_tensor(self.position_lower),
                cmd.pos.new_tensor(self.position_upper),
                self.force_limit, safety_velocity.new_tensor(self.velocity_limit),
                dt=self.remaining_brake_dt,
                margin=self.position_barrier_margin,
                inertia=self.remaining_brake_inertia,
                safety_factor=self.remaining_brake_safety_factor,
                max_target_step=self.remaining_brake_max_target_step,
            )
            self.entity._frnet_remaining_brake_active = modified
            self.entity._frnet_remaining_brake_allowed_velocity = allowed_velocity
            remaining_allowed_velocity=allowed_velocity
            cmd = replace(cmd, position_target=safe_target)
        torque = super().compute(cmd)
        assert self.force_limit is not None
        # Substep-level local interlock.  The policy target is updated at
        # 50 Hz, but MuJoCo advances four physics substeps per target.  Read
        # all joint velocities here so a single receiving joint is damped
        # before the next policy update; unrelated joints keep their torque.
        full_velocity = getattr(self.entity.data, "joint_vel", None)
        substep_guard = os.environ.get("PYTHON_FRNET_SUBSTEP_LOCAL_GUARD", "0") == "1"
        folded_transfer_only = os.environ.get("FRNET_FOLDED_TRANSFER_ONLY_GUARD", "0") == "1"
        if substep_guard or folded_transfer_only:
            if full_velocity is not None and full_velocity.shape[-1] == len(VELOCITY_LIMIT):
                full_limit = full_velocity.new_tensor(VELOCITY_LIMIT)
                local_trigger = (
                    full_velocity.abs() / full_limit.clamp_min(1e-6)
                ) >= float(os.environ.get("FRNET_SUBSTEP_LOCAL_TRIGGER", "0.30"))
                ids = self._frnet_target_ids.to(device=local_trigger.device)
                local_active = local_trigger.index_select(-1, ids)
                if folded_transfer_only and not substep_guard:
                    local_active = torch.zeros_like(local_active)
                # In the stricter early-transfer mode, apply the substep
                # interlock only to environments whose reset state is deeply
                # folded.  Ordinary fallen states keep the normal policy
                # response and recovery success rate.
                if os.environ.get("FRNET_SUBSTEP_LOCAL_DEEP_FOLDED_ONLY", "0") == "1":
                    deep_folded = getattr(self.entity, "_frnet_deep_folded", None)
                    if deep_folded is not None:
                        local_active = local_active & deep_folded.unsqueeze(-1)
                        if os.environ.get("FRNET_SUBSTEP_LOCAL_LEG_MASK", "0") == "1":
                            source_legs = getattr(self.entity, "_frnet_folded_legs", None)
                            receiver_legs = getattr(self.entity, "_frnet_folded_transfer_receivers", None)
                            if source_legs is not None and receiver_legs is not None:
                                protected_legs_full = (source_legs | receiver_legs).unsqueeze(-1).expand(-1, -1, 3).reshape(source_legs.shape[0], -1)
                                protected_legs = protected_legs_full.index_select(-1, ids)
                                local_active = local_active & protected_legs
                        if os.environ.get("FRNET_SUBSTEP_LOCAL_SINGLE_FOLDED_ONLY", "0") == "1":
                            folded_count = getattr(self.entity, "_frnet_deep_folded_count", None)
                            if folded_count is not None:
                                local_active = local_active & (folded_count == 1).unsqueeze(-1)
                if os.environ.get("FRNET_FOLDED_TRANSFER_GUARD", "0") == "1":
                    receivers = getattr(self.entity, "_frnet_folded_transfer_receivers", None)
                    dynamic_receivers = getattr(self.entity, "_frnet_dynamic_transfer_receiver", None)
                    if receivers is not None or dynamic_receivers is not None:
                        receiver_active_full = torch.zeros_like(full_velocity, dtype=torch.bool)
                        if receivers is not None:
                            receiver_active_full = receiver_active_full | receivers.unsqueeze(-1).expand(-1, -1, 3).reshape(receivers.shape[0], -1)
                            if os.environ.get("FRNET_FOLDED_TRANSFER_INCLUDE_SOURCE", "0") == "1":
                                source_legs = getattr(self.entity, "_frnet_folded_legs", None)
                                if source_legs is not None:
                                    receiver_active_full = receiver_active_full | source_legs.unsqueeze(-1).expand(-1, -1, 3).reshape(source_legs.shape[0], -1)
                        if (dynamic_receivers is not None
                                and os.environ.get("FRNET_DYNAMIC_TRANSFER_SUBSTEP", "0") == "1"
                                and not folded_transfer_only):
                            receiver_active_full = receiver_active_full | dynamic_receivers.unsqueeze(-1).expand(-1, -1, 3).reshape(dynamic_receivers.shape[0], -1)
                        receiver_active = receiver_active_full.index_select(-1, ids)
                        transfer_trigger = float(os.environ.get("FRNET_FOLDED_TRANSFER_TRIGGER", "0.0"))
                        if transfer_trigger > 0.0:
                            receiver_active = receiver_active & (local_trigger.index_select(-1, ids) >= transfer_trigger)
                        if os.environ.get("FRNET_SUBSTEP_LOCAL_DEEP_FOLDED_ONLY", "0") == "1":
                            deep_folded = getattr(self.entity, "_frnet_deep_folded", None)
                            if deep_folded is not None:
                                receiver_active = receiver_active & deep_folded.unsqueeze(-1)
                        local_active = local_active | receiver_active
                # Dampen the distal calf strongly, the thigh lightly, and
                # keep the hip available for posture correction.
                base_scale = min(1.0, max(0.0, float(
                    os.environ.get("FRNET_SUBSTEP_LOCAL_TORQUE_SCALE", "0.20")
                )))
                calf_scale = min(1.0, max(0.0, float(
                    os.environ.get("FRNET_SUBSTEP_LOCAL_CALF_SCALE", str(base_scale))
                )))
                thigh_scale = min(1.0, max(0.0, float(
                    os.environ.get("FRNET_SUBSTEP_LOCAL_THIGH_SCALE", str(min(1.0, base_scale + 0.30)))
                )))
                hip_scale = min(1.0, max(0.0, float(
                    os.environ.get("FRNET_SUBSTEP_LOCAL_HIP_SCALE", "1.0")
                )))
                joint_kind = torch.arange(local_active.shape[-1], device=local_active.device) % 3
                transfer_scale_vec = torch.where(
                    joint_kind == 0, local_active.new_tensor(float(os.environ.get("FRNET_FOLDED_TRANSFER_HIP_SCALE", "0.20"))),
                    torch.where(joint_kind == 1, local_active.new_tensor(float(os.environ.get("FRNET_FOLDED_TRANSFER_THIGH_SCALE", "0.05"))),
                                local_active.new_tensor(float(os.environ.get("FRNET_FOLDED_TRANSFER_CALF_SCALE", "0.0")))),
                )
                scale_vec = torch.where(
                    joint_kind == 0, local_active.new_tensor(hip_scale),
                    torch.where(joint_kind == 1, local_active.new_tensor(thigh_scale),
                                local_active.new_tensor(calf_scale)),
                )
                drive = safety_velocity * torque > 0.0
                if os.environ.get("FRNET_FOLDED_TRANSFER_GUARD", "0") == "1":
                    transfer_mask = receiver_active if "receiver_active" in locals() else torch.zeros_like(local_active)
                    effective_scale = torch.where(transfer_mask, transfer_scale_vec, scale_vec)
                else:
                    effective_scale = scale_vec
                torque = torch.where(local_active & drive, effective_scale * torque, torque)
                if os.environ.get("FRNET_FOLDED_TRANSFER_GUARD", "0") == "1":
                    transfer_speed_ratio = safety_velocity.abs() / full_velocity.new_tensor(VELOCITY_LIMIT).index_select(-1, ids).clamp_min(1e-6)
                    brake_min_ratio = float(os.environ.get("FRNET_FOLDED_TRANSFER_BRAKE_MIN_RATIO", "0.02"))
                    brake_active = transfer_mask & (transfer_speed_ratio >= brake_min_ratio)
                    brake_gain = float(os.environ.get("FRNET_FOLDED_TRANSFER_BRAKE_GAIN", "0.50"))
                    joint_kind_full = torch.arange(full_velocity.shape[-1], device=full_velocity.device) % 3
                    hip_gain = float(os.environ.get("FRNET_FOLDED_TRANSFER_BRAKE_HIP_GAIN", str(brake_gain)))
                    thigh_gain = float(os.environ.get("FRNET_FOLDED_TRANSFER_BRAKE_THIGH_GAIN", str(brake_gain)))
                    calf_gain = float(os.environ.get("FRNET_FOLDED_TRANSFER_BRAKE_CALF_GAIN", str(brake_gain)))
                    gain_vec = torch.where(joint_kind_full == 0, full_velocity.new_tensor(hip_gain), torch.where(joint_kind_full == 1, full_velocity.new_tensor(thigh_gain), full_velocity.new_tensor(calf_gain))).index_select(-1, ids)
                    brake = -torch.sign(safety_velocity) * self.force_limit * gain_vec
                    brake_hip = os.environ.get("FRNET_FOLDED_TRANSFER_BRAKE_HIP", "1") == "1"
                    brake_mask = brake_active if brake_hip else (brake_active & (joint_kind_full != 0).index_select(-1, ids))
                    torque = torch.where(
                        brake_mask & (safety_velocity > 0.0),
                        torch.minimum(torque, brake), torque,
                    )
                    torque = torch.where(
                        brake_mask & (safety_velocity < 0.0),
                        torch.maximum(torque, brake), torque,
                    )
                if os.environ.get("FRNET_SUBSTEP_LOCAL_BRAKE", "0") == "1":
                    local_speed_ratio = safety_velocity.abs() / full_velocity.new_tensor(VELOCITY_LIMIT).index_select(-1, ids).clamp_min(1e-6)
                    brake_min_ratio = float(os.environ.get("FRNET_SUBSTEP_LOCAL_BRAKE_MIN_RATIO", "0.02"))
                    brake_gain = float(os.environ.get("FRNET_SUBSTEP_LOCAL_BRAKE_GAIN", "0.20"))
                    brake_active = local_active & (local_speed_ratio >= brake_min_ratio)
                    brake = -torch.sign(safety_velocity) * self.force_limit * brake_gain
                    torque = torch.where(brake_active & (safety_velocity > 0.0), torch.minimum(torque, brake), torque)
                    torque = torch.where(brake_active & (safety_velocity < 0.0), torch.maximum(torque, brake), torque)
                self.entity._frnet_substep_local_guard_active = local_active
                if not hasattr(self.entity, "_frnet_substep_local_guard_counts") or self.entity._frnet_substep_local_guard_counts.shape != full_velocity.shape:
                    self.entity._frnet_substep_local_guard_counts = torch.zeros_like(full_velocity, dtype=torch.int32)
                    self.entity._frnet_folded_transfer_active_counts = torch.zeros_like(full_velocity, dtype=torch.int32)
                self.entity._frnet_substep_local_guard_counts[..., ids] += local_active.to(torch.int32)
                if "receiver_active" in locals():
                    self.entity._frnet_folded_transfer_active_counts[..., ids] += receiver_active.to(torch.int32)
        # Optional whole-body interlock retained for controlled comparisons.
        if os.environ.get("PYTHON_FRNET_SUBSTEP_GLOBAL_GUARD", "0") == "1":
            full_velocity = getattr(self.entity.data, "joint_vel", None)
            if full_velocity is not None and full_velocity.shape[-1] == len(VELOCITY_LIMIT):
                full_limit = full_velocity.new_tensor(VELOCITY_LIMIT)
                substep_trigger = (
                    full_velocity.abs() / full_limit.clamp_min(1e-6)
                ).amax(dim=-1) >= float(
                    os.environ.get("FRNET_SUBSTEP_GLOBAL_TRIGGER", "0.50")
                )
                if substep_trigger.any():
                    scale = min(1.0, max(0.0, float(
                        os.environ.get("FRNET_SUBSTEP_GLOBAL_TORQUE_SCALE", "0.20")
                    )))
                    drive = safety_velocity * torque > 0.0
                    while substep_trigger.ndim < torque.ndim:
                        substep_trigger = substep_trigger.unsqueeze(-1)
                    torque = torch.where(substep_trigger & drive, scale * torque, torque)
                    self.entity._frnet_substep_global_guard_active = substep_trigger
        if self.folded_calf_preload_enabled:
            safety_step=getattr(
                self.entity,"_frnet_safety_step",
                torch.zeros(cmd.pos.shape[:-1],device=cmd.pos.device,dtype=torch.int32),
            )
            torque=apply_folded_calf_preload(
                torque,cmd.pos,self.force_limit,safety_step,
                preload_nm=self.folded_calf_preload_nm,
                preload_steps=self.folded_calf_preload_steps,
            )
        limit = safety_velocity.new_tensor(self.velocity_limit)
        if self.velocity_brake_enabled:
            torque=apply_velocity_brake_torque(
                torque,safety_velocity,self.force_limit,limit,
                trigger_ratio=self.velocity_brake_trigger_ratio,
            )
            result=torque
        else:
            speed = safety_velocity.abs()
            approaching = speed > 0.85 * limit
            scale = ((limit - speed) / (0.15 * limit)).clamp(0.0, 1.0)
            limited = torch.where(safety_velocity * torque > 0, torque * scale, torque)
            braking = -torch.sign(safety_velocity) * self.force_limit
            limited = torch.where(speed >= limit, braking, limited)
            result = torch.where(approaching, limited, torque)
        if self.position_barrier_enabled:
            before_barrier=result
            result,active=apply_position_barrier_torque(
                result,cmd.pos,safety_velocity,self.force_limit,
                limit,
                cmd.pos.new_tensor(self.position_lower),
                cmd.pos.new_tensor(self.position_upper),
                cmd.pos.new_tensor(self.brake_accel),
                margin=self.position_barrier_margin,
                brake_gain=self.position_barrier_brake_gain,
            )
            if self.barrier_active_counts is None:
                self.barrier_active_counts=torch.zeros_like(active,dtype=torch.int32)
                self.barrier_max_torque_delta=torch.zeros_like(result)
            self.barrier_active_counts,self.barrier_max_torque_delta=(
                accumulate_barrier_diagnostics(
                    self.barrier_active_counts,
                    self.barrier_max_torque_delta,
                    active,
                    result-before_barrier,
                )
            )
            self.entity._frnet_position_barrier_active=active
        if self.position_damping_enabled:
            before_barrier=result
            result,active=apply_predictive_position_damping(
                result,cmd.pos,safety_velocity,self.force_limit,
                cmd.pos.new_tensor(self.position_lower),
                cmd.pos.new_tensor(self.position_upper),
                cmd.pos.new_tensor(self.brake_accel),
                margin=self.position_damping_margin,
                damping_gain=self.position_damping_gain,
            )
            if self.barrier_active_counts is None:
                self.barrier_active_counts=torch.zeros_like(active,dtype=torch.int32)
                self.barrier_max_torque_delta=torch.zeros_like(result)
            self.barrier_active_counts,self.barrier_max_torque_delta=(
                accumulate_barrier_diagnostics(
                    self.barrier_active_counts,
                    self.barrier_max_torque_delta,
                    active,
                    result-before_barrier,
                )
            )
            self.entity._frnet_position_barrier_active=active
        # Mildly increase velocity damping only in a clearly upright, low-rate
        # posture; recovery transients retain the calibrated damping.
        gravity = getattr(self.entity.data, "projected_gravity_b", None)
        ang = getattr(self.entity.data, "root_link_ang_vel_b", None)
        if gravity is not None and ang is not None:
            hold = (gravity[..., 2] < -0.90) & (ang.norm(dim=-1) < 0.35)
            while hold.ndim < result.ndim:
                hold = hold.unsqueeze(-1)
            result = result + torch.where(hold, 0.4 * self.damping * (cmd.velocity_target - safety_velocity), torch.zeros_like(result))
        # During the first few physics substeps after reset, contact
        # impulses can arrive before the policy has observed the new state.
        # Temporarily limit actuator effort without freezing the target; this
        # prevents a reset contact impulse from becoming a calf velocity spike.
        startup_scale = min(1.0, max(0.0, float(
            os.environ.get("FRNET_STARTUP_TORQUE_SCALE", "1.0")
        )))
        startup_steps = int(os.environ.get("FRNET_STARTUP_TORQUE_STEPS", "0"))
        safety_step = getattr(self.entity, "_frnet_safety_step", None)
        if startup_steps > 0 and safety_step is not None:
            startup_active = safety_step < startup_steps
            if os.environ.get("FRNET_STARTUP_DEEP_FOLDED_ONLY", "0") == "1":
                deep_folded = getattr(self.entity, "_frnet_deep_folded", None)
                if deep_folded is not None:
                    startup_active = startup_active & deep_folded
                    if os.environ.get("FRNET_STARTUP_SINGLE_FOLDED_ONLY", "0") == "1":
                        folded_count = getattr(self.entity, "_frnet_deep_folded_count", None)
                        if folded_count is not None:
                            startup_active = startup_active & (folded_count == 1)
            # A reset contact can make a neighboring receiver leg move before
            # the source leg is classified.  Keep this startup interlock local
            # to folded receivers so source and support legs remain available
            # for recovery.  This is opt-in for sim2real and targeted checks.
            if os.environ.get("FRNET_STARTUP_FOLDED_RECEIVER_ONLY", "0") == "1":
                receivers = getattr(self.entity, "_frnet_folded_transfer_receivers", None)
                if receivers is not None:
                    receiver_joint = receivers.repeat_interleave(3, dim=-1)
                    ids = self._frnet_target_ids.to(device=receiver_joint.device)
                    startup_active = startup_active.unsqueeze(-1) & receiver_joint.index_select(-1, ids)
            while startup_active.ndim < result.ndim:
                startup_active = startup_active.unsqueeze(-1)
            result = torch.where(startup_active, startup_scale * result, result)
            self.entity._frnet_startup_torque_active = startup_active
        if remaining_allowed_velocity is not None:
            result, torque_active = apply_remaining_braking_torque(
                result, safety_velocity, remaining_allowed_velocity,
                self.force_limit, limit, gain=self.remaining_brake_torque_gain,
                base_gain=self.remaining_brake_torque_base_gain,
            )
            self.entity._frnet_remaining_brake_torque_active = torque_active
        # When the action layer detects a whole-body impact, attenuate only
        # torques that would continue the current joint motion.  This is a
        # dissipative interlock and never commands a reverse torque.
        global_guard = getattr(self.entity, "_frnet_global_impact_guard_active", None)
        if global_guard is not None:
            active = global_guard
            if active.ndim == 2 and active.shape[-1] == 4:
                joint_active = active.repeat_interleave(3, dim=-1)
                ids = self._frnet_target_ids.to(device=joint_active.device)
                active = joint_active.index_select(-1, ids)
            while active.ndim < result.ndim:
                active = active.unsqueeze(-1)
            drive = safety_velocity * result > 0.0
            before_global = result
            global_torque_scale = min(
                1.0, max(0.0, float(os.environ.get("FRNET_GLOBAL_IMPACT_TORQUE_SCALE", "0.5")))
            )
            result = torch.where(active & drive, global_torque_scale * result, result)
            self.entity._frnet_global_impact_torque_active = (result != before_global)
        return result

@dataclass(kw_only=True)
class SafeIdealPdActuatorCfg(IdealPdActuatorCfg):
    velocity_limit: float
    velocity_brake_enabled: bool = False
    velocity_brake_trigger_ratio: float = 0.85
    position_barrier_enabled: bool = False
    position_lower: tuple[float,...] = ()
    position_upper: tuple[float,...] = ()
    brake_accel: float = 20.0
    position_barrier_margin: float = 0.05
    position_barrier_brake_gain: float = 0.25
    position_damping_enabled: bool = False
    position_damping_margin: float = 0.12
    position_damping_gain: float = 2.0
    folded_calf_preload_enabled: bool = False
    folded_calf_preload_nm: float = 0.0
    folded_calf_preload_steps: int = 0
    remaining_brake_enabled: bool = False
    remaining_brake_inertia: float = 0.10
    remaining_brake_safety_factor: float = 0.10
    remaining_brake_dt: float = 0.02
    remaining_brake_max_target_step: float = 0.015
    remaining_brake_torque_gain: float = 1.0
    remaining_brake_torque_base_gain: float = 0.25
    def build(self, entity, target_ids, target_names):
        return SafeIdealPdActuator(
            self,entity,target_ids,target_names,self.velocity_limit,
            self.velocity_brake_enabled,self.velocity_brake_trigger_ratio,
            self.position_barrier_enabled,self.position_lower,self.position_upper,
            self.brake_accel,self.position_barrier_margin,
            self.position_barrier_brake_gain,
            self.position_damping_enabled,self.position_damping_margin,
            self.position_damping_gain,
            self.folded_calf_preload_enabled,self.folded_calf_preload_nm,
            self.folded_calf_preload_steps,
            self.remaining_brake_enabled,self.remaining_brake_inertia,
            self.remaining_brake_safety_factor,self.remaining_brake_dt,
            self.remaining_brake_max_target_step,
            self.remaining_brake_torque_gain,
            self.remaining_brake_torque_base_gain,
        )

def get_robot_cfg(xml_path):
    path = Path(xml_path).resolve()
    effort_derate = float(os.environ.get("FRNET_EFFORT_DERATE", "1.0"))
    velocity_derate = float(os.environ.get("FRNET_VELOCITY_DERATE", "1.0"))
    if not 0.50 <= effort_derate <= 1.0:
        raise ValueError("FRNET_EFFORT_DERATE must be in [0.50, 1.0]")
    if not 0.50 <= velocity_derate <= 1.0:
        raise ValueError("FRNET_VELOCITY_DERATE must be in [0.50, 1.0]")
    official_passive_dynamics=(
        os.environ.get("PYTHON_FRNET_OFFICIAL_PASSIVE_DYNAMICS","0") == "1"
    )
    hard_joint_limits=(
        os.environ.get("PYTHON_FRNET_HARD_JOINT_LIMITS","0") == "1"
    )
    velocity_brake_enabled=(
        os.environ.get("PYTHON_FRNET_ACTUATOR_VELOCITY_BRAKE","0") == "1"
    )
    position_barrier_enabled=(
        os.environ.get("PYTHON_FRNET_ACTUATOR_POSITION_BARRIER","0") == "1"
    )
    position_damping_enabled=(
        os.environ.get("PYTHON_FRNET_ACTUATOR_POSITION_DAMPING","0") == "1"
    )
    remaining_brake_enabled=(
        os.environ.get("PYTHON_FRNET_REMAINING_BRAKE","0") == "1"
    )
    remaining_brake_inertia=float(os.environ.get("FRNET_REMAINING_BRAKE_INERTIA","0.10"))
    remaining_brake_safety_factor=float(os.environ.get("FRNET_REMAINING_BRAKE_SAFETY_FACTOR","0.10"))
    remaining_brake_dt=float(os.environ.get("FRNET_REMAINING_BRAKE_DT","0.02"))
    remaining_brake_max_target_step=float(os.environ.get("FRNET_REMAINING_BRAKE_MAX_TARGET_STEP","0.015"))
    remaining_brake_torque_gain=float(os.environ.get("FRNET_REMAINING_BRAKE_TORQUE_GAIN","1.0"))
    remaining_brake_torque_base_gain=float(os.environ.get("FRNET_REMAINING_BRAKE_TORQUE_BASE_GAIN","0.25"))
    if remaining_brake_inertia <= 0.0 or not 0.0 < remaining_brake_safety_factor <= 1.0:
        raise ValueError("remaining braking inertia/factor are invalid")
    if remaining_brake_dt <= 0.0 or remaining_brake_max_target_step <= 0.0:
        raise ValueError("remaining braking dt/target step are invalid")
    if not 0.0 <= remaining_brake_torque_gain <= 2.0:
        raise ValueError("remaining braking torque gain must be in [0, 2]")
    if not 0.0 <= remaining_brake_torque_base_gain <= remaining_brake_torque_gain:
        raise ValueError("remaining braking base gain must be in [0, max gain]")
    def spec_fn():
        spec = mujoco.MjSpec.from_file(str(path))
        for name, lower, upper in zip(JOINT_NAMES, POSITION_LOWER, POSITION_UPPER):
            joint = spec.joint(name)
            if joint is None:
                raise ValueError(f"Go2 joint missing from aligned URDF asset: {name}")
            joint.range = (lower, upper)
            if official_passive_dynamics:
                # Match src/assets/robots/unitree_go2/xmls/scene_go2.xml.
                joint.damping=0.1
            if hard_joint_limits:
                configure_joint_limit_constraint(
                    joint,
                    margin=float(os.environ.get("FRNET_JOINT_LIMIT_MARGIN","0.02")),
                    timeconst=float(os.environ.get("FRNET_JOINT_LIMIT_TIMECONST","0.01")),
                    dampratio=float(os.environ.get("FRNET_JOINT_LIMIT_DAMPRATIO","1.0")),
                    impedance_min=float(os.environ.get("FRNET_JOINT_LIMIT_IMPEDANCE_MIN","0.95")),
                    impedance_max=float(os.environ.get("FRNET_JOINT_LIMIT_IMPEDANCE_MAX","0.99")),
                )
        # Collision-only source geometry; give every geometry a stable name.
        for body in spec.bodies:
            for i, geom in enumerate(body.geoms):
                geom.name = f'{body.name}_collision_{i}'
        return spec
    calf_stiffness=float(os.environ.get("FRNET_ACTUATOR_CALF_STIFFNESS", "30.0"))
    general_stiffness=float(os.environ.get("FRNET_ACTUATOR_GENERAL_STIFFNESS", "30.0"))
    if not 1.0 <= calf_stiffness <= 120.0 or not 1.0 <= general_stiffness <= 120.0:
        raise ValueError("FRNET actuator stiffness must be in [1, 120]")
    return EntityCfg(
        spec_fn=spec_fn,
        init_state=EntityCfg.InitialStateCfg(pos=(0.,0.,.6),
            joint_pos={'.*hip_joint':0., '.*thigh_joint':1.5, '.*calf_joint':-2.4},
            joint_vel={'.*':0.}),
        articulation=EntityArticulationInfoCfg(soft_joint_pos_limit_factor=.9,
            actuators=tuple(SafeIdealPdActuatorCfg(target_names_expr=(f'.*{part}_joint',),
                stiffness=(calf_stiffness if part == 'calf' else general_stiffness), damping=actuator_damping_for_part(part), effort_limit=effort,
                frictionloss=0.2 if official_passive_dynamics else 0.0,
                armature=actuator_armature_for_part(part,official_passive_dynamics),
                velocity_limit={'hip':30.1,'thigh':30.1,'calf':15.70}[part] * velocity_derate,
                velocity_brake_enabled=velocity_brake_enabled,
                velocity_brake_trigger_ratio=float(os.environ.get(
                    "FRNET_ACTUATOR_CALF_BRAKE_TRIGGER" if part == "calf"
                    else "FRNET_ACTUATOR_GENERAL_BRAKE_TRIGGER",
                    "0.30" if part == "calf" else "0.50",
                )),
                position_barrier_enabled=position_barrier_enabled,
                position_lower=tuple(POSITION_LOWER[i] for i in range(
                    {'hip':0,'thigh':1,'calf':2}[part],12,3
                )),
                position_upper=tuple(POSITION_UPPER[i] for i in range(
                    {'hip':0,'thigh':1,'calf':2}[part],12,3
                )),
                brake_accel={'hip':35.0,'thigh':35.0,'calf':18.0}[part],
                position_barrier_margin=float(os.environ.get(
                    "FRNET_ACTUATOR_POSITION_MARGIN","0.05"
                )),
                position_barrier_brake_gain=float(os.environ.get(
                    "FRNET_ACTUATOR_POSITION_BRAKE_GAIN","0.25"
                )),
                position_damping_enabled=position_damping_enabled,
                position_damping_margin=float(os.environ.get(
                    "FRNET_ACTUATOR_POSITION_DAMPING_MARGIN","0.12"
                )),
                position_damping_gain=float(os.environ.get(
                    "FRNET_ACTUATOR_POSITION_DAMPING_GAIN","2.0"
                )),
                folded_calf_preload_enabled=(
                    part == "calf" and
                    os.environ.get("PYTHON_FRNET_FOLDED_CALF_PRELOAD","0") == "1"
                ),
                folded_calf_preload_nm=float(os.environ.get(
                    "FRNET_FOLDED_CALF_PRELOAD_NM","0.0"
                )),
                folded_calf_preload_steps=int(os.environ.get(
                    "FRNET_FOLDED_CALF_PRELOAD_STEPS","1"
                )),
                remaining_brake_enabled=remaining_brake_enabled,
                remaining_brake_inertia=remaining_brake_inertia,
                remaining_brake_safety_factor=remaining_brake_safety_factor_for_part(
                    part, remaining_brake_safety_factor
                ),
                remaining_brake_dt=remaining_brake_dt,
                remaining_brake_max_target_step=remaining_brake_max_target_step,
                remaining_brake_torque_gain=remaining_brake_torque_gain,
                remaining_brake_torque_base_gain=remaining_brake_torque_base_gain)
                for part,effort in (('hip',FRNET_EFFORT_LIMIT[0] * effort_derate),
                                    ('thigh',FRNET_EFFORT_LIMIT[1] * effort_derate),
                                    ('calf',FRNET_EFFORT_LIMIT[2] * effort_derate)))))

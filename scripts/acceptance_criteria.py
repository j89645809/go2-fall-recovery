"""Versioned, scenario-based recovery criteria; classification uses initial case only."""
import math
import torch

def case_criteria(case, terrain_name, profile="tiered"):
    if profile not in ("strict", "tiered", "stable"):
        raise ValueError(profile)
    if profile == "stable":
        return "stable_sim_calibrated", 0.90, 0.26
    if profile == "strict":
        return "standard", 0.90, 0.30
    if terrain_name != "flat":
        return "difficult_terrain", math.cos(math.radians(35)), 0.26
    if case["pose"] == "left" or case["angle_deg"] >= 120 or case["fold"] != "none":
        return "difficult_pose", math.cos(math.radians(30)), 0.28
    return "standard", 0.90, 0.30

class SuccessCounter:
    def __init__(self, size, device, hold_steps):
        if hold_steps < 1:
            raise ValueError("hold_steps must be positive")
        self.hold_steps = hold_steps
        self.hold = torch.zeros(size, dtype=torch.long, device=device)
        self.first = torch.full((size,), -1, dtype=torch.long, device=device)
    def update(self, ok, step):
        self.hold = torch.where(ok, self.hold+1, torch.zeros_like(self.hold))
        self.first = torch.where((self.first < 0) & (self.hold >= self.hold_steps), step, self.first)


def stable_recovery_ok(*, height, gravity_z, linear_speed, angular_speed,
                       joint_speed, loaded_feet, foot_support_ratio):
    """Simulation-calibrated provisional stand gate, not hardware calibration."""
    return ((height > 0.26) & (gravity_z < -0.90) & (linear_speed < 0.15)
            & (angular_speed < 0.50) & (joint_speed < 1.0)
            & (loaded_feet >= 3) & (foot_support_ratio >= 0.80))


class StableRecoveryCounter(SuccessCounter):
    """Require a terminal stable window and veto a confirmed fall after recovery."""
    def __init__(self, size, device, hold_steps, refall_steps):
        super().__init__(size, device, hold_steps)
        if refall_steps < 1:
            raise ValueError("refall_steps must be positive")
        self.refall_steps = refall_steps
        self.refall_run = torch.zeros(size, dtype=torch.long, device=device)
        self.refell = torch.zeros(size, dtype=torch.bool, device=device)
    def update(self, ok, major_fall, step):
        previously_recovered = self.first >= 0
        self.refall_run = torch.where(previously_recovered & major_fall,
                                      self.refall_run+1, torch.zeros_like(self.refall_run))
        self.refell |= self.refall_run >= self.refall_steps
        super().update(ok, step)
    @property
    def success(self):
        return (self.hold >= self.hold_steps) & ~self.refell

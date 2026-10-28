import os
import unittest
from unittest.mock import patch

import torch

from src.tasks.velocity.config.go2_recovery_aligned.aligned_actions import (
    apply_global_impact_guard,
    classify_impact_source,
    interpolate_reset_target,
    reset_contact_force_history,
    limit_target_slew,
    apply_predictive_target_brake,
    apply_source_leg_coordination,
    filter_dynamic_contact_rise,
)
from src.tasks.velocity.config.go2_recovery_aligned.aligned_asset import (
    remaining_brake_safety_factor_for_part,
)


class ImpactGuardEventModeTest(unittest.TestCase):
    def test_static_reset_contact_is_not_dynamic_impact(self):
        rising = torch.tensor([[True, False]])
        force = torch.tensor([[30.0, 2.0]])
        armed = torch.zeros((1, 2), dtype=torch.bool)
        age = torch.tensor([0], dtype=torch.int32)
        dynamic, next_armed = filter_dynamic_contact_rise(
            rising, force, armed, age, threshold=25.0,
            release_threshold=10.0, warmup_steps=20, enabled=True,
        )
        self.assertFalse(bool(dynamic[0, 0]))
        self.assertFalse(bool(next_armed[0, 0]))

    def test_recontact_after_unload_is_dynamic_impact(self):
        rising = torch.tensor([[True]])
        force = torch.tensor([[30.0]])
        armed = torch.tensor([[True]])
        age = torch.tensor([25], dtype=torch.int32)
        dynamic, _ = filter_dynamic_contact_rise(
            rising, force, armed, age, threshold=25.0,
            release_threshold=10.0, warmup_steps=20, enabled=True,
        )
        self.assertTrue(bool(dynamic[0, 0]))
    def test_predictive_brake_only_shortens_overshoot_target(self):
        q = torch.tensor([[0.0, 0.0]])
        velocity = torch.tensor([[2.0, 0.0]])
        desired = torch.tensor([[1.0, 1.0]])
        out = apply_predictive_target_brake(
            q, velocity, desired,
            torch.tensor([-3.0, -3.0]), torch.tensor([1.0, 3.0]),
            torch.tensor([2.0, 2.0]), torch.tensor([1.0, 1.0]), 0.02,
            margin=0.05, trigger_ratio=0.3, gain=0.5,
        )
        self.assertLess(float(out[0, 0]), 1.0)
        self.assertAlmostEqual(float(out[0, 1]), 1.0, places=6)

    def test_target_slew_limits_general_and_calf_steps(self):
        q = torch.zeros((1, 12))
        desired = torch.ones_like(q)
        out = limit_target_slew(q, desired, 0.20, calf_max_step=0.10)
        self.assertAlmostEqual(float(out[0, 0]), 0.20, places=6)
        self.assertAlmostEqual(float(out[0, 2]), 0.10, places=6)
        self.assertTrue(torch.all(out >= 0.0))

    def test_event_mode_identifies_accelerating_source_leg(self):
        q = torch.zeros((1, 12))
        velocity = torch.zeros_like(q)
        velocity[:, 2] = 4.0
        proposed = q.clone()
        proposed[:, 2] = 0.2
        lower = torch.full((12,), -3.0)
        upper = torch.full((12,), 3.0)
        limits = torch.full((12,), 10.0)
        previous_counter = torch.zeros((1, 4), dtype=torch.int32)
        velocity_delta = torch.zeros_like(q)
        velocity_delta[:, 2] = 1.0

        with patch.dict(
            os.environ,
            {
                "FRNET_GLOBAL_IMPACT_EVENT_MODE": "1",
                "FRNET_GLOBAL_IMPACT_EVENT_MIN_RATIO": "0.15",
                "FRNET_GLOBAL_IMPACT_EVENT_HIGH_RATIO": "0.60",
            },
            clear=False,
        ):
            _, _, _, trigger, _ = apply_global_impact_guard(
                q,
                velocity,
                proposed,
                lower,
                upper,
                limits,
                previous_counter,
                velocity_delta=velocity_delta,
                acceleration_threshold=0.5,
            )

        self.assertEqual(trigger.tolist(), [[True, False, False, False]])

    def test_prearm_soft_damping_acts_before_causal_source_trigger(self):
        q = torch.zeros((1, 12))
        q[:, 6] = 2.8
        velocity = torch.zeros_like(q)
        velocity[:, 6] = 1.0
        proposed = q.clone()
        proposed[:, 6] = 3.0
        lower = torch.full((12,), -3.0)
        upper = torch.full((12,), 3.0)
        limits = torch.full((12,), 3.0)
        previous_counter = torch.zeros((1, 4), dtype=torch.int32)
        velocity_delta = torch.zeros_like(q)
        velocity_delta[:, 6] = 1.0
        with patch.dict(os.environ, {
            "FRNET_GLOBAL_IMPACT_PREARM": "1",
            "FRNET_GLOBAL_IMPACT_PREARM_MIN_RATIO": "0.15",
            "FRNET_GLOBAL_IMPACT_PREARM_ACCEL_THRESHOLD": "0.25",
            "FRNET_GLOBAL_IMPACT_PREARM_SCALE": "0.85",
        }, clear=False):
            guarded, _, _, _, _ = apply_global_impact_guard(
                q, velocity, proposed, lower, upper, limits,
                previous_counter, velocity_delta=velocity_delta,
                acceleration_threshold=0.5,
            )
        self.assertLess(float(guarded[0, 6]), 3.0)

    def test_predictive_source_guard_triggers_before_boundary(self):
        q = torch.zeros((1, 12))
        velocity = torch.zeros_like(q)
        velocity[:, 2] = 3.0
        proposed = q.clone(); proposed[:, 2] = 1.0
        delta = torch.zeros_like(q); delta[:, 2] = 0.1
        with patch.dict(os.environ, {
            "FRNET_GLOBAL_IMPACT_PREDICTIVE_STOP": "1",
            "FRNET_GLOBAL_IMPACT_PREDICTIVE_ACCEL": "1.5",
        }, clear=False):
            guarded, _, active, trigger, _ = apply_global_impact_guard(
                q, velocity, proposed,
                torch.full((12,), -3.0), torch.full((12,), 3.0),
                torch.full((12,), 10.0), torch.zeros((1, 4), dtype=torch.int32),
                velocity_delta=delta,
                contact_leg_trigger=torch.tensor([[True, False, False, False]]),
            )
        self.assertTrue(bool(trigger[0, 0]))
        self.assertTrue(bool(active[0, 0]))
        self.assertLess(float(guarded[0, 2]), 1.0)

    def test_reset_contact_force_history_clears_only_reset_envs(self):
        history = torch.tensor([[20.0, 30.0], [40.0, 50.0]])
        reset_contact_force_history(history, torch.tensor([1]))
        self.assertEqual(history.tolist(), [[20.0, 30.0], [0.0, 0.0]])

    def test_static_contact_does_not_become_impact_source(self):
        q = torch.zeros((1, 12))
        velocity = torch.zeros_like(q)
        proposed = q.clone()
        delta = torch.zeros_like(q)
        limits = torch.full((12,), 10.0)
        previous_streak = torch.zeros((1, 4), dtype=torch.int32)
        out = classify_impact_source(
            q,
            velocity,
            proposed,
            torch.full((12,), -3.0),
            torch.full((12,), 3.0),
            limits,
            delta,
            torch.tensor([[True, False, False, False]]),
            previous_streak,
        )
        self.assertFalse(bool(out["raw_event"].any()))
        self.assertFalse(bool(out["trigger"].any()))
        self.assertEqual(out["source_leg"].tolist(), [-1])

    def test_strict_source_requires_dynamic_contact_evidence(self):
        q = torch.zeros((1, 12))
        velocity = torch.zeros_like(q); velocity[:, 2] = 2.0
        proposed = q.clone(); proposed[:, 2] = 0.5
        limits = torch.full((12,), 10.0)
        with patch.dict(os.environ, {"FRNET_IMPACT_SOURCE_STRICT": "1"}, clear=False):
            out = classify_impact_source(
                q, velocity, proposed,
                torch.full((12,), -3.0), torch.full((12,), 3.0), limits,
                torch.zeros_like(q), torch.tensor([[True, False, False, False]]),
                torch.zeros((1, 4), dtype=torch.int32), min_consecutive=1,
            )
        self.assertFalse(bool(out["raw_event"].any()))

    def test_source_requires_two_consecutive_motion_events(self):
        q = torch.zeros((1, 12))
        velocity = torch.zeros_like(q)
        velocity[:, 2] = 2.0
        proposed = q.clone()
        proposed[:, 2] = 0.5
        delta = torch.zeros_like(q)
        delta[:, 2] = 0.5
        lower = torch.full((12,), -3.0)
        upper = torch.full((12,), 3.0)
        limits = torch.full((12,), 10.0)
        contact_rise = torch.tensor([[True, False, False, False]])
        streak = torch.zeros((1, 4), dtype=torch.int32)
        first = classify_impact_source(
            q, velocity, proposed, lower, upper, limits, delta,
            contact_rise, streak, min_consecutive=2,
        )
        self.assertFalse(bool(first["trigger"].any()))
        second = classify_impact_source(
            q, velocity, proposed, lower, upper, limits, delta,
            contact_rise, first["streak"], min_consecutive=2,
        )
        self.assertTrue(bool(second["trigger"][0, 0]))
        self.assertEqual(second["source_leg"].tolist(), [0])

    def test_reset_target_interpolation_starts_at_static_pose(self):
        start = torch.zeros((1, 3))
        reference = torch.full((1, 3), 1.0)
        self.assertTrue(torch.equal(interpolate_reset_target(start, reference, 0, 20), start))
        self.assertTrue(torch.equal(interpolate_reset_target(start, reference, 20, 20), reference))

    def test_source_leg_coordination_preserves_hip_target(self):
        q = torch.zeros((1, 12))
        velocity = torch.ones_like(q)
        proposed = torch.full_like(q, 1.0)
        lower = torch.full((12,), -3.0)
        upper = torch.full((12,), 3.0)
        limits = torch.full((12,), 10.0)
        counters = torch.zeros((1, 4), dtype=torch.int32)
        with patch.dict(os.environ, {"FRNET_GLOBAL_IMPACT_PRESERVE_HIP": "1"}, clear=False):
            guarded, _, _, _, _ = apply_global_impact_guard(
                q, velocity, proposed, lower, upper, limits, counters,
                cooldown_steps=3, cross_joint_scale=0.35,
                contact_leg_trigger=torch.tensor([[True, False, False, False]]),
            )
        self.assertAlmostEqual(float(guarded[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(guarded[0, 1]), 0.35, places=6)

    def test_source_leg_coordination_only_damps_same_leg_motion(self):
        q = torch.zeros((1, 12))
        velocity = torch.ones_like(q)
        desired = torch.ones_like(q)
        # RL is leg index 2; its hip remains unchanged, while thigh/calf are
        # shortened. Other legs retain the policy target.
        out = apply_source_leg_coordination(
            q, velocity, desired,
            torch.tensor([2]), torch.tensor([8]), torch.tensor([True]),
            calf_scale=0.50, thigh_scale=0.75,
        )
        self.assertAlmostEqual(float(out[0, 6]), 1.0, places=6)
        self.assertAlmostEqual(float(out[0, 7]), 0.75, places=6)
        self.assertAlmostEqual(float(out[0, 8]), 0.50, places=6)
        self.assertAlmostEqual(float(out[0, 0]), 1.0, places=6)

    def test_remaining_brake_factor_can_be_set_per_joint_group(self):
        with patch.dict(os.environ, {"FRNET_REMAINING_BRAKE_HIP_SAFETY_FACTOR": "0.03"}, clear=False):
            self.assertAlmostEqual(
                remaining_brake_safety_factor_for_part("hip", 0.10), 0.03, places=6
            )
        self.assertAlmostEqual(
            remaining_brake_safety_factor_for_part("calf", 0.10), 0.10, places=6
        )

    def test_transferred_impact_can_brake_receiving_hip(self):
        q = torch.zeros((1, 12))
        velocity = torch.zeros_like(q)
        velocity[:, 9] = 1.0
        proposed = q.clone()
        proposed[:, 9] = 1.0
        delta = torch.zeros_like(q)
        delta[:, 9] = 1.0
        lower = torch.full((12,), -3.0)
        upper = torch.full((12,), 3.0)
        limits = torch.full((12,), 10.0)
        counters = torch.tensor([[0, 0, 1, 0]], dtype=torch.int32)
        with patch.dict(
            os.environ,
            {
                "FRNET_GLOBAL_IMPACT_TRANSFER_GUARD": "1",
                "FRNET_GLOBAL_IMPACT_TRANSFER_REQUIRE_PREVIOUS": "1",
                "FRNET_GLOBAL_IMPACT_TRANSFER_INCLUDE_HIP": "1",
                "FRNET_GLOBAL_IMPACT_TRANSFER_ACCEL_THRESHOLD": "0.4",
                "FRNET_GLOBAL_IMPACT_TRANSFER_SCALE": "0.35",
            },
            clear=False,
        ):
            guarded, _, _, _, _ = apply_global_impact_guard(
                q, velocity, proposed, lower, upper, limits, counters,
                cooldown_steps=3, cross_joint_scale=0.35,
                velocity_delta=delta,
            )
        self.assertAlmostEqual(float(guarded[0, 9]), 0.35, places=6)

    def test_disabled_soft_acceleration_does_not_activate_leg(self):
        q = torch.zeros((1, 12))
        velocity = torch.zeros_like(q)
        velocity[:, 9] = 1.0
        proposed = q.clone()
        proposed[:, 9] = 1.0
        delta = torch.zeros_like(q)
        delta[:, 9] = 1.0
        with patch.dict(os.environ, {"FRNET_GLOBAL_IMPACT_SOFT_ACCEL": "0"}, clear=False):
            _, _, active, trigger, _ = apply_global_impact_guard(
                q, velocity, proposed, torch.full((12,), -3.0),
                torch.full((12,), 3.0), torch.full((12,), 10.0),
                torch.zeros((1, 4), dtype=torch.int32),
                velocity_delta=delta, acceleration_threshold=0.4,
            )
        self.assertFalse(bool(active.any()))
        self.assertFalse(bool(trigger.any()))

    def test_transfer_speed_signal_is_joint_local_and_pair_filtered(self):
        q = torch.zeros((1, 12))
        velocity = torch.zeros_like(q)
        # FR hip is the receiving joint for an active RR source.
        velocity[:, 3] = 4.0
        velocity[:, 4:6] = 1.0
        proposed = q.clone()
        proposed[:, 3:6] = 1.0
        delta = torch.zeros_like(q)
        lower = torch.full((12,), -3.0)
        upper = torch.full((12,), 3.0)
        limits = torch.full((12,), 10.0)
        counters = torch.tensor([[0, 0, 0, 1]], dtype=torch.int32)
        with patch.dict(os.environ, {
            "FRNET_GLOBAL_IMPACT_TRANSFER_GUARD": "1",
            "FRNET_GLOBAL_IMPACT_TRANSFER_REQUIRE_PREVIOUS": "1",
            "FRNET_GLOBAL_IMPACT_TRANSFER_INCLUDE_HIP": "1",
            "FRNET_GLOBAL_IMPACT_TRANSFER_SPEED_RATIO": "0.30",
            "FRNET_GLOBAL_IMPACT_TRANSFER_ACCEL_THRESHOLD": "100.0",
            "FRNET_GLOBAL_IMPACT_TRANSFER_COORDINATE": "1",
            "FRNET_GLOBAL_IMPACT_TRANSFER_NEIGHBOR_MODE": "longitudinal",
            "FRNET_GLOBAL_IMPACT_TRANSFER_SCALE": "0.35",
        }, clear=False):
            guarded, _, _, _, _ = apply_global_impact_guard(
                q, velocity, proposed, lower, upper, limits, counters,
                cooldown_steps=3, cross_joint_scale=0.35,
                velocity_delta=delta,
            )
        self.assertAlmostEqual(float(guarded[0, 3]), 0.35, places=6)
        # The same-leg thigh/calf are coordinated, while the unrelated FL
        # receiving side is untouched.
        self.assertAlmostEqual(float(guarded[0, 4]), 0.35, places=6)
        self.assertAlmostEqual(float(guarded[0, 5]), 0.35, places=6)
        self.assertAlmostEqual(float(guarded[0, 0]), 0.0, places=6)

    def test_source_classifier_can_enable_narrow_longitudinal_transfer(self):
        q = torch.zeros((1, 12))
        velocity = torch.zeros_like(q)
        # RL hip is the receiving joint for an RR source event.
        velocity[:, 6] = 4.0
        proposed = q.clone(); proposed[:, 6] = 1.0
        delta = torch.zeros_like(q); delta[:, 6] = 1.0
        counters = torch.zeros((1, 4), dtype=torch.int32)
        with patch.dict(os.environ, {
            "FRNET_GLOBAL_IMPACT_TRANSFER_GUARD": "1",
            "FRNET_GLOBAL_IMPACT_TRANSFER_USE_SOURCE": "1",
            "FRNET_GLOBAL_IMPACT_TRANSFER_REQUIRE_PREVIOUS": "0",
            "FRNET_GLOBAL_IMPACT_TRANSFER_INCLUDE_HIP": "1",
            "FRNET_GLOBAL_IMPACT_TRANSFER_COORDINATE": "0",
            "FRNET_GLOBAL_IMPACT_TRANSFER_NEIGHBOR_MODE": "lateral",
            "FRNET_GLOBAL_IMPACT_TRANSFER_EXTRA_PAIRS": "RR>RL",
            "FRNET_GLOBAL_IMPACT_TRANSFER_SPEED_RATIO": "0.30",
            "FRNET_GLOBAL_IMPACT_TRANSFER_SCALE": "0.35",
        }, clear=False):
            guarded, _, _, _, _ = apply_global_impact_guard(
                q, velocity, proposed,
                torch.full((12,), -3.0), torch.full((12,), 3.0),
                torch.full((12,), 10.0), counters,
                velocity_delta=delta,
                contact_leg_trigger=torch.tensor([[False, False, False, True]]),
            )
        self.assertAlmostEqual(float(guarded[0, 6]), 0.35, places=6)

    def test_extra_transfer_edge_is_narrow(self):
        q = torch.zeros((1, 12))
        velocity = torch.zeros_like(q)
        velocity[:, 3] = 4.0
        proposed = q.clone(); proposed[:, 3] = 1.0
        delta = torch.zeros_like(q)
        counters = torch.tensor([[0, 0, 0, 1]], dtype=torch.int32)
        with patch.dict(os.environ, {
            "FRNET_GLOBAL_IMPACT_TRANSFER_GUARD": "1",
            "FRNET_GLOBAL_IMPACT_TRANSFER_REQUIRE_PREVIOUS": "1",
            "FRNET_GLOBAL_IMPACT_TRANSFER_INCLUDE_HIP": "1",
            "FRNET_GLOBAL_IMPACT_TRANSFER_SPEED_RATIO": "0.30",
            "FRNET_GLOBAL_IMPACT_TRANSFER_ACCEL_THRESHOLD": "100.0",
            "FRNET_GLOBAL_IMPACT_TRANSFER_NEIGHBOR_MODE": "lateral",
            "FRNET_GLOBAL_IMPACT_TRANSFER_EXTRA_PAIRS": "RR>FR",
            "FRNET_GLOBAL_IMPACT_TRANSFER_COORDINATE": "1",
            "FRNET_GLOBAL_IMPACT_TRANSFER_SCALE": "0.35",
        }, clear=False):
            guarded, _, _, _, _ = apply_global_impact_guard(
                q, velocity, proposed, torch.full((12,), -3.0),
                torch.full((12,), 3.0), torch.full((12,), 10.0), counters,
                cooldown_steps=3, cross_joint_scale=0.35,
                velocity_delta=delta,
            )
        self.assertAlmostEqual(float(guarded[0, 3]), 0.35, places=6)

    def test_causal_mode_blocks_unqualified_leg(self):
        q = torch.zeros((1, 12))
        velocity = torch.zeros_like(q)
        velocity[:, 0] = 9.0
        proposed = q.clone(); proposed[:, 0] = 1.0
        counters = torch.zeros((1, 4), dtype=torch.int32)
        with patch.dict(os.environ, {"FRNET_GLOBAL_IMPACT_CAUSAL_MODE": "1"}, clear=False):
            _, _, active, trigger, _ = apply_global_impact_guard(
                q, velocity, proposed, torch.full((12,), -3.0),
                torch.full((12,), 3.0), torch.full((12,), 10.0), counters,
                contact_leg_trigger=torch.tensor([[False, True, False, False]]),
            )
        self.assertFalse(bool(trigger[0, 0]))
        self.assertFalse(bool(active[0, 0]))


if __name__ == "__main__":
    unittest.main()

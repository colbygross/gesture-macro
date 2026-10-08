"""
Unit tests for MotionTracker (Swipes, Circles, Cooldowns, and Trajectories).
"""

import sys
import unittest
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.motion import MotionTracker


class TestMotionTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = MotionTracker(
            buffer_size=36,
            min_displacement=0.25,
            min_velocity=0.35,
            cooldown_seconds=0.5,
            min_duration_seconds=0.15,
            max_duration_seconds=0.85,
            require_starting_anchor=True,
            max_anchor_speed=0.25,
        )

    def test_swipe_right_with_starting_anchor(self):
        """Simulates pausing at starting position, then an exaggerated rightward swipe."""
        start_time = 100.0
        result = None
        # 1. Starting anchor pause for 4 frames
        for i in range(4):
            t = start_time + i * 0.025
            self.tracker.update(0.25, 0.50, timestamp=t)
        # 2. Exaggerated swipe from x=0.25 to x=0.65 (disp=0.40) in 10 frames
        for i in range(1, 11):
            x = 0.25 + (i / 10) * 0.40
            y = 0.50 + np.sin(i) * 0.005
            t = start_time + (3 + i) * 0.025
            res = self.tracker.update(x, y, timestamp=t)
            if res:
                result = res

        self.assertEqual(result, "swipe_right")

    def test_swipe_left_with_starting_anchor(self):
        """Simulates pausing at starting position, then an exaggerated leftward swipe."""
        start_time = 200.0
        result = None
        # 1. Starting anchor pause
        for i in range(4):
            t = start_time + i * 0.025
            self.tracker.update(0.75, 0.50, timestamp=t)
        # 2. Exaggerated swipe from x=0.75 to x=0.25 (disp=0.50)
        for i in range(1, 11):
            x = 0.75 - (i / 10) * 0.50
            y = 0.50
            t = start_time + (3 + i) * 0.025
            res = self.tracker.update(x, y, timestamp=t)
            if res:
                result = res

        self.assertEqual(result, "swipe_left")

    def test_swipe_up_with_starting_anchor(self):
        """Simulates pausing at starting position, then an exaggerated upward swipe."""
        self.tracker.enable_vertical_swipes = True
        start_time = 300.0
        result = None
        # 1. Starting anchor pause
        for i in range(4):
            t = start_time + i * 0.025
            self.tracker.update(0.50, 0.75, timestamp=t)
        # 2. Exaggerated swipe from y=0.75 to y=0.25 (disp=0.50)
        for i in range(1, 11):
            x = 0.50
            y = 0.75 - (i / 10) * 0.50
            t = start_time + (3 + i) * 0.025
            res = self.tracker.update(x, y, timestamp=t)
            if res:
                result = res

        self.assertEqual(result, "swipe_up")

    def test_swipe_down_with_starting_anchor(self):
        """Simulates pausing at starting position, then an exaggerated downward swipe."""
        self.tracker.enable_vertical_swipes = True
        start_time = 400.0
        result = None
        # 1. Starting anchor pause
        for i in range(4):
            t = start_time + i * 0.025
            self.tracker.update(0.50, 0.25, timestamp=t)
        # 2. Exaggerated swipe from y=0.25 to y=0.75 (disp=0.50)
        for i in range(1, 11):
            x = 0.50
            y = 0.25 + (i / 10) * 0.50
            t = start_time + (3 + i) * 0.025
            res = self.tracker.update(x, y, timestamp=t)
            if res:
                result = res

        self.assertEqual(result, "swipe_down")

    def test_reject_casual_small_motion(self):
        """Verifies that non-exaggerated casual movement (disp=0.12) is rejected."""
        start_time = 450.0
        result = None
        for i in range(4):
            self.tracker.update(0.40, 0.50, timestamp=start_time + i * 0.025)
        for i in range(1, 10):
            res = self.tracker.update(0.40 + (i / 9) * 0.12, 0.50, timestamp=start_time + (3 + i) * 0.025)
            if res:
                result = res
        self.assertIsNone(result)

    def test_reject_flailing_without_starting_anchor(self):
        """Verifies that continuous motion without a starting position pause is rejected."""
        start_time = 480.0
        result = None
        for i in range(15):
            res = self.tracker.update(0.20 + (i / 14) * 0.50, 0.50, timestamp=start_time + i * 0.025)
            if res:
                result = res
        self.assertIsNone(result)

    def test_circle_clockwise(self):
        """Simulates drawing a clockwise circle."""
        self.tracker.enable_circles = True
        start_time = 500.0
        result = None
        n_points = 20
        cx, cy, r = 0.5, 0.5, 0.15

        for i in range(n_points):
            theta = (i / (n_points - 1)) * (2 * np.pi * 1.1)
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            t = start_time + i * 0.035
            res = self.tracker.update(x, y, timestamp=t)
            if res:
                result = res

        self.assertEqual(result, "circle_cw")

    def test_circle_counter_clockwise(self):
        """Simulates drawing a counter-clockwise circle."""
        self.tracker.enable_circles = True
        start_time = 600.0
        result = None
        n_points = 20
        cx, cy, r = 0.5, 0.5, 0.15

        for i in range(n_points):
            theta = -(i / (n_points - 1)) * (2 * np.pi * 1.1)
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            t = start_time + i * 0.035
            res = self.tracker.update(x, y, timestamp=t)
            if res:
                result = res

        self.assertEqual(result, "circle_ccw")

    def test_motion_cooldown(self):
        """Verifies that an immediate second swipe is suppressed during cooldown."""
        start_time = 700.0
        first_swipe = None
        for i in range(4):
            self.tracker.update(0.20, 0.50, timestamp=start_time + i * 0.025)
        for i in range(1, 11):
            x = 0.20 + (i / 10) * 0.50
            t = start_time + (3 + i) * 0.025
            res = self.tracker.update(x, 0.5, timestamp=t)
            if res:
                first_swipe = res

        self.assertEqual(first_swipe, "swipe_right")

        # Immediate motion within 0.1s should return None due to 0.5s cooldown
        res_during_cooldown = self.tracker.update(0.8, 0.5, timestamp=start_time + 0.35)
        self.assertIsNone(res_during_cooldown)

    def test_ema_smoothing(self):
        """Verifies that coordinate smoothing dampens single-frame spikes."""
        tracker = MotionTracker(smoothing_alpha=0.5)
        tracker.update(0.0, 0.0, timestamp=1.0)
        self.assertEqual(tracker.smoothed_pos, (0.0, 0.0))
        # Large jump to 1.0
        tracker.update(1.0, 1.0, timestamp=1.1)
        # Smoothed pos should be 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        self.assertAlmostEqual(tracker.smoothed_pos[0], 0.5, places=3)
        self.assertAlmostEqual(tracker.smoothed_pos[1], 0.5, places=3)

    def test_apply_and_load_profile(self):
        """Verifies applying a personal calibration profile changes tracker thresholds."""
        tracker = MotionTracker()
        custom_prof = {
            "min_displacement": 0.14,
            "min_velocity": 0.22,
            "min_duration": 0.09,
            "max_duration": 0.70,
            "max_anchor_speed": 0.30,
            "smoothing_alpha": 0.60,
        }
        tracker.apply_profile(custom_prof)
        self.assertEqual(tracker.min_displacement, 0.14)
        self.assertEqual(tracker.min_velocity, 0.22)
        self.assertEqual(tracker.smoothing_alpha, 0.60)

    def test_motion_calibrator_workflow(self):
        """Verifies interactive 4-direction calibration flow and profile computation."""
        from src.motion import MotionCalibrator

        calib = MotionCalibrator()
        calib.start()
        self.assertTrue(calib.is_calibrating)
        self.assertEqual(calib.current_direction, "swipe_right")

        # Simulate 3 strokes for horizontal directions (swipe_right, swipe_left)
        for expected_dir in ["swipe_right", "swipe_left"]:
            self.assertEqual(calib.current_direction, expected_dir)
            for _ in range(3):
                calib.last_record_time = 0.0  # bypass debounce in test
                calib.record_stroke(displacement=0.20, velocity=0.40, duration=0.25)

        self.assertTrue(calib.is_finished)
        profile = calib.compute_calibrated_profile()
        self.assertIn("min_displacement", profile)
        self.assertIn("min_velocity", profile)
        self.assertLessEqual(profile["min_displacement"], 0.20)

        # Apply to tracker
        from unittest.mock import patch
        tracker = MotionTracker()
        with patch("src.motion.save_motion_profile"):
            calib.save_and_apply(tracker)
        self.assertEqual(tracker.min_displacement, profile["min_displacement"])

    def test_fast_flick_swipe_right(self):
        """Verifies an ultra-fast 3-frame flick (50ms) from start anchor is detected."""
        start_time = 800.0
        result = None
        # 1. Starting anchor pause for 3 frames
        for i in range(3):
            self.tracker.update(0.25, 0.50, timestamp=start_time + i * 0.02)

        # 2. Fast flick across 3 frames (50ms total, displacement = 0.35, high velocity)
        # Frame 1: fast move
        self.tracker.update(0.42, 0.50, timestamp=start_time + 0.06)
        # Frame 2: reach landing point
        res2 = self.tracker.update(0.60, 0.50, timestamp=start_time + 0.08)
        if res2:
            result = res2
        # Frame 3: settle at landing point
        res3 = self.tracker.update(0.60, 0.50, timestamp=start_time + 0.10)
        if res3:
            result = res3

        self.assertEqual(result, "swipe_right")

    def test_fast_flick_swipe_up_gated(self):
        """Verifies vertical upward flick is suppressed by default, and only active when explicitly enabled."""
        start_time = 900.0
        # By default, enable_vertical_swipes is False -> upward motion must return None (no false positive)
        self.tracker.enable_vertical_swipes = False
        result = None
        for i in range(3):
            self.tracker.update(0.50, 0.70, timestamp=start_time + i * 0.02)

        self.tracker.update(0.50, 0.52, timestamp=start_time + 0.06)
        res2 = self.tracker.update(0.50, 0.35, timestamp=start_time + 0.08)
        if res2:
            result = res2
        res3 = self.tracker.update(0.50, 0.35, timestamp=start_time + 0.10)
        if res3:
            result = res3

        self.assertIsNone(result)

        # When enable_vertical_swipes is True, it triggers swipe_up
        self.tracker.reset()
        self.tracker.enable_vertical_swipes = True
        start_time = 1000.0
        result_enabled = None
        for i in range(3):
            self.tracker.update(0.50, 0.70, timestamp=start_time + i * 0.02)
        self.tracker.update(0.50, 0.52, timestamp=start_time + 0.06)
        r2 = self.tracker.update(0.50, 0.35, timestamp=start_time + 0.08)
        if r2:
            result_enabled = r2
        r3 = self.tracker.update(0.50, 0.35, timestamp=start_time + 0.10)
        if r3:
            result_enabled = r3
        self.assertEqual(result_enabled, "swipe_up")

    def test_slow_movement_rejected_no_false_positives(self):
        """Verifies that slow, casual hand movements do NOT trigger swipes (velocity check)."""
        start_time = 1200.0
        # 1. Anchor
        for i in range(4):
            self.tracker.update(0.30, 0.50, timestamp=start_time + i * 0.03)

        # 2. Slow casual drift over 500ms (dx = 0.08, dt = 0.5s -> velocity = 0.16 < min_velocity = 0.35)
        detected = None
        for i in range(1, 11):
            t = start_time + 0.12 + (i * 0.05)
            x = 0.30 + (i / 10.0) * 0.08
            res = self.tracker.update(x, 0.50, timestamp=t)
            if res:
                detected = res

        self.assertIsNone(detected, "Slow movement triggered a false positive!")

    def test_calibration_reset_gate(self):
        """Verifies that calibration blocks rapid strokes until the user resets hand and pauses."""
        from src.motion import MotionCalibrator

        calib = MotionCalibrator()
        calib.start()

        # Stroke 1: recorded
        stroke1 = calib.check_and_record_stroke(
            [(0.25, 0.50, 10.0), (0.60, 0.50, 10.2)],
            is_anchored=False,
            active_anchor=(0.25, 0.50),
        )
        self.assertIsNotNone(stroke1)
        self.assertTrue(calib.waiting_for_reset)
        self.assertEqual(calib.current_count, 1)

        # Attempt Stroke 2 immediately while resetting -> Must be BLOCKED
        blocked = calib.check_and_record_stroke(
            [(0.60, 0.50, 10.3), (0.25, 0.50, 10.4)],
            is_anchored=False,
            active_anchor=None,
        )
        self.assertIsNone(blocked)
        self.assertEqual(calib.current_count, 1)

        # Simulate user returning hand and pausing still (is_anchored=True) after reset time
        calib.reset_start_time = 0.0  # simulate passage of reset cooldown
        calib.update_state(is_anchored=True, timestamp=10.0)
        calib.update_state(is_anchored=True, timestamp=10.4)  # > steady_hold_time (0.35s)
        self.assertFalse(calib.waiting_for_reset)

        # Now Stroke 2 can be recorded successfully!
        stroke2 = calib.check_and_record_stroke(
            [(0.25, 0.50, 11.0), (0.60, 0.50, 11.2)],
            is_anchored=False,
            active_anchor=(0.25, 0.50),
        )
        self.assertIsNotNone(stroke2)
        self.assertEqual(calib.current_count, 2)

    def test_micro_motions_rejected_as_false_positives(self):
        """Verifies small twitches/drifts (disp=0.04-0.07) are strictly rejected as false positives."""
        tracker = MotionTracker()
        tracker.load_calibrated_profile()
        start_time = 1500.0

        # Anchor
        for i in range(4):
            tracker.update(0.50, 0.50, timestamp=start_time + i * 0.03)

        # Micro-twitches at varying speeds and small pauses
        for dx in [0.04, 0.055, 0.07]:
            tracker.reset()
            t = start_time + 10.0
            for i in range(4):
                tracker.update(0.50, 0.50, timestamp=t + i * 0.03)
            r1 = tracker.update(0.50 + dx, 0.50, timestamp=t + 0.15)
            r2 = tracker.update(0.50 + dx, 0.50, timestamp=t + 0.18)
            r3 = tracker.update(0.50, 0.50, timestamp=t + 0.21)
            self.assertIsNone(r1, f"Micro-twitch dx={dx} triggered false positive on r1!")
            self.assertIsNone(r2, f"Micro-twitch dx={dx} triggered false positive on r2!")
            self.assertIsNone(r3, f"Micro-twitch dx={dx} triggered false positive on r3!")

    def test_large_natural_movement_recognized(self):
        """Verifies large movements across screen with natural vertical rotation/arc trigger reliably."""
        tracker = MotionTracker()
        tracker.load_calibrated_profile()
        start_time = 1800.0

        # Anchor
        for i in range(4):
            tracker.update(0.35, 0.45, timestamp=start_time + i * 0.03)

        # Large rightward movement with natural vertical curvature (dx=0.28, dy=0.21 -> ratio=1.33)
        detected = None
        t = start_time + 0.12
        for step in range(1, 10):
            prog = step / 9.0
            x = 0.35 + 0.28 * prog
            y = 0.45 + 0.18 * np.sin(prog * np.pi)
            t += 0.035
            res = tracker.update(x, y, timestamp=t)
            if res:
                detected = res
                break
        else:
            # Settle at end
            t += 0.040
            detected = tracker.update(0.63, 0.45, timestamp=t)

        self.assertEqual(detected, "swipe_right", "Large natural movement failed to trigger swipe_right!")

    def test_kinetic_rig_coasts_through_motion_blur(self):
        """Verifies that when MediaPipe drops hand during rapid motion blur, KineticRig coasts and triggers."""
        tracker = MotionTracker()
        tracker.load_calibrated_profile()
        start_time = 2000.0

        # Anchor
        for i in range(4):
            tracker.update(0.35, 0.50, timestamp=start_time + i * 0.03)

        # Frame 1: Rapid movement starts
        tracker.update(0.44, 0.50, timestamp=start_time + 0.12)

        # Frame 2: Severe motion blur -> MediaPipe drops hand!
        res_dropped = tracker.handle_tracking_loss(timestamp=start_time + 0.15)

        # Frame 3: Re-acquired at landing
        res_landing = tracker.update(0.60, 0.50, timestamp=start_time + 0.18)

        # Frame 4: Settle
        res_settle = tracker.update(0.60, 0.50, timestamp=start_time + 0.21)

        flick_detected = res_dropped or res_landing or res_settle
        self.assertEqual(flick_detected, "swipe_right", "Motion blur coasting failed to trigger flick!")

    def test_face_zone_resting_suppressed(self):
        """Verifies that resting hand on chin or touching face does not anchor or trigger swipes."""
        tracker = MotionTracker()
        tracker.load_calibrated_profile()
        face_center = (0.50, 0.30)
        start_time = 3000.0

        # Hand rests on chin right below face_center
        for i in range(5):
            res = tracker.update(0.50, 0.35, timestamp=start_time + i * 0.03, face_center=face_center)
            self.assertIsNone(res)

        # Ensure anchor was suppressed inside face zone
        self.assertFalse(tracker.is_anchored)
        self.assertIsNone(tracker.active_start_pos)

        # Small twitch on chin
        res_twitch = tracker.update(0.53, 0.35, timestamp=start_time + 0.20, face_center=face_center)
        self.assertIsNone(res_twitch)

    def test_face_occlusion_teleport_rejection(self):
        """Verifies that a sudden MediaPipe teleport jump to face or shoulder is gated out."""
        tracker = MotionTracker()
        tracker.load_calibrated_profile()
        face_center = (0.50, 0.25)
        start_time = 4000.0

        # Hand steady in free space
        for i in range(4):
            tracker.update(0.75, 0.65, timestamp=start_time + i * 0.03, face_center=face_center)

        # Single frame occlusion teleport: landmark jumps 0.45 units straight to the face!
        res_teleport = tracker.update(0.50, 0.25, timestamp=start_time + 0.15, face_center=face_center)
        self.assertIsNone(res_teleport, "Occlusion teleport should not trigger a false gesture!")

        # Next frame: hand returns to where it actually was
        res_return = tracker.update(0.75, 0.65, timestamp=start_time + 0.18, face_center=face_center)
        self.assertIsNone(res_return, "Returning from occlusion should not trigger a false gesture!")

    def test_pull_away_from_face_suppressed(self):
        """Verifies that moving hand away from touching one's face does not fire a swipe."""
        tracker = MotionTracker()
        tracker.load_calibrated_profile()
        face_center = (0.50, 0.30)
        start_time = 5000.0

        # Hand on face
        for i in range(4):
            tracker.update(0.51, 0.32, timestamp=start_time + i * 0.03, face_center=face_center)

        # Pull hand rapidly away to the right
        res1 = tracker.update(0.65, 0.45, timestamp=start_time + 0.15, face_center=face_center)
        res2 = tracker.update(0.80, 0.55, timestamp=start_time + 0.18, face_center=face_center)
        self.assertIsNone(res1)
        self.assertIsNone(res2, "Pulling hand away from face should not trigger a swipe!")

    def test_return_to_center_suppressed(self):
        """Verifies that returning hand to center after a rightward swipe is NEVER detected as swipe_left."""
        tracker = MotionTracker()
        tracker.load_calibrated_profile()
        start_time = 6000.0

        # 1. Starting anchor pause at center (0.45, 0.50)
        for i in range(4):
            tracker.update(0.45, 0.50, timestamp=start_time + i * 0.03)

        # 2. Crisp flick to the right (to 0.70)
        tracker.update(0.58, 0.50, timestamp=start_time + 0.15)
        res_right = tracker.update(0.70, 0.50, timestamp=start_time + 0.18)
        self.assertEqual(res_right, "swipe_right")

        # 3. Hand moves back towards center (0.70 -> 0.58 -> 0.45) during return stroke
        res_ret1 = tracker.update(0.62, 0.50, timestamp=start_time + 0.30)
        res_ret2 = tracker.update(0.52, 0.50, timestamp=start_time + 0.45)
        res_ret3 = tracker.update(0.45, 0.50, timestamp=start_time + 0.60)
        # Even after cooldown (0.18 + 0.80 = 0.98s):
        res_ret4 = tracker.update(0.45, 0.50, timestamp=start_time + 1.02)

        self.assertIsNone(res_ret1, "Return to center should not trigger any gesture")
        self.assertIsNone(res_ret2, "Return to center should not trigger any gesture")
        self.assertIsNone(res_ret3, "Return to center should not trigger any gesture")
        self.assertIsNone(res_ret4, "Settling at center should not trigger any gesture")

        # 4. Hand settles steady at center for 8 frames (~240ms)
        for i in range(8):
            tracker.update(0.45, 0.50, timestamp=start_time + 1.05 + i * 0.03)

        # 5. Now an intentional leftward flick (0.45 -> 0.20) should trigger swipe_left!
        tracker.update(0.32, 0.50, timestamp=start_time + 1.35)
        res_left1 = tracker.update(0.20, 0.50, timestamp=start_time + 1.38)
        res_left2 = tracker.update(0.20, 0.50, timestamp=start_time + 1.41)
        res_left = res_left1 or res_left2
        self.assertEqual(res_left, "swipe_left", "Intentional swipe_left after settling at center should trigger!")

    def test_consecutive_same_flicks(self):
        """Verifies that flicking right, returning to center, and flicking right again works cleanly."""
        tracker = MotionTracker()
        tracker.load_calibrated_profile()
        start_time = 7000.0

        # Anchor at center
        for i in range(4):
            tracker.update(0.45, 0.50, timestamp=start_time + i * 0.03)

        # First flick right
        tracker.update(0.58, 0.50, timestamp=start_time + 0.15)
        res1 = tracker.update(0.70, 0.50, timestamp=start_time + 0.18)
        self.assertEqual(res1, "swipe_right")

        # Return to center and settle past cooldown
        for i in range(6):
            tracker.update(0.45, 0.50, timestamp=start_time + 0.90 + i * 0.03)

        # Second flick right
        tracker.update(0.58, 0.50, timestamp=start_time + 1.20)
        res2 = tracker.update(0.70, 0.50, timestamp=start_time + 1.23)
        self.assertEqual(res2, "swipe_right", "Second deliberate flick after returning to center should trigger!")


if __name__ == "__main__":
    unittest.main()


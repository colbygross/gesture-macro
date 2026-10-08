"""
Unit tests for Dynamic Motion Learner (Resampling, Feature Extraction, Dataset Storage,
Random Forest Model Training, Inference, and Tracking Persistence Coasting).
"""

import sys
import unittest
import tempfile
import math
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.motion_learner import (
    resample_trajectory,
    extract_trajectory_features,
    get_feature_names,
    save_motion_samples,
    load_motion_dataset,
    train_motion_model,
    MotionModel,
    MotionGestureRecorder,
)
from src.motion import MotionTracker


class TestMotionLearner(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "test_motion_gestures.csv"
        self.model_path = Path(self.temp_dir.name) / "test_motion_model.joblib"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _generate_synthetic_stroke(self, gesture_type: str, num_frames: int = 15) -> list:
        """Generates realistic synthetic motion trajectory points."""
        stroke = []
        t0 = 100.0
        dt = 0.025

        if gesture_type == "swipe_right":
            for i in range(num_frames):
                x = 0.20 + (i / num_frames) * 0.40
                y = 0.50 + 0.01 * math.sin(i)
                stroke.append((x, y, t0 + i * dt))
        elif gesture_type == "swipe_left":
            for i in range(num_frames):
                x = 0.70 - (i / num_frames) * 0.40
                y = 0.50 + 0.01 * math.sin(i)
                stroke.append((x, y, t0 + i * dt))
        elif gesture_type == "circle_cw":
            radius = 0.15
            cx, cy = 0.50, 0.50
            for i in range(num_frames):
                angle = (i / num_frames) * 2 * math.pi
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                stroke.append((x, y, t0 + i * dt))
        elif gesture_type == "v_shape":
            half = num_frames // 2
            for i in range(half):
                x = 0.30 + (i / half) * 0.15
                y = 0.30 + (i / half) * 0.30
                stroke.append((x, y, t0 + i * dt))
            for i in range(num_frames - half):
                x = 0.45 + (i / (num_frames - half)) * 0.15
                y = 0.60 - (i / (num_frames - half)) * 0.30
                stroke.append((x, y, t0 + (half + i) * dt))
        return stroke

    def test_resample_trajectory(self):
        """Verifies spatial arc-length resampling produces normalized equidistant points."""
        stroke = self._generate_synthetic_stroke("swipe_right", num_frames=12)
        resampled = resample_trajectory(stroke, num_points=24)

        self.assertEqual(resampled.shape, (24, 2))
        # First point must be relative origin (0, 0)
        self.assertAlmostEqual(resampled[0, 0], 0.0, places=5)
        self.assertAlmostEqual(resampled[0, 1], 0.0, places=5)
        # Final point should reflect net displacement
        self.assertGreater(resampled[-1, 0], 0.30)

    def test_extract_features(self):
        """Verifies 58-dimensional kinematic and geometric feature extraction."""
        stroke_right = self._generate_synthetic_stroke("swipe_right", num_frames=15)
        feats_right = extract_trajectory_features(stroke_right, num_points=24)
        self.assertEqual(len(feats_right), 58)

        # Straight swipe should have straightness ratio (D/L) close to 1.0
        straightness = feats_right[4]
        self.assertGreater(straightness, 0.90)

        # Circular gesture should have very low straightness ratio
        stroke_circle = self._generate_synthetic_stroke("circle_cw", num_frames=20)
        feats_circle = extract_trajectory_features(stroke_circle, num_points=24)
        self.assertLess(feats_circle[4], 0.30)

    def test_feature_names(self):
        """Verifies feature name generation matches 58 extracted features."""
        names = get_feature_names(num_points=24)
        self.assertEqual(len(names), 58)
        self.assertEqual(names[0], "net_dx")
        self.assertEqual(names[-1], "p23_dy")

    def test_save_and_load_dataset(self):
        """Tests appending trajectory strokes to CSV and reloading."""
        strokes_r = [self._generate_synthetic_stroke("swipe_right") for _ in range(5)]
        strokes_l = [self._generate_synthetic_stroke("swipe_left") for _ in range(5)]

        saved_r = save_motion_samples("swipe_right", strokes_r, csv_path=self.csv_path)
        saved_l = save_motion_samples("swipe_left", strokes_l, csv_path=self.csv_path)

        self.assertEqual(saved_r, 5)
        self.assertEqual(saved_l, 5)

        X, y, classes = load_motion_dataset(csv_path=self.csv_path)
        self.assertEqual(X.shape, (10, 58))
        self.assertEqual(len(y), 10)
        self.assertEqual(classes, ["swipe_left", "swipe_right"])

    def test_train_and_predict_motion_model(self):
        """Trains a Random Forest classifier on motion samples and verifies live inference."""
        classes = ["swipe_right", "swipe_left", "v_shape"]
        for cls in classes:
            samples = []
            for _ in range(12):
                stroke = self._generate_synthetic_stroke(cls)
                # Add slight noise
                noisy = [(p[0] + np.random.normal(0, 0.01), p[1] + np.random.normal(0, 0.01), p[2]) for p in stroke]
                samples.append(noisy)
            save_motion_samples(cls, samples, csv_path=self.csv_path)

        # Train model
        metrics = train_motion_model(csv_path=self.csv_path, model_path=self.model_path)
        self.assertGreaterEqual(metrics["accuracy"], 0.85)
        self.assertEqual(metrics["classes"], sorted(classes))

        # Test inference wrapper
        model = MotionModel(model_path=self.model_path)
        self.assertTrue(model.is_loaded)

        test_stroke = self._generate_synthetic_stroke("swipe_right")
        pred_label, conf = model.predict(test_stroke)
        self.assertEqual(pred_label, "swipe_right")
        self.assertGreater(conf, 0.60)

    def test_motion_gesture_recorder_with_hand_reset_gate(self):
        """Verifies the Hand-Reset Gate ensures clean, discrete stroke recording."""
        recorder = MotionGestureRecorder(min_reset_time=0.30, min_stroke_disp=0.15)
        recorder.start_recording("wave", target_count=2)

        t = 10.0
        # 1. Hand at anchor (steady)
        recorder.feed_point(0.50, 0.50, t, is_anchored=True)
        self.assertFalse(recorder.waiting_for_reset)

        # 2. Hand moves and reaches peak displacement
        recorder.feed_point(0.55, 0.50, t + 0.03, is_anchored=False)
        recorder.feed_point(0.68, 0.50, t + 0.06, is_anchored=False)
        # Pullback triggers stroke completion
        completed = recorder.feed_point(0.62, 0.50, t + 0.09, is_anchored=False)
        self.assertIsNotNone(completed)
        self.assertEqual(recorder.current_count, 1)
        self.assertTrue(recorder.waiting_for_reset)

        # 3. During reset wait, feed_point returns None
        self.assertIsNone(recorder.feed_point(0.55, 0.50, t + 0.12, is_anchored=False))

        # 4. Hand pauses at anchor, but before min_reset_time elapsed
        recorder.update_state(is_anchored=True, timestamp=t + 0.15)
        self.assertTrue(recorder.waiting_for_reset)

        # 5. Hand pauses at anchor after min_reset_time (0.30s passed)
        recorder.update_state(is_anchored=True, timestamp=t + 0.45)
        self.assertTrue(recorder.waiting_for_reset)  # Still waiting for steady hold (0.35s)
        recorder.update_state(is_anchored=True, timestamp=t + 0.85)  # Steady hold satisfied
        self.assertFalse(recorder.waiting_for_reset)

    def test_tracking_coasting_persistence(self):
        """
        Verifies that high-speed frame dropouts do NOT instantly wipe tracker buffers,
        and that strokes landing during tracking loss are successfully detected.
        """
        tracker = MotionTracker(
            buffer_size=36,
            min_displacement=0.20,
            require_starting_anchor=True,
            cooldown_seconds=0.3,
        )

        start_time = 50.0
        # 1. Establish rest anchor
        for i in range(4):
            tracker.update(0.20, 0.50, timestamp=start_time + i * 0.025)
        self.assertTrue(tracker.is_anchored)

        # 2. Fast sweep right in mid-flight (displacement = 0.14 < min_displacement = 0.20)
        for i in range(1, 3):
            tracker.update(0.20 + (i / 2) * 0.14, 0.50, timestamp=start_time + (3 + i) * 0.025)

        # 3. Simulate MediaPipe frame dropout due to high speed motion blur
        # Hand was lost for 1 frame
        res1 = tracker.handle_tracking_loss(timestamp=start_time + 0.18)
        # Tracker should have maintained active start position during coasting
        self.assertIsNotNone(tracker.active_start_pos)
        self.assertEqual(tracker.missed_frames, 1)

        # Complete stroke landing
        tracker.update(0.55, 0.50, timestamp=start_time + 0.22)
        res2 = tracker.handle_tracking_loss(timestamp=start_time + 0.26)
        self.assertIn(tracker.last_triggered_gesture or res2, ["swipe_right"])

        # 4. Verify true loss of tracking (> 6 frames) cleanly resets
        tracker.reset()
        for i in range(4):
            tracker.update(0.50, 0.50, timestamp=100.0 + i * 0.025)
        for _ in range(7):
            tracker.handle_tracking_loss(timestamp=100.5)
        # Trajectory buffer should now be cleared
        self.assertEqual(len(tracker.trajectory), 0)
        self.assertFalse(tracker.is_anchored)


if __name__ == "__main__":
    unittest.main()

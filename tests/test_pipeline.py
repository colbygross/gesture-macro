"""
Unit and integration tests for Gesture Macro Controller.
Tests landmark normalization invariance, model training, and key translation.
"""

import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detector import HandDetector
from src.actions import ActionDispatcher
from src.train import train_gesture_classifier


class TestGesturePipeline(unittest.TestCase):

    def test_landmark_normalization_invariance(self):
        """
        Verify that landmark normalization produces identical results regardless of
        translation (position on screen) or uniform scaling (distance from webcam).
        """
        np.random.seed(42)
        base_landmarks = np.random.uniform(low=0.2, high=0.8, size=(21, 3))
        # Ensure wrist (0) and middle MCP (9) are distinct
        base_landmarks[0] = [0.5, 0.8, 0.0]
        base_landmarks[9] = [0.5, 0.4, 0.0]

        normalized_base = HandDetector.normalize_landmarks(base_landmarks)

        # 1. Test translation invariance: shift all points by (+15.0, -8.0, +3.0)
        shifted_landmarks = base_landmarks + np.array([15.0, -8.0, 3.0])
        normalized_shifted = HandDetector.normalize_landmarks(shifted_landmarks)

        np.testing.assert_allclose(
            normalized_base,
            normalized_shifted,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Normalization failed translation invariance test!"
        )

        # 2. Test scale invariance: uniformly scale all points by 4.5x
        # To scale relative to wrist:
        scaled_relative = (base_landmarks - base_landmarks[0]) * 4.5
        scaled_landmarks = base_landmarks[0] + scaled_relative
        normalized_scaled = HandDetector.normalize_landmarks(scaled_landmarks)

        np.testing.assert_allclose(
            normalized_base,
            normalized_scaled,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Normalization failed scale invariance test!"
        )

    def test_action_dispatcher_keys_and_cooldown(self):
        """Test key translation and cooldown logic."""
        dispatcher = ActionDispatcher(cooldown_seconds=0.5, dry_run=True)

        # Platform key translations
        if dispatcher.is_mac:
            self.assertEqual(dispatcher._translate_key("cmd"), "command")
            self.assertEqual(dispatcher._translate_key("win"), "command")
        else:
            self.assertEqual(dispatcher._translate_key("cmd"), "ctrl")
            self.assertEqual(dispatcher._translate_key("win"), "win")

        # Test cooldown logic
        action_cfg = {"description": "Test Action", "type": "hotkey", "keys": ["ctrl", "c"]}
        first_fire = dispatcher.execute("test_gesture", action_cfg)
        self.assertTrue(first_fire)

        # Immediate second trigger should be blocked by cooldown
        second_fire = dispatcher.execute("test_gesture", action_cfg)
        self.assertFalse(second_fire)

        # Test scroll action execution (dry_run)
        scroll_cfg = {"description": "Scroll Up", "type": "scroll", "amount": 300}
        scroll_fire = dispatcher.execute("swipe_up", scroll_cfg)
        self.assertTrue(scroll_fire)

        # Test alt-tab hotkey execution (dry_run)
        alt_tab_cfg = {"description": "Switch App", "type": "hotkey", "keys": ["alt", "tab"]}
        alt_tab_fire = dispatcher.execute("swipe_right", alt_tab_cfg)
        self.assertTrue(alt_tab_fire)

    def test_end_to_end_training_and_inference(self):
        """
        Synthesizes a minimal dataset, trains a model, and verifies serialization and inference.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            csv_path = tmp_path / "test_gestures.csv"
            model_path = tmp_path / "test_model.joblib"

            # Create synthetic dataset with 2 distinct clusters
            np.random.seed(123)
            n_samples = 40
            fist_features = np.random.normal(loc=0.0, scale=0.1, size=(n_samples, 63))
            palm_features = np.random.normal(loc=1.0, scale=0.1, size=(n_samples, 63))

            data = []
            for row in fist_features:
                data.append(["fist"] + list(row))
            for row in palm_features:
                data.append(["open_palm"] + list(row))

            df = pd.DataFrame(data, columns=["label"] + [f"f{i}" for i in range(63)])
            df.to_csv(csv_path, index=False)

            # Train classifier
            train_gesture_classifier(
                csv_path=csv_path,
                output_model_path=model_path,
                model_type="rf",
                test_size=0.25,
            )

            self.assertTrue(model_path.exists())

            # Load model and verify prediction
            payload = joblib.load(model_path)
            model = payload["model"]
            classes = payload["classes"]

            self.assertIn("fist", classes)
            self.assertIn("open_palm", classes)

            # Test inference on test vector
            test_fist = np.random.normal(loc=0.0, scale=0.05, size=(1, 63))
            pred = model.predict(test_fist)[0]
            self.assertEqual(pred, "fist")

    def test_fingertip_action_center(self):
        """Verifies extended fingertip identification and centroid calculation."""
        # 1. Empty landmarks fallback
        norm, px = HandDetector.compute_fingertip_center([], [])
        self.assertEqual(norm, (0.5, 0.5))

        # 2. Pointing gesture (index extended, others folded)
        raw = [[0.5, 0.8, 0.0]] * 21
        raw[0] = [0.5, 0.8, 0.0]   # Wrist
        raw[6] = [0.5, 0.5, 0.0]   # Index PIP
        raw[8] = [0.5, 0.2, 0.0]   # Index Tip (extended)
        for i in [10, 12, 14, 16, 18, 20]:
            raw[i] = [0.5, 0.75, 0.0]  # Folded
        px = [(int(x * 640), int(y * 480)) for x, y, _ in raw]

        norm_res, px_res = HandDetector.compute_fingertip_center(raw, px)
        self.assertAlmostEqual(norm_res[0], 0.5, places=3)
        self.assertAlmostEqual(norm_res[1], 0.2, places=3)
        self.assertEqual(px_res, (320, 96))

        # 3. Open hand (all 4 fingers extended)
        raw_open = [[0.5, 0.8, 0.0]] * 21
        raw_open[0] = [0.5, 0.8, 0.0]   # Wrist
        # Tips at y=0.2, PIPs at y=0.5
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            raw_open[pip] = [0.5, 0.5, 0.0]
            raw_open[tip] = [0.5, 0.2, 0.0]
        raw_open[8] = [0.35, 0.2, 0.0]   # Index tip
        raw_open[12] = [0.45, 0.18, 0.0]  # Middle tip
        raw_open[16] = [0.55, 0.2, 0.0]   # Ring tip
        raw_open[20] = [0.65, 0.25, 0.0]  # Pinky tip
        px_open = [(int(x * 640), int(y * 480)) for x, y, _ in raw_open]

        norm_open, _ = HandDetector.compute_fingertip_center(raw_open, px_open)
        # Average x: (0.35 + 0.45 + 0.55 + 0.65) / 4 = 0.50
        self.assertAlmostEqual(norm_open[0], 0.50, places=2)
        # Average y: (0.2 + 0.18 + 0.2 + 0.25) / 4 = 0.2075
        self.assertAlmostEqual(norm_open[1], 0.2075, places=2)


if __name__ == "__main__":
    unittest.main()

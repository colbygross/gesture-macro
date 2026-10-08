"""
Unit and integration tests for Upper-Body Rig, Chest Normalization, and Gesture Studio components.
"""

import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detector import UpperBodyRigDetector
from src.train import train_gesture_classifier
from src.config import load_macro_actions, save_macro_actions


class TestUpperBodyRig(unittest.TestCase):

    def test_upper_body_normalization_invariance(self):
        """
        Verify that chest-centered, shoulder-width normalization is invariant
        to translation (user moving across webcam frame) and scaling (user moving closer/further).
        """
        np.random.seed(42)
        base_body = np.random.uniform(low=0.2, high=0.8, size=(15, 3))
        # Left shoulder is index 1, Right shoulder is index 2
        base_body[1] = [0.4, 0.4, 0.0]
        base_body[2] = [0.6, 0.4, 0.0]

        norm_base = UpperBodyRigDetector.normalize_upper_body(base_body)

        # 1. Translation test: shift by (100, -50, 20)
        shifted_body = base_body + np.array([100.0, -50.0, 20.0])
        norm_shifted = UpperBodyRigDetector.normalize_upper_body(shifted_body)

        np.testing.assert_allclose(
            norm_base,
            norm_shifted,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Upper body normalization failed translation invariance!",
        )

        # 2. Scaling test: scale around chest midpoint by 3.5x
        chest_center = (base_body[1] + base_body[2]) / 2.0
        scaled_body = chest_center + (base_body - chest_center) * 3.5
        norm_scaled = UpperBodyRigDetector.normalize_upper_body(scaled_body)

        np.testing.assert_allclose(
            norm_base,
            norm_scaled,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Upper body normalization failed scale invariance!",
        )

    def test_108_feature_training_and_inference(self):
        """
        Verifies that train_gesture_classifier works smoothly with 108-dimensional
        upper body rig feature vectors.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            csv_path = tmp_path / "test_upper_body_gestures.csv"
            model_path = tmp_path / "test_upper_body_model.joblib"

            np.random.seed(99)
            n_samples = 30
            # 108 features: 45 body + 63 hands
            crossed_arms = np.random.normal(loc=-1.0, scale=0.1, size=(n_samples, 108))
            hand_to_chin = np.random.normal(loc=1.0, scale=0.1, size=(n_samples, 108))

            rows = []
            for r in crossed_arms:
                rows.append(["arms_crossed"] + list(r))
            for r in hand_to_chin:
                rows.append(["hand_to_chin"] + list(r))

            df = pd.DataFrame(rows, columns=["label"] + [f"f{i}" for i in range(108)])
            df.to_csv(csv_path, index=False)

            result = train_gesture_classifier(
                csv_path=csv_path,
                output_model_path=model_path,
                model_type="rf",
                test_size=0.25,
            )

            self.assertTrue(result["success"])
            self.assertEqual(result["feature_count"], 108)
            self.assertGreaterEqual(result["accuracy"], 0.95)

            # Check model serialization
            self.assertTrue(model_path.exists())
            payload = joblib.load(model_path)
            self.assertEqual(payload["feature_count"], 108)

            # Test inference
            test_vec = np.random.normal(loc=-1.0, scale=0.05, size=(1, 108))
            pred = payload["model"].predict(test_vec)[0]
            self.assertEqual(pred, "arms_crossed")

    def test_macro_json_loading_and_saving(self):
        """Verifies macro actions can be serialized to and from JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_json = Path(tmpdir) / "test_macros.json"
            sample_macros = {
                "arms_crossed": {"description": "Show Desktop", "type": "hotkey", "keys": ["win", "d"]},
                "hand_to_chin": {"description": "Mute Mic", "type": "hotkey", "keys": ["ctrl", "shift", "m"]},
            }

            import src.config
            old_path = src.config.MACROS_JSON_PATH
            try:
                src.config.MACROS_JSON_PATH = tmp_json
                save_macro_actions(sample_macros)
                self.assertTrue(tmp_json.exists())

                loaded = load_macro_actions()
                self.assertIn("arms_crossed", loaded)
                self.assertEqual(loaded["arms_crossed"]["keys"], ["win", "d"])
            finally:
                src.config.MACROS_JSON_PATH = old_path


if __name__ == "__main__":
    unittest.main()

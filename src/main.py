"""
Main real-time execution loop for the Gesture Macro Controller.
Captures webcam frames, recognizes hand gestures using the trained model,
displays an interactive HUD, debounces input, and executes macros.
Cross-platform support for Windows and macOS.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import cv2
import numpy as np
import joblib

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.camera import Camera
from src.detector import UpperBodyRigDetector
from src.actions import ActionDispatcher
from src.motion import MotionTracker
from src.motion_learner import MotionModel
from src.config import (
    MODEL_PATH,
    MOTION_MODEL_PATH,
    DEFAULT_GESTURE_ACTIONS,
    DEFAULT_MACRO_ACTIONS,
    load_macro_actions,
    DEBOUNCE_CONSECUTIVE_FRAMES,
    ACTION_COOLDOWN_SECONDS,
    PREDICTION_CONFIDENCE_THRESHOLD,
)


class GestureMacroApp:
    """
    Main controller tying together camera feed, hand detector, ML model,
    debouncing state machine, and action dispatcher.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        actions: Optional[Dict[str, Any]] = None,
        debounce_frames: int = DEBOUNCE_CONSECUTIVE_FRAMES,
        cooldown_seconds: float = ACTION_COOLDOWN_SECONDS,
        confidence_threshold: float = PREDICTION_CONFIDENCE_THRESHOLD,
        dry_run: bool = False,
        camera_index: int = 0,
    ):
        self.model_path = model_path
        self.actions = actions or load_macro_actions()
        self.debounce_frames = debounce_frames
        self.cooldown_seconds = cooldown_seconds
        self.confidence_threshold = confidence_threshold
        self.dry_run = dry_run

        self.model_payload = self._load_model()
        self.model = self.model_payload["model"]
        self.classes = self.model_payload["classes"]

        self.camera = Camera(camera_index=camera_index)
        self.detector = UpperBodyRigDetector()
        self.motion_tracker = MotionTracker()
        self.motion_tracker.load_calibrated_profile()
        self.motion_model = MotionModel(model_path=MOTION_MODEL_PATH)
        if self.motion_model.is_loaded:
            self.motion_tracker.set_motion_model(self.motion_model)
            print(f"[MODEL] Loaded dynamic motion model: {self.motion_model.classes}")
        self.dispatcher = ActionDispatcher(cooldown_seconds=self.cooldown_seconds, dry_run=self.dry_run)

        # State tracking for debounce
        self.last_candidate_gesture: Optional[str] = None
        self.consecutive_frame_count = 0
        self.last_triggered_banner = ""
        self.last_triggered_time = 0.0

    def _load_model(self) -> Dict[str, Any]:
        """Loads serialized model and metadata."""
        if not self.model_path.exists():
            print("\n" + "=" * 60)
            print(f"[ERROR] Trained model not found at: {self.model_path}")
            print("To create a model:")
            print(" 1. Collect gestures: python -m src.collect_data --gesture <name>")
            print(" 2. Train model:      python -m src.train")
            print("=" * 60 + "\n")
            sys.exit(1)

        print(f"[LOAD] Loading gesture model from: {self.model_path}")
        payload = joblib.load(self.model_path)
        print(f"[MODEL] Recognized gestures: {payload['classes']}")
        return payload

    def run(self):
        """Main real-time processing loop."""
        if not self.camera.open():
            sys.exit(1)

        print("\n" + "=" * 60)
        print("  GESTURE MACRO CONTROLLER RUNNING")
        print(f"  Hold any gesture for {self.debounce_frames} frames to trigger.")
        print(f"  Dry run mode: {self.dry_run}")
        print("  Press [Q] or [ESC] to exit.")
        print("=" * 60 + "\n")

        window_name = "Gesture Macro Controller (Hold pose to trigger | Press Q to quit)"
        prev_time = time.time()

        try:
            while True:
                ret, frame = self.camera.read_frame()
                if not ret:
                    print("[ERROR] Failed to read webcam frame.")
                    break

                # FPS calculation
                now = time.time()
                fps = int(1.0 / max(now - prev_time, 1e-5))
                prev_time = now

                # Detect landmarks and normalize
                features, debug_info = self.detector.find_rig_landmarks(frame, draw=True)
                h, w, _ = frame.shape

                # Dynamic Motion Tracking (Fingertip Action Center for wrist flicks)
                detected_motion = None
                tracking_pt = debug_info.get("tracking_point") or debug_info.get("palm_center")
                face_center = debug_info.get("face_center")
                if tracking_pt:
                    norm_x, norm_y = tracking_pt
                    detected_motion = self.motion_tracker.update(
                        norm_x, norm_y, timestamp=now, frame_bgr=frame, face_center=face_center
                    )
                else:
                    detected_motion = self.motion_tracker.handle_tracking_loss(
                        timestamp=now, frame_bgr=frame, face_center=face_center
                    )

                # Render glowing comet trail on frame
                self.motion_tracker.draw_trail(frame)

                predicted_gesture = None
                confidence = 0.0
                triggered_now = False

                # Handle dynamic motion triggers immediately
                if detected_motion:
                    predicted_gesture = detected_motion
                    confidence = 1.0
                    action_cfg = self.actions.get(detected_motion)
                    if action_cfg:
                        success = self.dispatcher.execute(detected_motion, action_cfg)
                        if success:
                            self.last_triggered_banner = action_cfg.get("description", detected_motion)
                            self.last_triggered_time = time.time()
                            triggered_now = True

                elif features is not None:
                    # Run static pose inference
                    feat_len = len(features)
                    expected_len = getattr(self.model, "n_features_in_", feat_len)
                    feats_to_predict = None

                    if expected_len == 63 and debug_info.get("all_norm_hands"):
                        best_g = None
                        best_c = 0.0
                        for h_norm in debug_info["all_norm_hands"]:
                            h_2d = h_norm.reshape(1, -1)
                            if hasattr(self.model, "predict_proba"):
                                probs = self.model.predict_proba(h_2d)[0]
                                idx = int(np.argmax(probs))
                                conf = float(probs[idx])
                                if conf > best_c and conf >= self.confidence_threshold:
                                    best_c = conf
                                    best_g = self.classes[idx]
                            else:
                                best_g = self.model.predict(h_2d)[0]
                                best_c = 1.0
                        predicted_gesture = best_g
                        confidence = best_c
                    else:
                        feats_to_predict = features.reshape(1, -1) if feat_len == expected_len else features[:expected_len].reshape(1, -1)
                        if hasattr(self.model, "predict_proba"):
                            probs = self.model.predict_proba(feats_to_predict)[0]
                            best_idx = np.argmax(probs)
                            confidence = probs[best_idx]
                            if confidence >= self.confidence_threshold:
                                predicted_gesture = self.classes[best_idx]
                        else:
                            predicted_gesture = self.model.predict(feats_to_predict)[0]
                            confidence = 1.0

                    # Explicitly ignore open_palm and fist so they do nothing right now
                    if predicted_gesture in ("open_palm", "fist"):
                        predicted_gesture = None
                        confidence = 0.0

                    # Update Debounce State Machine for static gestures
                    triggered_now = self._update_debounce_and_dispatch(predicted_gesture, confidence)

                # Render HUD
                self._draw_hud(
                    frame=frame,
                    predicted_gesture=predicted_gesture,
                    confidence=confidence,
                    fps=fps,
                    triggered_now=triggered_now,
                )

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("[EXIT] User requested shutdown.")
                    break

        finally:
            self.camera.release()
            self.detector.close()
            cv2.destroyAllWindows()

    def _update_debounce_and_dispatch(
        self, predicted_gesture: Optional[str], confidence: float
    ) -> bool:
        """
        Updates frame counters. If a gesture is held for required frames, triggers action.
        """
        if predicted_gesture is None or predicted_gesture in ("open_palm", "fist"):
            self.consecutive_frame_count = max(0, self.consecutive_frame_count - 1)
            return False

        if predicted_gesture == self.last_candidate_gesture:
            self.consecutive_frame_count += 1
        else:
            self.last_candidate_gesture = predicted_gesture
            self.consecutive_frame_count = 1

        # Check if debounce threshold met
        if self.consecutive_frame_count >= self.debounce_frames:
            # Check if this gesture has an action mapped
            action_cfg = self.actions.get(predicted_gesture)
            if action_cfg:
                success = self.dispatcher.execute(predicted_gesture, action_cfg)
                if success:
                    self.last_triggered_banner = action_cfg.get("description", predicted_gesture)
                    self.last_triggered_time = time.time()
                    self.consecutive_frame_count = 0  # Reset counter after firing
                    return True
            else:
                # Gesture recognized but unmapped
                self.consecutive_frame_count = 0

        return False

    def _draw_hud(
        self,
        frame: np.ndarray,
        predicted_gesture: Optional[str],
        confidence: float,
        fps: int,
        triggered_now: bool,
    ):
        """Draws visual feedback overlay on the frame."""
        h, w, _ = frame.shape

        # Top banner background
        cv2.rectangle(frame, (0, 0), (w, 70), (25, 25, 25), -1)

        # FPS indicator
        cv2.putText(frame, f"FPS: {fps}", (w - 110, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        if predicted_gesture:
            action_info = self.actions.get(predicted_gesture, {})
            desc = action_info.get("description", "No action mapped")

            # Gesture name and confidence
            text = f"Pose: {predicted_gesture.upper()} ({confidence * 100:.0f}%)"
            cv2.putText(frame, text, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Action description
            cv2.putText(frame, f"Action: {desc}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            # Debounce charge meter
            charge = min(self.consecutive_frame_count / self.debounce_frames, 1.0)
            bar_w = 180
            bar_x = w - 210
            bar_y = 40
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 16), (60, 60, 60), -1)
            # Progress fill
            fill_color = (0, 255, 255) if charge < 1.0 else (0, 255, 0)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * charge), bar_y + 16), fill_color, -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 16), (200, 200, 200), 1)
        else:
            cv2.putText(
                frame,
                "Ready - Show Hand Gesture",
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (160, 160, 160),
                2,
            )

        # Flash banner on recently triggered action
        now = time.time()
        if (now - self.last_triggered_time) < 1.0:
            banner_text = f"TRIGGERED: {self.last_triggered_banner.upper()}"
            cv2.rectangle(frame, (0, h - 50), (w, h), (0, 140, 0), -1)
            cv2.putText(
                frame,
                banner_text,
                (25, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )


def main():
    parser = argparse.ArgumentParser(description="Run the Gesture Macro Controller.")
    parser.add_argument(
        "--model",
        type=str,
        default=str(MODEL_PATH),
        help="Path to trained model .joblib file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate detections and log actions to terminal without actually pressing hotkeys.",
    )
    parser.add_argument(
        "--debounce",
        type=int,
        default=DEBOUNCE_CONSECUTIVE_FRAMES,
        help=f"Consecutive frames required to trigger (default: {DEBOUNCE_CONSECUTIVE_FRAMES}).",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=ACTION_COOLDOWN_SECONDS,
        help=f"Cooldown in seconds between repeated triggers (default: {ACTION_COOLDOWN_SECONDS}).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0).",
    )
    args = parser.parse_args()

    app = GestureMacroApp(
        model_path=Path(args.model),
        debounce_frames=args.debounce,
        cooldown_seconds=args.cooldown,
        dry_run=args.dry_run,
        camera_index=args.camera,
    )
    app.run()


if __name__ == "__main__":
    main()

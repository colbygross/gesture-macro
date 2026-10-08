"""
Visual camera and hand detection diagnostic tool.
Allows live preview, switching camera indices (0, 1, 2) on the fly with keyboard keys,
and tweaking detection sensitivity in real time.
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.camera import Camera
from src.detector import HandDetector


def run_diagnostics(initial_camera_idx: int = 0):
    camera_idx = initial_camera_idx
    camera = Camera(camera_index=camera_idx)

    if not camera.open():
        print(f"[ERROR] Could not open camera {camera_idx}")
        return

    confidence = 0.35
    detector = HandDetector(min_detection_confidence=confidence, min_tracking_confidence=confidence)

    window_name = "Webcam & Hand Detection Diagnostic (Press Q to quit)"
    prev_time = time.time()

    print("\n" + "=" * 60)
    print("  CAMERA & DETECTION DIAGNOSTIC TOOL")
    print("=" * 60)
    print("Controls:")
    print("  [0], [1], [2] : Switch between Camera Index 0, 1, or 2")
    print("  [+] / [-]     : Increase / Decrease detection confidence threshold")
    print("  [Q] or [ESC]  : Quit")
    print("=" * 60 + "\n")

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret:
                cv2.waitKey(30)
                continue

            now = time.time()
            fps = int(1.0 / max(now - prev_time, 1e-5))
            prev_time = now

            # Detect hand landmarks
            features, pixel_points = detector.find_landmarks(frame, draw=True)

            h, w, _ = frame.shape

            # Draw top info bar
            cv2.rectangle(frame, (0, 0), (w, 65), (30, 30, 30), -1)

            cam_info = f"Cam: {camera.camera_index} | FPS: {fps} | Conf: {confidence:.2f}"
            cv2.putText(frame, cam_info, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            if pixel_points:
                status = f"HAND DETECTED ({len(pixel_points)} landmarks)"
                color = (0, 255, 0)
                # Draw bounding box
                xs = [p[0] for p in pixel_points]
                ys = [p[1] for p in pixel_points]
                cv2.rectangle(frame, (min(xs) - 10, min(ys) - 10), (max(xs) + 10, max(ys) + 10), (0, 255, 0), 2)
            else:
                status = "NO HAND DETECTED - Move hand into frame"
                color = (0, 0, 255)

            cv2.putText(frame, status, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Help hints at bottom
            cv2.putText(
                frame,
                "Keys: [0/1/2] switch cam | [+/-] sensitivity | [Q] quit",
                (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break
            elif key in (ord("0"), ord("1"), ord("2")):
                new_idx = int(chr(key))
                if new_idx != camera.camera_index:
                    print(f"\n[SWITCH] Switching to camera index {new_idx}...")
                    camera.release()
                    camera = Camera(camera_index=new_idx)
                    if not camera.open():
                        print(f"[WARN] Camera {new_idx} failed, switching back to {camera_idx}")
                        camera = Camera(camera_index=camera_idx)
                        camera.open()
                    else:
                        camera_idx = new_idx
            elif key == ord("+") or key == ord("="):
                confidence = min(0.9, confidence + 0.05)
                detector.close()
                detector = HandDetector(min_detection_confidence=confidence, min_tracking_confidence=confidence)
                print(f"[CONFIDENCE] Increased to {confidence:.2f}")
            elif key == ord("-") or key == ord("_"):
                confidence = max(0.1, confidence - 0.05)
                detector.close()
                detector = HandDetector(min_detection_confidence=confidence, min_tracking_confidence=confidence)
                print(f"[CONFIDENCE] Decreased to {confidence:.2f}")

    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_diagnostics(0)

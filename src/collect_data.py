"""
Interactive data collection tool for custom hand gestures.
Captures normalized 3D landmarks from webcam and saves them to a CSV dataset.
Works seamlessly on Windows and macOS.
"""

import sys
import time
import argparse
from pathlib import Path
import csv
import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.camera import Camera
from src.detector import HandDetector
from src.config import GESTURES_CSV_PATH, DATA_DIR


def ensure_dataset_file_exists(csv_path: Path):
    """Creates the dataset CSV with header if it doesn't already exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["label"] + [f"f{i}" for i in range(63)]
            writer.writerow(header)
        print(f"[INIT] Created new dataset file at {csv_path}")


def get_existing_counts(csv_path: Path) -> dict:
    """Returns a dictionary with sample counts per label in the dataset."""
    if not csv_path.exists():
        return {}
    counts = {}
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if row:
                label = row[0]
                counts[label] = counts.get(label, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="Record custom hand gestures for macro control.")
    parser.add_argument(
        "--gesture",
        type=str,
        required=True,
        help="Name of the gesture to record (e.g. fist, open_palm, peace, thumbs_up).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of hand landmark samples to collect (default: 100).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0).",
    )
    args = parser.parse_args()

    gesture_name = args.gesture.strip().lower()
    target_samples = args.samples

    ensure_dataset_file_exists(GESTURES_CSV_PATH)
    counts = get_existing_counts(GESTURES_CSV_PATH)
    print("\n" + "=" * 60)
    print(f"  GESTURE DATA COLLECTOR - Target: '{gesture_name}'")
    print(f"  Existing samples: {counts}")
    print("=" * 60)
    print("Instructions:")
    print("  1. Position your hand in front of the camera.")
    print("  2. Press [SPACE] to start recording.")
    print("  3. Slowly rotate/tilt your hand slightly to capture variations.")
    print("  4. Press [Q] or [ESC] at any time to exit.")
    print("=" * 60 + "\n")

    camera = Camera(camera_index=args.camera)
    if not camera.open():
        sys.exit(1)

    detector = HandDetector()
    is_recording = False
    samples_collected = 0
    saved_rows = []

    window_name = f"Data Collector - {gesture_name} (Press SPACE to start)"

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret:
                print("[ERROR] Failed to grab webcam frame.")
                break

            # Find landmarks and draw skeleton
            features, _ = detector.find_landmarks(frame, draw=True)

            h, w, _ = frame.shape

            # Status overlay banner
            if not is_recording:
                cv2.rectangle(frame, (0, 0), (w, 50), (40, 40, 40), -1)
                cv2.putText(
                    frame,
                    f"Target: [{gesture_name}] | Press SPACE to record {target_samples} samples",
                    (15, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )
            else:
                # In recording mode
                if features is not None:
                    saved_rows.append([gesture_name] + list(features))
                    samples_collected += 1

                # Recording HUD banner
                progress = min(samples_collected / target_samples, 1.0)
                cv2.rectangle(frame, (0, 0), (w, 55), (0, 100, 0), -1)
                # Progress bar
                cv2.rectangle(frame, (0, 50), (int(w * progress), 55), (0, 255, 0), -1)

                status_text = f"RECORDING '{gesture_name}': {samples_collected}/{target_samples}"
                if features is None:
                    status_text += " [NO HAND DETECTED]"

                cv2.putText(
                    frame,
                    status_text,
                    (15, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                if samples_collected >= target_samples:
                    print(f"\n[DONE] Collected {samples_collected} samples for '{gesture_name}'!")
                    break

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):  # Space to start
                if not is_recording:
                    is_recording = True
                    print(f"[START] Recording {target_samples} samples for '{gesture_name}'...")
            elif key == ord("q") or key == 27:  # q or ESC
                print("[ABORT] Stopped by user.")
                break

    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()

    # Save collected samples
    if saved_rows:
        with open(GESTURES_CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(saved_rows)
        print(f"[SAVED] Appended {len(saved_rows)} rows to {GESTURES_CSV_PATH}")
        updated_counts = get_existing_counts(GESTURES_CSV_PATH)
        print(f"[DATASET STATUS] Total samples per gesture: {updated_counts}")
    else:
        print("[INFO] No samples were saved.")


if __name__ == "__main__":
    main()

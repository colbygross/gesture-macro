"""
Dynamic Motion Gesture Learner & Inference Engine.
Allows users to train custom machine learning models on their own distinct hand motions
(e.g., custom swipes, waves, circles, V-shapes, zigzags) rather than relying only on heuristic thresholds.

Features:
- Arc-length equidistant spatial resampling (invariant to stroke speed).
- Kinematic & trajectory feature extraction (net displacement, total path length,
  straightness ratio, start-to-end angle, velocity, max displacement, and 48 resampled coordinates).
- Dataset CSV logging and fast Random Forest training.
- Real-time inference with confidence estimation and graceful fallback.
- Interactive MotionGestureRecorder with Hand-Reset Gate.
"""

import os
import time
import math
import csv
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import joblib

from src.config import MOTION_DATA_CSV_PATH, MOTION_MODEL_PATH, DATA_DIR, MODELS_DIR

NUM_RESAMPLE_POINTS = 24
MIN_STROKE_POINTS = 2
MIN_STROKE_DISPLACEMENT = 0.045


def resample_trajectory(
    pts: List[Tuple[float, float, float]], num_points: int = NUM_RESAMPLE_POINTS
) -> np.ndarray:
    """
    Spatially resamples a temporal trajectory into `num_points` equidistant points
    along its cumulative arc-length. The trajectory is normalized relative to its
    start point (0, 0).

    Args:
        pts: List of (x, y, timestamp) tuples.
        num_points: Target number of spatial points (default 24).

    Returns:
        Array of shape (num_points, 2) with resampled (dx, dy) coordinates.
    """
    if len(pts) < 2:
        return np.zeros((num_points, 2), dtype=np.float32)

    x0, y0 = pts[0][0], pts[0][1]
    rel_coords = np.array([[p[0] - x0, p[1] - y0] for p in pts], dtype=np.float32)

    # Compute incremental arc lengths
    diffs = np.diff(rel_coords, axis=0)
    step_dists = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_dists = np.insert(np.cumsum(step_dists), 0, 0.0)
    total_len = cum_dists[-1]

    if total_len < 1e-5:
        return np.zeros((num_points, 2), dtype=np.float32)

    # Equidistant sample points along the arc length
    target_dists = np.linspace(0.0, total_len, num_points)
    resampled_x = np.interp(target_dists, cum_dists, rel_coords[:, 0])
    resampled_y = np.interp(target_dists, cum_dists, rel_coords[:, 1])

    return np.column_stack([resampled_x, resampled_y]).astype(np.float32)


def get_feature_names(num_points: int = NUM_RESAMPLE_POINTS) -> List[str]:
    """Returns the names of all 58 extracted trajectory features."""
    meta_names = [
        "net_dx",
        "net_dy",
        "net_dist",
        "total_path_length",
        "straightness_ratio",
        "aspect_ratio",
        "start_to_end_angle",
        "duration",
        "avg_speed",
        "max_disp",
    ]
    coord_names = []
    for i in range(num_points):
        coord_names.extend([f"p{i}_dx", f"p{i}_dy"])
    return meta_names + coord_names


def extract_trajectory_features(
    pts: List[Tuple[float, float, float]], num_points: int = NUM_RESAMPLE_POINTS
) -> np.ndarray:
    """
    Extracts a 58-dimensional kinematic and geometric feature vector from a motion trajectory.

    Features:
    1-10: net_dx, net_dy, net_dist, total_path, straightness_ratio (D/L),
          aspect_ratio, start_to_end_angle, duration, avg_speed, max_disp.
    11-58: 48 resampled coordinates (x_k, y_k) relative to start point.
    """
    total_feat_count = 10 + (num_points * 2)
    if len(pts) < 2:
        return np.zeros(total_feat_count, dtype=np.float32)

    start_x, start_y, start_t = pts[0]
    end_x, end_y, end_t = pts[-1]

    net_dx = float(end_x - start_x)
    net_dy = float(end_y - start_y)
    net_dist = float(math.hypot(net_dx, net_dy))

    all_x = np.array([p[0] for p in pts], dtype=np.float32)
    all_y = np.array([p[1] for p in pts], dtype=np.float32)

    diffs_x = np.diff(all_x)
    diffs_y = np.diff(all_y)
    total_path = float(np.sum(np.hypot(diffs_x, diffs_y)))

    straightness_ratio = float(net_dist / max(total_path, 1e-4))
    aspect_ratio = float(abs(net_dx) / max(abs(net_dy), 1e-4))
    angle = float(math.atan2(net_dy, net_dx))

    duration = float(max(end_t - start_t, 1e-4))
    avg_speed = float(total_path / duration)

    disp_from_start = np.hypot(all_x - start_x, all_y - start_y)
    max_disp = float(np.max(disp_from_start))

    resampled = resample_trajectory(pts, num_points=num_points)
    flat_resampled = resampled.flatten()

    meta = np.array([
        net_dx,
        net_dy,
        net_dist,
        total_path,
        straightness_ratio,
        aspect_ratio,
        angle,
        duration,
        avg_speed,
        max_disp,
    ], dtype=np.float32)

    return np.concatenate([meta, flat_resampled]).astype(np.float32)


def save_motion_samples(
    gesture_name: str,
    strokes: List[List[Tuple[float, float, float]]],
    csv_path: Path = MOTION_DATA_CSV_PATH,
    num_points: int = NUM_RESAMPLE_POINTS,
) -> int:
    """
    Appends recorded motion trajectory strokes to the training dataset CSV.
    """
    if not strokes:
        return 0

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    feature_names = get_feature_names(num_points=num_points)
    header = ["gesture"] + feature_names

    saved_count = 0
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)

        for stroke in strokes:
            if len(stroke) < MIN_STROKE_POINTS:
                continue
            feats = extract_trajectory_features(stroke, num_points=num_points)
            row = [gesture_name] + [f"{v:.6f}" for v in feats]
            writer.writerow(row)
            saved_count += 1

    return saved_count


def load_motion_dataset(
    csv_path: Path = MOTION_DATA_CSV_PATH,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Loads dataset from CSV and returns (X, y, class_names).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Motion dataset not found at {csv_path}")

    X_list = []
    y_list = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError("CSV file is empty")

        for row in reader:
            if not row or len(row) < 2:
                continue
            gesture_label = row[0].strip()
            feature_vals = [float(val) for val in row[1:]]
            y_list.append(gesture_label)
            X_list.append(feature_vals)

    if not X_list:
        raise ValueError("No samples found in motion dataset CSV")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    classes = sorted(list(set(y_list)))

    return X, y, classes


def train_motion_model(
    csv_path: Path = MOTION_DATA_CSV_PATH,
    model_path: Path = MOTION_MODEL_PATH,
    num_points: int = NUM_RESAMPLE_POINTS,
) -> Dict[str, Any]:
    """
    Trains a Random Forest Classifier on recorded motion gestures and saves it to disk.
    Uses lazy scikit-learn imports to preserve fast cold startup time.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

    X, y, classes = load_motion_dataset(csv_path=csv_path)

    if len(classes) < 2:
        raise ValueError(
            f"Need at least 2 distinct motion classes to train. Currently have: {classes}"
        )

    # Train / test split if dataset is large enough
    if len(y) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    clf = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, target_names=classes, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": clf,
        "classes": classes,
        "feature_names": get_feature_names(num_points=num_points),
        "num_points": num_points,
        "accuracy": accuracy,
        "trained_at": time.time(),
    }
    joblib.dump(payload, model_path)

    return {
        "accuracy": accuracy,
        "classes": classes,
        "report": report,
        "confusion_matrix": cm,
        "sample_count": len(y),
        "model_path": str(model_path),
    }


class MotionModel:
    """
    Live inference wrapper for predicting custom dynamic hand motion gestures.
    """

    def __init__(self, model_path: Path = MOTION_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.classes: List[str] = []
        self.num_points = NUM_RESAMPLE_POINTS
        self.load()

    def load(self) -> bool:
        """Loads model from disk if present."""
        if not self.model_path.exists():
            self.model = None
            self.classes = []
            return False

        try:
            payload = joblib.load(self.model_path)
            self.model = payload["model"]
            self.classes = payload["classes"]
            self.num_points = payload.get("num_points", NUM_RESAMPLE_POINTS)
            return True
        except Exception as e:
            print(f"[WARN] Failed to load motion model: {e}")
            self.model = None
            self.classes = []
            return False

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and len(self.classes) > 0

    def predict(
        self, pts: List[Tuple[float, float, float]], min_disp: float = MIN_STROKE_DISPLACEMENT
    ) -> Tuple[Optional[str], float]:
        """
        Predicts motion gesture class and confidence from a trajectory stroke.
        """
        if not self.is_loaded or len(pts) < MIN_STROKE_POINTS:
            return None, 0.0

        # Quick displacement check across stroke to ignore stationary twitches
        # Checks peak displacement relative to start point, preserving flicks even after hand begins return
        start_x, start_y = pts[0][0], pts[0][1]
        disp = max(math.hypot(p[0] - start_x, p[1] - start_y) for p in pts)
        if disp < min_disp:
            return None, 0.0

        feats = extract_trajectory_features(pts, num_points=self.num_points)
        x_in = feats.reshape(1, -1)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x_in)[0]
            best_idx = int(np.argmax(probs))
            conf = float(probs[best_idx])
            pred_class = self.classes[best_idx]
            return pred_class, conf
        else:
            pred_class = self.model.predict(x_in)[0]
            return pred_class, 1.0


class MotionGestureRecorder:
    """
    Interactive stroke recorder for training distinct user motion gestures.
    Implements Start-Anchor and Hand-Reset Gate so strokes are clean and individual
    movements are not merged with hand-reset returns.
    """

    def __init__(
        self,
        min_reset_time: float = 1.8,
        min_stroke_disp: float = 0.045,
        max_stroke_disp_pullback: float = 0.008,
        steady_hold_time: float = 0.35,
    ):
        self.min_reset_time = min_reset_time
        self.min_stroke_disp = min_stroke_disp
        self.max_stroke_disp_pullback = max_stroke_disp_pullback
        self.steady_hold_time = steady_hold_time

        self.is_recording = False
        self.gesture_name = ""
        self.target_count = 10
        self.recorded_strokes: List[List[Tuple[float, float, float]]] = []

        # Hand-Reset Gate state
        self.waiting_for_reset = False
        self.reset_start_time = 0.0
        self.anchor_hold_start: Optional[float] = None

        # Current active stroke buffer
        self.current_stroke: List[Tuple[float, float, float]] = []
        self.peak_disp = 0.0
        self.last_stroke_time = 0.0

    def start_recording(self, gesture_name: str, target_count: int = 10):
        """Initializes a new recording session for gesture_name."""
        self.gesture_name = gesture_name.strip().lower()
        self.target_count = target_count
        self.recorded_strokes = []
        self.current_stroke = []
        self.waiting_for_reset = False
        self.reset_start_time = 0.0
        self.anchor_hold_start = None
        self.peak_disp = 0.0
        self.is_recording = True

    def stop_recording(self):
        """Stops the recording session."""
        self.is_recording = False
        self.current_stroke = []
        self.anchor_hold_start = None

    @property
    def current_count(self) -> int:
        return len(self.recorded_strokes)

    @property
    def is_finished(self) -> bool:
        return self.is_recording and len(self.recorded_strokes) >= self.target_count

    def reset_time_remaining(self, timestamp: Optional[float] = None) -> float:
        """Returns seconds remaining in the mandatory reset phase."""
        if not self.waiting_for_reset:
            return 0.0
        now = timestamp if timestamp is not None else time.time()
        return max(0.0, self.min_reset_time - (now - self.reset_start_time))

    def update_state(self, is_anchored: bool, timestamp: Optional[float] = None):
        """Checks if the user has returned hand to starting anchor and rested."""
        if not self.is_recording:
            return
        now = timestamp if timestamp is not None else time.time()
        if self.waiting_for_reset:
            # 1. Mandatory reset pause (1.8s) to bring hand back without rushing
            if (now - self.reset_start_time) < self.min_reset_time:
                self.anchor_hold_start = None
                return
            # 2. Must hold hand still at rest anchor for steady_hold_time
            if is_anchored:
                if self.anchor_hold_start is None:
                    self.anchor_hold_start = now
                elif (now - self.anchor_hold_start) >= self.steady_hold_time:
                    self.waiting_for_reset = False
                    self.current_stroke = []
                    self.peak_disp = 0.0
                    self.anchor_hold_start = None
            else:
                self.anchor_hold_start = None

    def feed_point(
        self,
        x: float,
        y: float,
        timestamp: float,
        is_anchored: bool,
    ) -> Optional[List[Tuple[float, float, float]]]:
        """
        Feeds a hand position point into the stroke recorder.
        Returns the completed stroke when a repetition lands, or None if still in progress.
        """
        if not self.is_recording or self.waiting_for_reset:
            return None

        # If user is at anchor, reset stroke buffer to begin fresh from here
        if is_anchored:
            self.current_stroke = [(x, y, timestamp)]
            self.peak_disp = 0.0
            return None

        # User has left anchor and is executing the gesture
        self.current_stroke.append((x, y, timestamp))
        if len(self.current_stroke) < 2:
            return None

        start_x, start_y = self.current_stroke[0][0], self.current_stroke[0][1]
        cur_disp = math.hypot(x - start_x, y - start_y)

        if cur_disp > self.peak_disp:
            self.peak_disp = cur_disp

        # Stroke completion check:
        # User has achieved significant displacement and either decelerated, peaked, or pulled back
        stroke_completed = False
        if self.peak_disp >= self.min_stroke_disp:
            # 1. Pullback detection (hand pulled back after reaching peak)
            if cur_disp <= (self.peak_disp - self.max_stroke_disp_pullback):
                stroke_completed = True
            # 2. Settled or paused at end point for >= 20ms
            elif (timestamp - self.current_stroke[0][2]) >= 0.020:
                if len(self.current_stroke) >= 3:
                    recent = self.current_stroke[-2:]
                    disp_step = math.hypot(recent[1][0] - recent[0][0], recent[1][1] - recent[0][1])
                    dt_step = max(recent[1][2] - recent[0][2], 1e-4)
                    if (disp_step / dt_step) < 0.25:
                        stroke_completed = True
                elif cur_disp < self.peak_disp:
                    stroke_completed = True

        if stroke_completed:
            completed = list(self.current_stroke)
            self.recorded_strokes.append(completed)
            self.waiting_for_reset = True
            self.reset_start_time = timestamp
            self.anchor_hold_start = None
            self.current_stroke = []
            self.peak_disp = 0.0
            return completed

        return None

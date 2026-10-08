"""
Gesture Macro Studio - Unified Desktop GUI Application.
Features:
- Live Upper-Body & Arm Rig webcam view with skeleton overlay.
- Interactive Gesture Library manager & 3-2-1 countdown sample recorder.
- 1-Click in-memory ML model retrainer with live accuracy reporting.
- Interactive Macro & Hotkey binding matrix with persistent JSON storage.
- Active macro listening toggle with visual debounce charge meter.
Cross-platform support for Windows and macOS (PySide6).
"""

import sys
import os
import time
import csv
from pathlib import Path
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
import joblib

from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont, QColor
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QTabWidget,
    QProgressBar,
    QGroupBox,
    QMessageBox,
    QFrame,
)

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.camera import Camera
from src.detector import UpperBodyRigDetector
from src.actions import ActionDispatcher
from src.config import (
    MODEL_PATH,
    GESTURES_CSV_PATH,
    MOTION_DATA_CSV_PATH,
    MOTION_MODEL_PATH,
    DATA_DIR,
    DEFAULT_GESTURE_ACTIONS,
    load_macro_actions,
    save_macro_actions,
    load_motion_profile,
    save_motion_profile,
    DEBOUNCE_CONSECUTIVE_FRAMES,
    ACTION_COOLDOWN_SECONDS,
    PREDICTION_CONFIDENCE_THRESHOLD,
    IS_MAC,
)
from src.motion import MotionTracker, MotionCalibrator
from src.motion_learner import (
    MotionModel,
    MotionGestureRecorder,
    save_motion_samples,
    train_motion_model,
)


class VideoWorker(QThread):
    """
    Background worker thread running OpenCV camera capture,
    MediaPipe Upper-Body detection, and real-time inference.
    """

    frame_ready = Signal(QImage)
    prediction_ready = Signal(str, float, float)  # gesture, confidence, debounce_progress
    macro_triggered = Signal(str, str)            # gesture, description
    status_updated = Signal(str)
    calibration_progress = Signal(str, int, int)  # direction, current_count, target_count
    calibration_finished = Signal(dict)           # calibrated profile dict
    motion_recording_progress = Signal(str, int, int)  # gesture, current_count, target_count
    motion_recording_finished = Signal(str, int)       # gesture, total_samples

    def __init__(
        self,
        camera_index: int = 0,
        debounce_frames: int = DEBOUNCE_CONSECUTIVE_FRAMES,
        cooldown_seconds: float = ACTION_COOLDOWN_SECONDS,
        confidence_threshold: float = PREDICTION_CONFIDENCE_THRESHOLD,
    ):
        super().__init__()
        self.camera_index = camera_index
        self.debounce_frames = debounce_frames
        self.cooldown_seconds = cooldown_seconds
        self.confidence_threshold = confidence_threshold

        self.running = True
        self.listening = True
        self.model_payload: Optional[Dict[str, Any]] = None
        self.model = None
        self.classes = []

        self.camera: Optional[Camera] = None
        self.detector: Optional[UpperBodyRigDetector] = None
        self.dispatcher = ActionDispatcher(cooldown_seconds=self.cooldown_seconds)
        self.macros = load_macro_actions()
        self.motion_tracker = MotionTracker()
        self.motion_tracker.load_calibrated_profile()
        self.motion_calibrator = MotionCalibrator()
        self.motion_model = MotionModel(model_path=MOTION_MODEL_PATH)
        self.motion_recorder = MotionGestureRecorder()

        # State tracking
        self.last_candidate_gesture: Optional[str] = None
        self.consecutive_frame_count = 0

        # Recording state
        self.is_recording = False
        self.recording_gesture = ""
        self.target_samples = 100
        self.recorded_samples_buffer: List[List[Any]] = []
        self.countdown_seconds = 0
        self.countdown_start_time = 0.0

        self._load_classifier_if_exists()
        self._load_motion_classifier_if_exists()

    def _load_classifier_if_exists(self):
        if MODEL_PATH.exists():
            try:
                self.model_payload = joblib.load(MODEL_PATH)
                self.model = self.model_payload["model"]
                self.classes = self.model_payload["classes"]
                print(f"[WORKER] Loaded model with classes: {self.classes}")
            except Exception as e:
                print(f"[WORKER] Could not load model: {e}")

    def _load_motion_classifier_if_exists(self):
        if self.motion_model.load():
            self.motion_tracker.set_motion_model(self.motion_model)
            print(f"[WORKER] Loaded dynamic motion model with classes: {self.motion_model.classes}")

    def reload_model(self):
        self._load_classifier_if_exists()
        self._load_motion_classifier_if_exists()

    def start_motion_recording(self, gesture_name: str, target_count: int = 8):
        self.motion_recorder.start_recording(gesture_name, target_count)

    def set_camera_index(self, index: int):
        self.camera_index = index
        if self.camera:
            self.camera.release()
            self.camera = Camera(camera_index=index)
            self.camera.open()

    def start_recording(self, gesture_name: str, target_samples: int):
        self.recording_gesture = gesture_name.strip().lower()
        self.target_samples = target_samples
        self.recorded_samples_buffer = []
        self.countdown_seconds = 3
        self.countdown_start_time = time.time()
        self.is_recording = True

    def run(self):
        self.camera = Camera(camera_index=self.camera_index)
        retries = 0
        while self.running and not self.camera.open():
            retries += 1
            if retries <= 5:
                self.status_updated.emit(f"Camera busy, retrying... ({retries}/5)")
                self.msleep(600)
            else:
                self.status_updated.emit("Camera busy or disconnected.")
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "CAMERA UNAVAILABLE / IN USE", (70, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(placeholder, "Close other apps using camera or pick another Camera above", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
                rgb = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                self.frame_ready.emit(q_img)
                return

        self.status_updated.emit("Camera connected")
        self.detector = UpperBodyRigDetector()
        prev_time = time.time()

        while self.running:
            ret, frame = self.camera.read_frame()
            if not ret or frame is None:
                self.msleep(30)
                continue

            now = time.time()
            fps = int(1.0 / max(now - prev_time, 1e-5))
            prev_time = now

            # Extract landmarks and draw skeleton
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

            # Interactive Dynamic Motion Gesture Recording Mode
            if self.motion_recorder.is_recording:
                self.motion_recorder.update_state(self.motion_tracker.is_anchored, timestamp=now)
                g_name = self.motion_recorder.gesture_name
                count = self.motion_recorder.current_count
                target = self.motion_recorder.target_count
                remaining = self.motion_recorder.reset_time_remaining(now)

                if self.motion_recorder.waiting_for_reset:
                    cv2.rectangle(frame, (0, 0), (w, 65), (0, 140, 255), -1)
                    if remaining > 0.0:
                        header_txt = f"RESET HAND ({remaining:.1f}s): Return to start & pause"
                    else:
                        header_txt = "RESET HAND: Hold hand still at start position..."
                    cv2.putText(
                        frame,
                        header_txt,
                        (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"Stroke captured [{count}/{target}]. Rest hand in neutral position...",
                        (15, 52),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.rectangle(frame, (0, 0), (w, 65), (20, 110, 20), -1)
                    cv2.putText(
                        frame,
                        f"READY: Flick '{g_name.upper()}'",
                        (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 200),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"Perform crisp wrist flick [{count}/{target}]",
                        (15, 52),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (200, 255, 200),
                        1,
                        cv2.LINE_AA,
                    )

                tracking_pt = debug_info.get("tracking_point") or debug_info.get("palm_center")
                if tracking_pt:
                    norm_x, norm_y = tracking_pt
                    stroke = self.motion_recorder.feed_point(
                        norm_x, norm_y, now, is_anchored=self.motion_tracker.is_anchored
                    )
                    if stroke:
                        self.motion_recording_progress.emit(g_name, self.motion_recorder.current_count, target)
                        if self.motion_recorder.is_finished:
                            saved = save_motion_samples(g_name, self.motion_recorder.recorded_strokes)
                            self.motion_recorder.stop_recording()
                            self.motion_recording_finished.emit(g_name, saved)

            # Interactive Motion Calibration Mode Handling
            elif self.motion_calibrator.is_calibrating:
                target_dir = self.motion_calibrator.current_direction
                h, w, _ = frame.shape

                # Update Hand-Reset Gate with current anchorage state & timestamp
                self.motion_calibrator.update_state(self.motion_tracker.is_anchored, timestamp=now)

                label_text = self.motion_calibrator.current_label
                count = self.motion_calibrator.current_count
                target = self.motion_calibrator.TARGET_SAMPLES_PER_DIR
                remaining = self.motion_calibrator.reset_time_remaining(now)

                if self.motion_calibrator.waiting_for_reset:
                    # Amber Banner: Hand needs to return to starting position & pause
                    cv2.rectangle(frame, (0, 0), (w, 65), (0, 140, 255), -1)
                    if remaining > 0.0:
                        header_txt = f"RESET HAND ({remaining:.1f}s): Return to start & pause"
                    else:
                        header_txt = "RESET HAND: Hold hand still at start position..."
                    cv2.putText(
                        frame,
                        header_txt,
                        (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"Stroke captured [{count}/{target}]. Rest hand in neutral position...",
                        (15, 52),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                else:
                    # Green Banner: Ready to swipe!
                    cv2.rectangle(frame, (0, 0), (w, 65), (20, 110, 20), -1)
                    cv2.putText(
                        frame,
                        f"READY: Flick {label_text}",
                        (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 200),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"Perform crisp wrist flick [{count}/{target}]",
                        (15, 52),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (200, 255, 200),
                        1,
                        cv2.LINE_AA,
                    )

                # Check if recent motion completed a stroke in target direction (Start-to-End comparison)
                stroke = self.motion_calibrator.check_and_record_stroke(
                    list(self.motion_tracker.trajectory),
                    is_anchored=self.motion_tracker.is_anchored,
                    active_anchor=self.motion_tracker.active_start_pos,
                )
                if stroke:
                    self.motion_tracker.reset()
                    self.calibration_progress.emit(
                        self.motion_calibrator.current_direction or target_dir or "",
                        self.motion_calibrator.current_count,
                        target,
                    )
                    if self.motion_calibrator.is_finished:
                        profile = self.motion_calibrator.save_and_apply(self.motion_tracker)
                        self.calibration_finished.emit(profile)

            # Trigger dynamic motion macro immediately if detected (and not calibrating or recording)
            elif detected_motion and self.listening and not self.is_recording and not self.motion_recorder.is_recording:
                action_cfg = self.macros.get(detected_motion)
                if action_cfg:
                    success = self.dispatcher.execute(detected_motion, action_cfg)
                    if success:
                        desc = action_cfg.get("description", detected_motion)
                        self.macro_triggered.emit(detected_motion, desc)
                else:
                    self.macro_triggered.emit(detected_motion, detected_motion.replace("_", " ").title())

            predicted_gesture = None
            confidence = 0.0

            # Real-time inference if model is loaded
            if features is not None and self.model is not None:
                try:
                    feat_len = len(features)
                    expected_len = getattr(self.model, "n_features_in_", feat_len)
                    feats_to_predict = None

                    if expected_len == 63 and debug_info.get("all_norm_hands"):
                        # Test both hands and choose the one making a confident gesture
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
                        if feat_len == expected_len:
                            feats_to_predict = features.reshape(1, -1)
                        elif expected_len == 108 and feat_len == 63:
                            feats_to_predict = np.pad(features, (45, 0)).reshape(1, -1)
                        else:
                            feats_to_predict = features.reshape(1, -1)

                        if hasattr(self.model, "predict_proba"):
                            probs = self.model.predict_proba(feats_to_predict)[0]
                            best_idx = int(np.argmax(probs))
                            confidence = float(probs[best_idx])
                            if confidence >= self.confidence_threshold:
                                predicted_gesture = self.classes[best_idx]
                        else:
                            predicted_gesture = self.model.predict(feats_to_predict)[0]
                            confidence = 1.0
                except Exception:
                    pass

                # Explicitly ignore open_palm and fist so they do nothing right now
                if predicted_gesture in ("open_palm", "fist"):
                    predicted_gesture = None
                    confidence = 0.0

            # Handle Sample Recording Mode
            if self.is_recording:
                elapsed = now - self.countdown_start_time
                if elapsed < 3.0:
                    remaining = 3 - int(elapsed)
                    # Overlay large countdown
                    h, w, _ = frame.shape
                    cv2.putText(
                        frame,
                        f"GET READY: {remaining}",
                        (w // 2 - 170, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.8,
                        (0, 255, 255),
                        4,
                    )
                else:
                    # Capturing samples
                    if features is not None:
                        # Check existing dataset column count to match structure
                        expected_feats = getattr(self.model, "n_features_in_", len(features))
                        if expected_feats == 63 and debug_info.get("hand_detected"):
                            self.recorded_samples_buffer.append([self.recording_gesture] + list(debug_info["norm_hand"]))
                        else:
                            self.recorded_samples_buffer.append([self.recording_gesture] + list(features))

                    cur_count = len(self.recorded_samples_buffer)
                    h, w, _ = frame.shape
                    progress = min(cur_count / self.target_samples, 1.0)
                    cv2.rectangle(frame, (0, 0), (w, 50), (0, 120, 0), -1)
                    cv2.rectangle(frame, (0, 45), (int(w * progress), 50), (0, 255, 0), -1)
                    cv2.putText(
                        frame,
                        f"RECORDING '{self.recording_gesture}': {cur_count}/{self.target_samples}",
                        (15, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                    if cur_count >= self.target_samples:
                        self._save_recorded_samples()
                        self.is_recording = False

            # Update debounce & trigger actions if active
            debounce_progress = 0.0
            if detected_motion:
                predicted_gesture = detected_motion
                confidence = 1.0
                debounce_progress = 1.0
            elif self.listening and not self.is_recording:
                debounce_progress = self._update_debounce(predicted_gesture)

            # Signal prediction and charge to GUI
            self.prediction_ready.emit(
                predicted_gesture or "",
                confidence,
                debounce_progress,
            )

            # Convert frame to QImage for GUI
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.frame_ready.emit(q_img.copy())

        # Cleanup
        if self.camera:
            self.camera.release()
        if self.detector:
            self.detector.close()

    def _update_debounce(self, predicted_gesture: Optional[str]) -> float:
        if predicted_gesture is None or predicted_gesture in ("open_palm", "fist"):
            self.consecutive_frame_count = max(0, self.consecutive_frame_count - 1)
            return 0.0

        if predicted_gesture == self.last_candidate_gesture:
            self.consecutive_frame_count += 1
        else:
            self.last_candidate_gesture = predicted_gesture
            self.consecutive_frame_count = 1

        progress = min(self.consecutive_frame_count / self.debounce_frames, 1.0)

        if self.consecutive_frame_count >= self.debounce_frames:
            action_cfg = self.macros.get(predicted_gesture)
            if action_cfg:
                success = self.dispatcher.execute(predicted_gesture, action_cfg)
                if success:
                    desc = action_cfg.get("description", predicted_gesture)
                    self.macro_triggered.emit(predicted_gesture, desc)
                    self.consecutive_frame_count = 0
                    return 1.0
            self.consecutive_frame_count = 0

        return progress

    def _save_recorded_samples(self):
        if not self.recorded_samples_buffer:
            return

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        file_exists = GESTURES_CSV_PATH.exists()

        with open(GESTURES_CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                feat_count = len(self.recorded_samples_buffer[0]) - 1
                header = ["label"] + [f"f{i}" for i in range(feat_count)]
                writer.writerow(header)
            writer.writerows(self.recorded_samples_buffer)

        print(f"[WORKER] Saved {len(self.recorded_samples_buffer)} samples for '{self.recording_gesture}'")
        self.status_updated.emit(f"Recorded {len(self.recorded_samples_buffer)} samples for '{self.recording_gesture}'")

    def stop(self):
        self.running = False
        self.wait()


class GestureStudioWindow(QMainWindow):
    """
    Main Studio Window with live video rig, dataset recorder,
    one-click model retrainer, and macro mapping matrix.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture Macro Studio - Upper Body & Hands")
        self.resize(1180, 720)
        self.setMinimumSize(960, 600)

        self.macros = load_macro_actions()

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(14)

        # 1. Left Panel (Live Video & Rig Feed)
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, stretch=6)

        # 2. Right Panel (Tabs for Gestures, Training & Macros)
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, stretch=5)

        # Apply dark styling
        self._apply_dark_theme()

        # Start Video Worker thread
        self.worker = VideoWorker()
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.prediction_ready.connect(self._on_prediction_ready)
        self.worker.macro_triggered.connect(self._on_macro_triggered)
        self.worker.status_updated.connect(self._on_status_updated)
        self.worker.calibration_progress.connect(self._on_calibration_progress)
        self.worker.calibration_finished.connect(self._on_calibration_finished)
        self.worker.motion_recording_progress.connect(self._on_motion_recording_progress)
        self.worker.motion_recording_finished.connect(self._on_motion_recording_finished)
        self.worker.start()

        # Load initial dataset counts & motion profile
        self._refresh_dataset_table()
        self._refresh_motion_profile_ui()

    def _create_left_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Header controls
        ctrl_bar = QHBoxLayout()
        cam_lbl = QLabel("Camera:")
        self.cam_combo = QComboBox()
        self.cam_combo.addItems(["Camera 0", "Camera 1", "Camera 2"])
        self.cam_combo.currentIndexChanged.connect(self._on_camera_changed)

        self.listen_btn = QPushButton("● Listening Active")
        self.listen_btn.setCheckable(True)
        self.listen_btn.setChecked(True)
        self.listen_btn.setObjectName("listenBtn")
        self.listen_btn.clicked.connect(self._toggle_listening)

        ctrl_bar.addWidget(cam_lbl)
        ctrl_bar.addWidget(self.cam_combo)
        ctrl_bar.addStretch()
        ctrl_bar.addWidget(self.listen_btn)
        layout.addLayout(ctrl_bar)

        # Video Canvas
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #151515; border-radius: 8px;")
        self.video_label.setMinimumSize(540, 380)
        layout.addWidget(self.video_label, stretch=1)

        # Live Status & Debounce HUD
        hud_box = QGroupBox("Real-Time Detection & Macro Status")
        hud_layout = QVBoxLayout(hud_box)

        status_row = QHBoxLayout()
        self.pose_label = QLabel("Pose: Ready - Show Upper Body / Hand")
        pose_font = QFont("Segoe UI", 12)
        pose_font.setBold(True)
        self.pose_label.setFont(pose_font)
        self.conf_label = QLabel("Conf: --%")
        status_row.addWidget(self.pose_label)
        status_row.addStretch()
        status_row.addWidget(self.conf_label)
        hud_layout.addLayout(status_row)

        self.charge_bar = QProgressBar()
        self.charge_bar.setRange(0, 100)
        self.charge_bar.setValue(0)
        self.charge_bar.setTextVisible(False)
        self.charge_bar.setFixedHeight(8)
        hud_layout.addWidget(self.charge_bar)

        self.trigger_banner = QLabel("Last Triggered: None")
        self.trigger_banner.setStyleSheet("color: #00FF88; font-weight: bold;")
        hud_layout.addWidget(self.trigger_banner)

        layout.addWidget(hud_box)
        return widget

    def _create_right_panel(self) -> QWidget:
        tabs = QTabWidget()

        # Tab 1: Dataset & Training
        train_tab = QWidget()
        train_layout = QVBoxLayout(train_tab)
        train_layout.setContentsMargins(10, 10, 10, 10)
        train_layout.setSpacing(12)

        # Dataset list table
        table_group = QGroupBox("Gesture Library & Dataset")
        table_layout = QVBoxLayout(table_group)
        self.dataset_table = QTableWidget()
        self.dataset_table.setColumnCount(3)
        self.dataset_table.setHorizontalHeaderLabels(["Gesture", "Samples", "Action"])
        self.dataset_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_layout.addWidget(self.dataset_table)
        train_layout.addWidget(table_group, stretch=1)

        # Record New Gesture Section
        rec_group = QGroupBox("Record New Gesture")
        rec_layout = QGridLayout(rec_group)

        rec_layout.addWidget(QLabel("Gesture Name:"), 0, 0)
        self.rec_name_input = QLineEdit()
        self.rec_name_input.setPlaceholderText("e.g. arms_crossed, fist, peace")
        rec_layout.addWidget(self.rec_name_input, 0, 1)

        rec_layout.addWidget(QLabel("Samples:"), 1, 0)
        self.rec_samples_spin = QSpinBox()
        self.rec_samples_spin.setRange(20, 500)
        self.rec_samples_spin.setValue(100)
        rec_layout.addWidget(self.rec_samples_spin, 1, 1)

        self.record_btn = QPushButton("Record (3-2-1 Countdown)")
        self.record_btn.setObjectName("recordBtn")
        self.record_btn.clicked.connect(self._on_record_clicked)
        rec_layout.addWidget(self.record_btn, 2, 0, 1, 2)

        train_layout.addWidget(rec_group)

        # Retrain Section
        train_btn_box = QHBoxLayout()
        self.train_btn = QPushButton("Retrain Model")
        self.train_btn.setObjectName("trainBtn")
        self.train_btn.clicked.connect(self._on_retrain_clicked)
        train_btn_box.addWidget(self.train_btn)

        self.accuracy_badge = QLabel("Model Status: Ready")
        self.accuracy_badge.setStyleSheet("color: #AAAAAA;")
        train_btn_box.addWidget(self.accuracy_badge)
        train_layout.addLayout(train_btn_box)

        # Record Dynamic Motion Gestures Section
        motion_rec_group = QGroupBox("Train Distinct Dynamic Hand Motions (Swipes, Waves, Custom)")
        motion_rec_layout = QGridLayout(motion_rec_group)

        motion_rec_layout.addWidget(QLabel("Motion Name:"), 0, 0)
        self.rec_motion_name_input = QLineEdit()
        self.rec_motion_name_input.setPlaceholderText("e.g. swipe_right, wave, custom_flick")
        motion_rec_layout.addWidget(self.rec_motion_name_input, 0, 1)

        motion_rec_layout.addWidget(QLabel("Strokes to Record:"), 1, 0)
        self.rec_motion_strokes_spin = QSpinBox()
        self.rec_motion_strokes_spin.setRange(3, 30)
        self.rec_motion_strokes_spin.setValue(8)
        motion_rec_layout.addWidget(self.rec_motion_strokes_spin, 1, 1)

        motion_btn_row = QHBoxLayout()
        self.record_motion_btn = QPushButton("▶ Record Motion Strokes (Hand-Reset Gate)")
        self.record_motion_btn.setObjectName("recordMotionBtn")
        self.record_motion_btn.clicked.connect(self._on_record_motion_clicked)
        motion_btn_row.addWidget(self.record_motion_btn)

        self.train_motion_btn = QPushButton("⚡ Train Motion Model")
        self.train_motion_btn.setObjectName("trainMotionBtn")
        self.train_motion_btn.clicked.connect(self._on_train_motion_clicked)
        motion_btn_row.addWidget(self.train_motion_btn)
        motion_rec_layout.addLayout(motion_btn_row, 2, 0, 1, 2)

        self.motion_model_badge = QLabel("Motion Classifier: Ready")
        self.motion_model_badge.setStyleSheet("color: #00E5FF; font-size: 11px;")
        motion_rec_layout.addWidget(self.motion_model_badge, 3, 0, 1, 2)

        train_layout.addWidget(motion_rec_group)

        tabs.addTab(train_tab, "Gestures & Training")

        # Tab 2: Macro Bindings
        macro_tab = QWidget()
        macro_layout = QVBoxLayout(macro_tab)
        macro_layout.setContentsMargins(10, 10, 10, 10)
        macro_layout.setSpacing(12)

        macro_box = QGroupBox("Configured Macro Actions")
        mb_layout = QVBoxLayout(macro_box)
        self.macro_table = QTableWidget()
        self.macro_table.setColumnCount(3)
        self.macro_table.setHorizontalHeaderLabels(["Gesture", "Action Type", "Keys / Command"])
        self.macro_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        mb_layout.addWidget(self.macro_table)
        macro_layout.addWidget(macro_box, stretch=1)

        # Quick Add / Edit Action
        edit_box = QGroupBox("Add or Edit Macro Binding")
        eb_layout = QGridLayout(edit_box)

        eb_layout.addWidget(QLabel("Target Gesture:"), 0, 0)
        self.macro_gesture_input = QLineEdit()
        self.macro_gesture_input.setPlaceholderText("e.g. fist, swipe_right, swipe_left, circle_cw")
        eb_layout.addWidget(self.macro_gesture_input, 0, 1)

        motion_hint = QLabel("Dynamic gestures: swipe_left, swipe_right, swipe_up, swipe_down, circle_cw, circle_ccw")
        motion_hint.setStyleSheet("color: #00D2FF; font-size: 11px;")
        eb_layout.addWidget(motion_hint, 1, 0, 1, 2)

        eb_layout.addWidget(QLabel("Type:"), 2, 0)
        self.macro_type_combo = QComboBox()
        self.macro_type_combo.addItems(["hotkey", "media", "scroll", "command"])
        eb_layout.addWidget(self.macro_type_combo, 2, 1)

        eb_layout.addWidget(QLabel("Keys / Command:"), 3, 0)
        self.macro_keys_input = QLineEdit()
        self.macro_keys_input.setPlaceholderText("e.g. alt+tab, 300 (scroll), or playpause")
        eb_layout.addWidget(self.macro_keys_input, 3, 1)

        eb_layout.addWidget(QLabel("Description:"), 4, 0)
        self.macro_desc_input = QLineEdit()
        self.macro_desc_input.setPlaceholderText("e.g. Mute Microphone")
        eb_layout.addWidget(self.macro_desc_input, 4, 1)

        self.save_macro_btn = QPushButton("Save / Update Binding")
        self.save_macro_btn.setObjectName("saveMacroBtn")
        self.save_macro_btn.clicked.connect(self._on_save_macro_clicked)
        eb_layout.addWidget(self.save_macro_btn, 5, 0, 1, 2)

        macro_layout.addWidget(edit_box)
        tabs.addTab(macro_tab, "Macro Mappings")

        # Tab 3: Motion & Swipe Calibration
        calib_tab = self._create_motion_calibration_tab()
        tabs.addTab(calib_tab, "Motion Calibration")

        self._refresh_macro_table()
        return tabs

    # Slots
    @Slot(QImage)
    def _on_frame_ready(self, q_img: QImage):
        w = self.video_label.width()
        h = self.video_label.height()
        if w < 100 or h < 100:
            w, h = 640, 360
        scaled = q_img.scaled(
            w, h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(QPixmap.fromImage(scaled))

    @Slot(str, float, float)
    def _on_prediction_ready(self, gesture: str, confidence: float, debounce_prog: float):
        if gesture:
            self.pose_label.setText(f"Pose: {gesture.upper()}")
            self.conf_label.setText(f"Conf: {confidence * 100:.0f}%")
        else:
            self.pose_label.setText("Pose: Ready - Show Upper Body / Hand")
            self.conf_label.setText("Conf: --%")

        self.charge_bar.setValue(int(debounce_prog * 100))

    @Slot(str, str)
    def _on_macro_triggered(self, gesture: str, description: str):
        self.trigger_banner.setText(f"TRIGGERED: {description.upper()} ({gesture})")
        QTimer.singleShot(2500, lambda: self.trigger_banner.setText("Last Triggered: Idle"))

    @Slot(str)
    def _on_status_updated(self, msg: str):
        self.accuracy_badge.setText(msg)
        self._refresh_dataset_table()

    def _on_camera_changed(self, idx: int):
        self.worker.set_camera_index(idx)

    def _toggle_listening(self):
        active = self.listen_btn.isChecked()
        self.worker.listening = active
        if active:
            self.listen_btn.setText("● Listening Active")
            self.listen_btn.setStyleSheet("background-color: #008844; color: white;")
        else:
            self.listen_btn.setText("○ Paused")
            self.listen_btn.setStyleSheet("background-color: #555555; color: #CCCCCC;")

    def _on_record_clicked(self):
        name = self.rec_name_input.text().strip().lower()
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a name for the gesture.")
            return

        samples = self.rec_samples_spin.value()
        self.worker.start_recording(name, samples)

    def _on_retrain_clicked(self):
        self.accuracy_badge.setText("Training model...")
        QApplication.processEvents()

        from src.train import train_gesture_classifier
        res = train_gesture_classifier()
        if res.get("success"):
            acc = res.get("accuracy", 0.0) * 100
            classes = res.get("classes", [])
            self.accuracy_badge.setText(f"Model Accuracy: {acc:.1f}% ({len(classes)} gestures)")
            self.worker.reload_model()
            QMessageBox.information(
                self,
                "Training Complete",
                f"Successfully trained gesture model!\n\nOverall Test Accuracy: {acc:.1f}%\nClasses: {', '.join(classes)}",
            )
        else:
            err = res.get("error", "Unknown error")
            self.accuracy_badge.setText("Training failed")
            QMessageBox.warning(self, "Training Notice", err)

        self._refresh_dataset_table()

    @Slot(str, int, int)
    def _on_motion_recording_progress(self, gesture: str, count: int, target: int):
        self.motion_model_badge.setText(f"Recording '{gesture}': {count}/{target} strokes captured")

    @Slot(str, int)
    def _on_motion_recording_finished(self, gesture: str, total: int):
        self.motion_model_badge.setText(f"Recorded {total} strokes for '{gesture}'")
        self.record_motion_btn.setEnabled(True)
        self._refresh_dataset_table()
        QMessageBox.information(
            self,
            "Motion Recording Complete",
            f"Successfully recorded {total} strokes for '{gesture}'!\n\nClick 'Train Motion Model' to update the classifier.",
        )

    def _on_record_motion_clicked(self):
        name = self.rec_motion_name_input.text().strip().lower()
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a name for the dynamic motion gesture.")
            return

        strokes = self.rec_motion_strokes_spin.value()
        self.record_motion_btn.setEnabled(False)
        self.motion_model_badge.setText(f"Starting recording for '{name}'... follow video banner")
        self.worker.start_motion_recording(name, strokes)

    def _on_train_motion_clicked(self):
        self.motion_model_badge.setText("Training motion classifier...")
        QApplication.processEvents()

        try:
            res = train_motion_model()
            acc = res.get("accuracy", 0.0) * 100
            classes = res.get("classes", [])
            self.motion_model_badge.setText(f"Dynamic Model: {acc:.1f}% ({len(classes)} gestures)")
            self.worker.reload_model()
            QMessageBox.information(
                self,
                "Motion Model Trained",
                f"Successfully trained Dynamic Motion Classifier!\n\nAccuracy: {acc:.1f}%\nClasses: {', '.join(classes)}\n\nNow active for real-time gesture triggering!",
            )
        except Exception as e:
            self.motion_model_badge.setText("Training failed")
            QMessageBox.warning(self, "Motion Training Notice", f"Could not train motion model: {e}")

        self._refresh_dataset_table()

    def _refresh_dataset_table(self):
        # 1. Static pose counts
        static_counts = {}
        if GESTURES_CSV_PATH.exists():
            try:
                with open(GESTURES_CSV_PATH, mode="r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    for row in reader:
                        if row:
                            lbl = row[0].strip()
                            static_counts[lbl] = static_counts.get(lbl, 0) + 1
            except Exception:
                static_counts = {}

        # 2. Dynamic motion counts
        motion_counts = {}
        if MOTION_DATA_CSV_PATH.exists():
            try:
                with open(MOTION_DATA_CSV_PATH, mode="r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    for row in reader:
                        if row:
                            lbl = row[0].strip()
                            motion_counts[lbl] = motion_counts.get(lbl, 0) + 1
            except Exception:
                motion_counts = {}

        total_rows = len(static_counts) + len(motion_counts)
        self.dataset_table.setRowCount(total_rows)

        cur_row = 0
        for gesture, count in static_counts.items():
            self.dataset_table.setItem(cur_row, 0, QTableWidgetItem(f"{gesture} (static)"))
            self.dataset_table.setItem(cur_row, 1, QTableWidgetItem(str(count)))
            action_info = self.macros.get(gesture, {})
            desc = action_info.get("description", "Not mapped")
            self.dataset_table.setItem(cur_row, 2, QTableWidgetItem(desc))
            cur_row += 1

        for gesture, count in motion_counts.items():
            self.dataset_table.setItem(cur_row, 0, QTableWidgetItem(f"{gesture} (motion)"))
            self.dataset_table.setItem(cur_row, 1, QTableWidgetItem(f"{count} strokes"))
            action_info = self.macros.get(gesture, {})
            desc = action_info.get("description", "Not mapped")
            self.dataset_table.setItem(cur_row, 2, QTableWidgetItem(desc))
            cur_row += 1

    def _refresh_macro_table(self):
        self.macro_table.setRowCount(len(self.macros))
        for row_idx, (gesture, cfg) in enumerate(self.macros.items()):
            act_type = cfg.get("type", "hotkey")
            keys = cfg.get("keys", [])
            if act_type == "scroll":
                key_str = f"scroll {cfg.get('amount', 300)}"
            else:
                key_str = "+".join(keys) if keys else cfg.get("key", cfg.get("command", ""))
            self.macro_table.setItem(row_idx, 0, QTableWidgetItem(gesture))
            self.macro_table.setItem(row_idx, 1, QTableWidgetItem(act_type))
            self.macro_table.setItem(row_idx, 2, QTableWidgetItem(f"{key_str} ({cfg.get('description', '')})"))

    def _on_save_macro_clicked(self):
        gesture = self.macro_gesture_input.text().strip().lower()
        if not gesture:
            QMessageBox.warning(self, "Missing Gesture", "Please enter a gesture name to map.")
            return

        act_type = self.macro_type_combo.currentText()
        raw_keys = self.macro_keys_input.text().strip()
        desc = self.macro_desc_input.text().strip() or gesture

        new_entry: Dict[str, Any] = {"description": desc, "type": act_type}
        if act_type == "hotkey":
            new_entry["keys"] = [k.strip() for k in raw_keys.split("+")]
        elif act_type == "media":
            new_entry["key"] = raw_keys
        elif act_type == "scroll":
            try:
                new_entry["amount"] = int(raw_keys)
            except ValueError:
                new_entry["amount"] = 300
        elif act_type == "command":
            new_entry["command"] = raw_keys

        self.macros[gesture] = new_entry
        save_macro_actions(self.macros)
        self.worker.macros = self.macros
        self._refresh_macro_table()
        self._refresh_dataset_table()
        QMessageBox.information(self, "Macro Saved", f"Mapped gesture '{gesture}' to {desc}!")

    def _create_motion_calibration_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # Overview Box
        info_box = QGroupBox("Whole-Hand (Palm Centroid) Motion Tracking")
        ib_layout = QVBoxLayout(info_box)
        info_desc = QLabel(
            "Tracking is locked to your whole palm centroid (rather than a finger point). "
            "Whether your fingers are open, curled, or in a fist, the system follows your hand movement. "
            "Calibrate below to match your natural swipe distance and speed!"
        )
        info_desc.setWordWrap(True)
        info_desc.setStyleSheet("color: #CCCCCC; font-size: 11px;")
        ib_layout.addWidget(info_desc)
        layout.addWidget(info_box)

        # Guided Calibration Box
        calib_box = QGroupBox("Guided Swipe Calibration")
        cb_layout = QVBoxLayout(calib_box)
        cb_layout.setSpacing(8)

        self.calib_instruction_lbl = QLabel("Click 'Start Guided Calibration' to begin recording your natural swipes.")
        self.calib_instruction_lbl.setStyleSheet("color: #00E5FF; font-weight: bold;")
        self.calib_instruction_lbl.setWordWrap(True)
        cb_layout.addWidget(self.calib_instruction_lbl)

        # Direction status checklist
        grid = QGridLayout()
        self.calib_labels: Dict[str, QLabel] = {}
        dirs = [
            ("swipe_right", "1. Flick RIGHT (Alt+Tab):", 0),
            ("swipe_left", "2. Flick LEFT (Shift+Alt+Tab):", 1),
        ]
        for key, text, row in dirs:
            grid.addWidget(QLabel(text), row, 0)
            status_lbl = QLabel("Pending (0/3)")
            status_lbl.setStyleSheet("color: #888888;")
            self.calib_labels[key] = status_lbl
            grid.addWidget(status_lbl, row, 1)

        cb_layout.addLayout(grid)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.start_calib_btn = QPushButton("▶ Start Guided Calibration")
        self.start_calib_btn.setObjectName("startCalibBtn")
        self.start_calib_btn.clicked.connect(self._on_start_calibration_clicked)
        btn_layout.addWidget(self.start_calib_btn)

        self.cancel_calib_btn = QPushButton("✖ Cancel")
        self.cancel_calib_btn.setEnabled(False)
        self.cancel_calib_btn.clicked.connect(self._on_cancel_calibration_clicked)
        btn_layout.addWidget(self.cancel_calib_btn)

        cb_layout.addLayout(btn_layout)
        layout.addWidget(calib_box)

        # Profile Parameters & Manual Adjustments Box
        param_box = QGroupBox("Active Motion Profile & Tuning")
        pb_layout = QGridLayout(param_box)
        pb_layout.setSpacing(8)

        pb_layout.addWidget(QLabel("Min Stroke Distance (% of screen):"), 0, 0)
        self.disp_spin = QDoubleSpinBox()
        self.disp_spin.setRange(0.04, 0.35)
        self.disp_spin.setSingleStep(0.005)
        self.disp_spin.setDecimals(3)
        pb_layout.addWidget(self.disp_spin, 0, 1)

        pb_layout.addWidget(QLabel("Min Stroke Speed (vel):"), 1, 0)
        self.vel_spin = QDoubleSpinBox()
        self.vel_spin.setRange(0.08, 0.70)
        self.vel_spin.setSingleStep(0.01)
        self.vel_spin.setDecimals(2)
        pb_layout.addWidget(self.vel_spin, 1, 1)

        pb_layout.addWidget(QLabel("Palm Smoothing Alpha (0=smooth, 1=raw):"), 2, 0)
        self.smooth_spin = QDoubleSpinBox()
        self.smooth_spin.setRange(0.20, 0.95)
        self.smooth_spin.setSingleStep(0.05)
        self.smooth_spin.setDecimals(2)
        pb_layout.addWidget(self.smooth_spin, 2, 1)

        # Buttons
        p_btn_layout = QHBoxLayout()
        self.apply_profile_btn = QPushButton("Save Manual Profile")
        self.apply_profile_btn.clicked.connect(self._on_save_manual_profile_clicked)
        p_btn_layout.addWidget(self.apply_profile_btn)

        self.reset_profile_btn = QPushButton("Reset to Factory Defaults")
        self.reset_profile_btn.clicked.connect(self._on_reset_motion_defaults_clicked)
        p_btn_layout.addWidget(self.reset_profile_btn)

        pb_layout.addLayout(p_btn_layout, 3, 0, 1, 2)
        layout.addWidget(param_box)

        layout.addStretch()
        return widget

    def _refresh_motion_profile_ui(self):
        prof = load_motion_profile()
        self.disp_spin.setValue(float(prof.get("min_displacement", 0.065)))
        self.vel_spin.setValue(float(prof.get("min_velocity", 0.16)))
        self.smooth_spin.setValue(float(prof.get("smoothing_alpha", 0.75)))

    def _on_start_calibration_clicked(self):
        self.start_calib_btn.setEnabled(False)
        self.cancel_calib_btn.setEnabled(True)
        for lbl in self.calib_labels.values():
            lbl.setText("Pending (0/3)")
            lbl.setStyleSheet("color: #888888;")
        self.calib_instruction_lbl.setText("Follow top video banner: Swipe RIGHT 3 times naturally.")
        self.worker.motion_calibrator.start()
        self._on_calibration_progress("swipe_right", 0, 3)

    def _on_cancel_calibration_clicked(self):
        self.worker.motion_calibrator.cancel()
        self.start_calib_btn.setEnabled(True)
        self.cancel_calib_btn.setEnabled(False)
        self.calib_instruction_lbl.setText("Calibration cancelled.")

    @Slot(str, int, int)
    def _on_calibration_progress(self, direction: str, count: int, target: int):
        lbl = self.calib_labels.get(direction)
        if lbl:
            if count >= target:
                lbl.setText("✔ Complete (3/3)")
                lbl.setStyleSheet("color: #00FF88; font-weight: bold;")
            else:
                lbl.setText(f"In Progress ({count}/{target})")
                lbl.setStyleSheet("color: #00E5FF; font-weight: bold;")

        next_dir = self.worker.motion_calibrator.current_direction
        if next_dir:
            dir_name = next_dir.replace("swipe_", "").upper()
            self.calib_instruction_lbl.setText(f"Follow video banner: Swipe {dir_name} 3 times naturally.")

    @Slot(dict)
    def _on_calibration_finished(self, profile: dict):
        self.start_calib_btn.setEnabled(True)
        self.cancel_calib_btn.setEnabled(False)
        disp_pct = profile.get('min_displacement', 0.16) * 100
        vel_val = profile.get('min_velocity', 0.25)
        self.calib_instruction_lbl.setText(
            f"✔ Calibration Complete! Saved: Min Distance {disp_pct:.0f}%, Velocity {vel_val:.2f}"
        )
        self._refresh_motion_profile_ui()
        QMessageBox.information(
            self,
            "Calibration Complete",
            f"Personal motion profile created and applied!\n\n"
            f"• Min Stroke Distance: {disp_pct:.1f}% screen\n"
            f"• Stroke Velocity: {vel_val:.2f}\n\n"
            f"Swipes are now calibrated to your natural hand movements.",
        )

    def _on_save_manual_profile_clicked(self):
        profile = {
            "min_displacement": round(self.disp_spin.value(), 3),
            "min_velocity": round(self.vel_spin.value(), 3),
            "smoothing_alpha": round(self.smooth_spin.value(), 2),
            "min_duration": 0.10,
            "max_duration": 0.85,
            "max_anchor_speed": 0.35,
        }
        save_motion_profile(profile)
        self.worker.motion_tracker.apply_profile(profile)
        QMessageBox.information(self, "Profile Saved", "Saved and applied manual motion profile!")

    def _on_reset_motion_defaults_clicked(self):
        from src.config import (
            SWIPE_MIN_DISPLACEMENT,
            SWIPE_MIN_VELOCITY,
            SWIPE_MIN_DURATION_SECONDS,
            SWIPE_MAX_DURATION_SECONDS,
            MAX_ANCHOR_SPEED,
            SMOOTHING_ALPHA,
            save_motion_profile,
        )
        defaults = {
            "min_displacement": SWIPE_MIN_DISPLACEMENT,
            "min_velocity": SWIPE_MIN_VELOCITY,
            "min_duration": SWIPE_MIN_DURATION_SECONDS,
            "max_duration": SWIPE_MAX_DURATION_SECONDS,
            "max_anchor_speed": MAX_ANCHOR_SPEED,
            "smoothing_alpha": SMOOTHING_ALPHA,
        }
        save_motion_profile(defaults)
        self.worker.motion_tracker.apply_profile(defaults)
        self._refresh_motion_profile_ui()
        QMessageBox.information(self, "Profile Reset", "Reset motion tracking profile to factory defaults.")

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()

    def _apply_dark_theme(self):
        dark_stylesheet = """
        QMainWindow {
            background-color: #1A1A1A;
            color: #FFFFFF;
        }
        QWidget {
            font-family: 'Segoe UI', -apple-system, Roboto, sans-serif;
            font-size: 10pt;
            color: #E0E0E0;
        }
        QGroupBox {
            border: 1px solid #333333;
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 12px;
            font-weight: bold;
            color: #00E5FF;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
        }
        QPushButton {
            background-color: #2D2D2D;
            border: 1px solid #444444;
            border-radius: 4px;
            padding: 7px 14px;
            font-weight: 500;
        }
        QPushButton:hover {
            background-color: #3D3D3D;
            border-color: #00E5FF;
        }
        QPushButton#listenBtn {
            background-color: #008844;
            color: white;
            font-weight: bold;
        }
        QPushButton#recordBtn {
            background-color: #CC4400;
            color: white;
            font-weight: bold;
        }
        QPushButton#recordBtn:hover {
            background-color: #EE5500;
        }
        QPushButton#trainBtn {
            background-color: #0077CC;
            color: white;
            font-weight: bold;
        }
        QPushButton#trainBtn:hover {
            background-color: #0099FF;
        }
        QPushButton#saveMacroBtn {
            background-color: #008844;
            color: white;
            font-weight: bold;
        }
        QLineEdit, QSpinBox, QComboBox {
            background-color: #252525;
            border: 1px solid #3E3E3E;
            border-radius: 4px;
            padding: 5px 8px;
            color: #FFFFFF;
        }
        QTableWidget {
            background-color: #202020;
            border: 1px solid #333333;
            border-radius: 4px;
            gridline-color: #2E2E2E;
            selection-background-color: #005588;
        }
        QHeaderView::section {
            background-color: #282828;
            color: #BBBBBB;
            padding: 5px;
            border: 1px solid #333333;
            font-weight: bold;
        }
        QTabWidget::pane {
            border: 1px solid #333333;
            border-radius: 6px;
            background-color: #1E1E1E;
        }
        QTabBar::tab {
            background-color: #282828;
            padding: 8px 18px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            color: #AAAAAA;
        }
        QTabBar::tab:selected {
            background-color: #1E1E1E;
            color: #00E5FF;
            font-weight: bold;
            border-bottom: 2px solid #00E5FF;
        }
        QProgressBar {
            background-color: #2A2A2A;
            border-radius: 4px;
        }
        QProgressBar::chunk {
            background-color: #00E5FF;
            border-radius: 4px;
        }
        """
        self.setStyleSheet(dark_stylesheet)


def main():
    app = QApplication(sys.argv)
    window = GestureStudioWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

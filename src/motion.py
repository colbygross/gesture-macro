"""
Dynamic Motion Gesture Tracker.
Analyzes temporal trajectories of hand landmarks (wrist / index fingertip) to detect:
- Swipe Left / Swipe Right
- Swipe Up / Swipe Down
- Circle Clockwise / Circle Counter-Clockwise
Provides visual fading comet trail rendering on OpenCV frames.
Cross-platform support for Windows and macOS.
"""

import time
import math
from collections import deque
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2

from src.config import (
    MOTION_BUFFER_FRAMES,
    SWIPE_MIN_DISPLACEMENT,
    SWIPE_MIN_VELOCITY,
    SWIPE_MIN_DURATION_SECONDS,
    SWIPE_MAX_DURATION_SECONDS,
    MOTION_COOLDOWN_SECONDS,
    REQUIRE_STARTING_ANCHOR,
    MAX_ANCHOR_SPEED,
    ENABLE_CIRCLES,
    ENABLE_VERTICAL_SWIPES,
    SMOOTHING_ALPHA,
    load_motion_profile,
    save_motion_profile,
)
from src.rig_tracker import KineticRigTracker


class MotionTracker:
    """
    Real-time kinematic trajectory tracker for dynamic hand gestures.
    Supports palm centroid EMA smoothing, natural stroke thresholds,
    starting position rest anchors, and personal profile calibration.
    """

    def __init__(
        self,
        buffer_size: int = MOTION_BUFFER_FRAMES,
        min_displacement: float = SWIPE_MIN_DISPLACEMENT,
        min_velocity: float = SWIPE_MIN_VELOCITY,
        cooldown_seconds: float = MOTION_COOLDOWN_SECONDS,
        min_duration_seconds: float = SWIPE_MIN_DURATION_SECONDS,
        max_duration_seconds: float = SWIPE_MAX_DURATION_SECONDS,
        require_starting_anchor: bool = REQUIRE_STARTING_ANCHOR,
        max_anchor_speed: float = MAX_ANCHOR_SPEED,
        enable_circles: bool = ENABLE_CIRCLES,
        enable_vertical_swipes: bool = ENABLE_VERTICAL_SWIPES,
        smoothing_alpha: float = SMOOTHING_ALPHA,
    ):
        self.buffer_size = buffer_size
        self.min_displacement = min_displacement
        self.min_velocity = min_velocity
        self.cooldown_seconds = cooldown_seconds
        self.min_duration_seconds = min_duration_seconds
        self.max_duration_seconds = max_duration_seconds
        self.require_starting_anchor = require_starting_anchor
        self.max_anchor_speed = max_anchor_speed
        self.enable_circles = enable_circles
        self.enable_vertical_swipes = enable_vertical_swipes
        self.smoothing_alpha = smoothing_alpha

        # Deque of (x, y, timestamp)
        self.trajectory: deque = deque(maxlen=buffer_size)
        self.last_trigger_time = -100.0
        self.last_triggered_gesture = ""

        # EMA Smoothed position
        self.smoothed_pos: Optional[Tuple[float, float]] = None

        # Starting Anchor State
        self.is_anchored = False
        self.anchor_position: Optional[Tuple[float, float]] = None

        # Fast-Flick Start-to-End Tracking
        self.active_start_pos: Optional[Tuple[float, float]] = None
        self.active_start_time: float = 0.0
        self.peak_pos: Optional[Tuple[float, float]] = None
        self.peak_disp: float = 0.0
        self.peak_time: float = 0.0

        # Tracking Coasting & Persistence for High-Speed Movements
        self.missed_frames = 0
        self.max_coast_frames = 6
        self.last_velocity = (0.0, 0.0)

        # Kinetic Rig Tracker (4D Kalman Filter & Optical Motion Field Fusion)
        self.kinetic_rig = KineticRigTracker()
        self.current_speed = 0.0
        self.current_velocity = (0.0, 0.0)

        # Optional ML Motion Model for Distinct Hand Motions
        self.motion_model: Optional[Any] = None
        self.model_confidence_threshold: float = 0.70

        # Return-to-Center & Reversal Suppression State
        self.return_pending = False
        self.last_trigger_dir: Optional[str] = None  # "right", "left", "up", "down"
        self.consecutive_steady_frames = 0
        self.min_steady_frames_for_reset = 4

    def set_motion_model(self, model: Any, confidence_threshold: float = 0.70) -> None:
        """Sets an optional trained MotionModel for dynamic motion gesture inference."""
        self.motion_model = model
        self.model_confidence_threshold = confidence_threshold

    def apply_profile(self, profile: Dict[str, Any]) -> None:
        """Applies calibrated or configured motion parameters."""
        if "min_displacement" in profile:
            self.min_displacement = float(profile["min_displacement"])
        if "min_velocity" in profile:
            self.min_velocity = float(profile["min_velocity"])
        if "min_duration" in profile:
            self.min_duration_seconds = float(profile["min_duration"])
        if "max_duration" in profile:
            self.max_duration_seconds = float(profile["max_duration"])
        if "max_anchor_speed" in profile:
            self.max_anchor_speed = float(profile["max_anchor_speed"])
        if "smoothing_alpha" in profile:
            self.smoothing_alpha = float(profile["smoothing_alpha"])

    def load_calibrated_profile(self) -> None:
        """Loads and applies profile from disk if present."""
        profile = load_motion_profile()
        self.apply_profile(profile)

    def reset(self):
        """Clears trajectory buffer, smoothed position, and resets starting anchor."""
        self.trajectory.clear()
        self.smoothed_pos = None
        self.is_anchored = False
        self.anchor_position = None
        self.active_start_pos = None
        self.peak_pos = None
        self.peak_disp = 0.0
        self.peak_time = 0.0
        self.missed_frames = 0
        self.last_velocity = (0.0, 0.0)
        self.current_speed = 0.0
        self.current_velocity = (0.0, 0.0)
        self.return_pending = False
        self.last_trigger_dir = None
        self.consecutive_steady_frames = 0
        if hasattr(self, "kinetic_rig"):
            self.kinetic_rig.reset()

    def handle_tracking_loss(
        self,
        timestamp: Optional[float] = None,
        frame_bgr: Optional[np.ndarray] = None,
        face_center: Optional[Tuple[float, float]] = None,
    ) -> Optional[str]:
        """
        Maintains tracking persistence across momentary frame dropouts caused by high-speed
        hand movement or motion blur using 4D Kalman filtering and optical motion field estimation.
        """
        now = timestamp if timestamp is not None else time.time()
        self.missed_frames += 1

        if self.missed_frames <= self.max_coast_frames:
            # Check cooldown
            if (now - self.last_trigger_time) < self.cooldown_seconds:
                return None

            # Attempt predictive kinetic coasting across blur
            if hasattr(self, "kinetic_rig"):
                k_pos, k_vel, k_speed = self.kinetic_rig.update(None, frame_bgr=frame_bgr, timestamp=now)
                if k_pos is not None:
                    return self.update(k_pos[0], k_pos[1], timestamp=now, frame_bgr=frame_bgr, face_center=face_center, _coasted=True)

            # If a fast flick had sufficient displacement right before dropout, check if it completed
            if self.active_start_pos is not None and self.peak_pos is not None and self.peak_disp >= self.min_displacement:
                if face_center is not None:
                    if math.hypot(self.peak_pos[0] - face_center[0], self.peak_pos[1] - face_center[1]) < 0.12:
                        return None
                    if math.hypot(self.active_start_pos[0] - face_center[0], self.active_start_pos[1] - face_center[1]) < 0.12:
                        return None

                # 0. Check custom ML model first
                if self.motion_model is not None and getattr(self.motion_model, "is_loaded", False) and len(self.trajectory) >= 3:
                    pred_label, conf = self.motion_model.predict(list(self.trajectory))
                    if pred_label and conf >= self.model_confidence_threshold:
                        self.last_trigger_time = now
                        self.last_triggered_gesture = pred_label
                        self.reset()
                        return pred_label

                # 1. Start-to-end swipe evaluation
                fast_swipe = self._detect_start_to_end_swipe(
                    now,
                    cur_x=self.peak_pos[0],
                    cur_y=self.peak_pos[1],
                    cur_speed=0.0
                )
                if fast_swipe:
                    self.last_trigger_time = now
                    self.last_triggered_gesture = fast_swipe
                    self.reset()
                    return fast_swipe
            return None
        else:
            # Tracking truly lost for > max_coast_frames
            self.reset()
            return None

    def update(
        self,
        x: float,
        y: float,
        timestamp: Optional[float] = None,
        frame_bgr: Optional[np.ndarray] = None,
        face_center: Optional[Tuple[float, float]] = None,
        _coasted: bool = False,
    ) -> Optional[str]:
        """
        Feeds a normalized landmark coordinate (0.0 to 1.0) into the tracker.
        Applies Kinetic Rig 4D Kalman filtering to track true velocity and eliminate jitter.

        Returns:
            Detected motion gesture name (e.g. 'swipe_right', 'swipe_left', 'circle_cw'),
            or None if no motion gesture was triggered.
        """
        now = timestamp if timestamp is not None else time.time()

        if not _coasted and hasattr(self, "kinetic_rig"):
            k_pos, k_vel, k_speed = self.kinetic_rig.update((x, y), frame_bgr=frame_bgr, timestamp=now)
            if k_pos is not None:
                cur_x, cur_y = k_pos
            else:
                cur_x, cur_y = x, y
            self.current_speed = k_speed
            self.current_velocity = k_vel
            self.missed_frames = 0
        else:
            cur_x, cur_y = x, y

        # Maintain EMA smoothed position
        if self.smoothing_alpha < 1.0:
            if self.smoothed_pos is None:
                self.smoothed_pos = (x, y)
            else:
                sx = self.smoothing_alpha * x + (1.0 - self.smoothing_alpha) * self.smoothed_pos[0]
                sy = self.smoothing_alpha * y + (1.0 - self.smoothing_alpha) * self.smoothed_pos[1]
                self.smoothed_pos = (sx, sy)
        else:
            self.smoothed_pos = (x, y)

        # Track velocity internally
        if len(self.trajectory) >= 1:
            prev_x, prev_y, prev_t = self.trajectory[-1]
            dt = max(now - prev_t, 1e-4)
            inst_vx = (cur_x - prev_x) / dt
            inst_vy = (cur_y - prev_y) / dt
            alpha_v = 0.5
            self.last_velocity = (
                alpha_v * inst_vx + (1.0 - alpha_v) * self.last_velocity[0],
                alpha_v * inst_vy + (1.0 - alpha_v) * self.last_velocity[1],
            )

        # Immediate Cooldown Lockout:
        # During cooldown, the hand is finishing follow-through or returning to center.
        # We MUST completely suppress trajectory buildup and anchor acquisition during cooldown,
        # otherwise returning the hand to center will lock an anchor or arm a false return gesture.
        if (now - self.last_trigger_time) < self.cooldown_seconds:
            self.trajectory.clear()
            self.is_anchored = False
            self.anchor_position = None
            self.active_start_pos = None
            self.peak_pos = None
            self.peak_disp = 0.0
            return None

        self.trajectory.append((cur_x, cur_y, now))

        # Check speed for steady rest / anchor readiness
        cur_speed = 0.0
        if len(self.trajectory) >= 2:
            recent_pts = list(self.trajectory)[-4:]
            speeds = [
                math.hypot(recent_pts[k][0] - recent_pts[k - 1][0], recent_pts[k][1] - recent_pts[k - 1][1])
                / max(recent_pts[k][2] - recent_pts[k - 1][2], 1e-4)
                for k in range(1, len(recent_pts))
            ]
            cur_speed = speeds[-1] if speeds else 0.0
            is_steady = max(speeds) <= self.max_anchor_speed
        else:
            is_steady = False

        # Return-to-Center Settlement:
        # After a gesture, hand must return to neutral and remain steady for min_steady_frames
        # before return_pending is cleared and a new gesture can arm.
        if is_steady:
            self.consecutive_steady_frames += 1
            if self.consecutive_steady_frames >= self.min_steady_frames_for_reset:
                self.return_pending = False
                self.last_trigger_dir = None
        else:
            self.consecutive_steady_frames = 0

        # Update starting anchor readiness (only when NOT in pending return-to-center state)
        if is_steady and len(self.trajectory) >= 3 and not self.return_pending:
            # If resting or idling near the face/chin, do NOT set starting anchor
            is_near_face = False
            if face_center is not None:
                if math.hypot(cur_x - face_center[0], cur_y - face_center[1]) < 0.14:
                    is_near_face = True

            if not is_near_face:
                self.is_anchored = True
                self.anchor_position = (cur_x, cur_y)
                # When steady at rest, lock this as the active starting position
                self.active_start_pos = (cur_x, cur_y)
                self.active_start_time = now
                self.peak_pos = (cur_x, cur_y)
                self.peak_disp = 0.0
                self.peak_time = now
            else:
                self.is_anchored = False
                self.anchor_position = None
                self.active_start_pos = None
        elif self.is_anchored and self.anchor_position is not None:
            disp_from_anchor = math.hypot(cur_x - self.anchor_position[0], cur_y - self.anchor_position[1])
            if cur_speed > self.max_anchor_speed or disp_from_anchor > 0.04:
                self.is_anchored = False
        else:
            self.is_anchored = False

        # Track displacement and peak point since leaving start anchor
        if self.active_start_pos is not None:
            dt_from_start = now - self.active_start_time
            if dt_from_start > self.max_duration_seconds:
                self.active_start_pos = None
            else:
                disp_from_start = math.hypot(
                    cur_x - self.active_start_pos[0],
                    cur_y - self.active_start_pos[1]
                )
                if disp_from_start > self.peak_disp:
                    self.peak_disp = disp_from_start
                    self.peak_pos = (cur_x, cur_y)
                    self.peak_time = now

        # Face zone immunity:
        # If the hand is currently resting on or directly in front of the face, or gesture started from the face,
        # suppress macro triggers to avoid face touching / resting cheek on hand triggering swipes.
        if face_center is not None:
            dist_to_face = math.hypot(cur_x - face_center[0], cur_y - face_center[1])
            if dist_to_face < 0.12:
                return None
            if self.active_start_pos is not None:
                if math.hypot(self.active_start_pos[0] - face_center[0], self.active_start_pos[1] - face_center[1]) < 0.12:
                    return None

        # 0. Evaluate Custom Machine Learning Motion Model if trained
        if self.motion_model is not None and getattr(self.motion_model, "is_loaded", False) and len(self.trajectory) >= 2:
            if self.peak_disp >= self.min_displacement:
                cur_disp = math.hypot(cur_x - self.active_start_pos[0], cur_y - self.active_start_pos[1]) if self.active_start_pos else 0.0
                stroke_landed = (
                    cur_disp <= (self.peak_disp - 0.008)
                    or (now - self.peak_time) >= 0.020
                    or cur_speed <= (self.max_anchor_speed * 1.5)
                )
                if stroke_landed:
                    start_t = self.active_start_time if self.active_start_time else (now - 0.5)
                    stroke_pts = [p for p in self.trajectory if p[2] >= (start_t - 0.05)]
                    if len(stroke_pts) >= 2:
                        pred_label, conf = self.motion_model.predict(stroke_pts, min_disp=self.min_displacement)
                        if pred_label and conf >= self.model_confidence_threshold:
                            self.last_trigger_time = now
                            self.last_triggered_gesture = pred_label
                            self.last_trigger_dir = "right" if "right" in pred_label else ("left" if "left" in pred_label else None)
                            self.return_pending = True
                            self.consecutive_steady_frames = 0
                            self.reset()
                            self.last_trigger_time = now
                            self.last_triggered_gesture = pred_label
                            self.last_trigger_dir = "right" if "right" in pred_label else ("left" if "left" in pred_label else None)
                            self.return_pending = True
                            return pred_label

        # 1. Evaluate Fast Start-to-End Position Swipes (Instant Anchor-to-Land)
        fast_swipe = self._detect_start_to_end_swipe(now, cur_x, cur_y, cur_speed)
        if fast_swipe:
            self.last_trigger_time = now
            self.last_triggered_gesture = fast_swipe
            self.last_trigger_dir = "right" if "right" in fast_swipe else ("left" if "left" in fast_swipe else None)
            self.return_pending = True
            self.consecutive_steady_frames = 0
            self.trajectory.clear()
            self.is_anchored = False
            self.anchor_position = None
            self.active_start_pos = None
            self.peak_pos = None
            self.peak_disp = 0.0
            return fast_swipe


        # 2. Evaluate Continuous Trajectory Swipes (as few as 3 frames)
        if len(self.trajectory) >= 3:
            swipe_gesture = self._detect_swipe(now, face_center=face_center)
            if swipe_gesture:
                self.last_trigger_time = now
                self.last_triggered_gesture = swipe_gesture
                self.last_trigger_dir = "right" if "right" in swipe_gesture else ("left" if "left" in swipe_gesture else None)
                self.return_pending = True
                self.consecutive_steady_frames = 0
                self.trajectory.clear()
                self.is_anchored = False
                self.anchor_position = None
                self.active_start_pos = None
                self.peak_pos = None
                self.peak_disp = 0.0
                return swipe_gesture

        # 3. Evaluate Circular Gestures (only if explicitly enabled)
        if self.enable_circles and len(self.trajectory) >= 6:
            circle_gesture = self._detect_circle(now)
            if circle_gesture:
                self.last_trigger_time = now
                self.last_triggered_gesture = circle_gesture
                self.trajectory.clear()
                self.is_anchored = False
                self.anchor_position = None
                self.active_start_pos = None
                self.peak_pos = None
                self.peak_disp = 0.0
                return circle_gesture

        return None

    def _detect_start_to_end_swipe(
        self, now: float, cur_x: float, cur_y: float, cur_speed: float
    ) -> Optional[str]:
        """
        Detects fast swipes by directly comparing Start Position to Landing/End Position.
        Eliminates intermediate duration and path distortion filters for rapid hand flicks.
        """
        if self.active_start_pos is None or self.peak_pos is None:
            return None

        dt = now - self.active_start_time
        if dt < 0.015:
            return None
        if dt > self.max_duration_seconds:
            return None

        # Must have reached minimum displacement
        if self.peak_disp < self.min_displacement:
            return None

        # Velocity check: prevents false positives on slow casual movements
        dt_peak = max(self.peak_time - self.active_start_time, 1e-4)
        stroke_velocity = self.peak_disp / dt_peak
        if stroke_velocity < self.min_velocity:
            return None

        # End position landing criteria for wrist flicks:
        # 1) Hand displacement peaked and pulled back slightly (reversal)
        # 2) Or hand lingered at landing point for >= 20ms
        # 3) Or hand has decelerated at landing point
        cur_disp = math.hypot(cur_x - self.active_start_pos[0], cur_y - self.active_start_pos[1])
        landed = (
            cur_disp <= (self.peak_disp - 0.010)
            or (now - self.peak_time) >= 0.020
            or cur_speed <= (self.max_anchor_speed * 1.5)
        )

        if not landed:
            return None

        # Compare Start Position to End Position
        start_x, start_y = self.active_start_pos
        end_x, end_y = self.peak_pos
        dx = end_x - start_x
        dy = end_y - start_y
        dist_x = abs(dx)
        dist_y = abs(dy)

        # Horizontal Swipes (allows natural human arm/wrist arc: 1.08x)
        if dist_x >= self.min_displacement and dist_x >= 1.08 * dist_y:
            # Reversal / Return-to-Center Filter:
            # If a flick just fired in one direction (e.g. swipe_right), moving back in the opposite direction
            # (e.g. swipe_left) during return-to-center is ALWAYS suppressed.
            if self.last_trigger_dir == "right" and dx < 0:
                return None
            if self.last_trigger_dir == "left" and dx > 0:
                return None
            return "swipe_right" if dx > 0 else "swipe_left"

        # Vertical Swipes (disabled by default to focus exclusively on left/right flicks)
        if self.enable_vertical_swipes:
            if dist_y >= self.min_displacement and dist_y >= 1.08 * dist_x:
                if self.last_trigger_dir == "down" and dy < 0:
                    return None
                if self.last_trigger_dir == "up" and dy > 0:
                    return None
                return "swipe_up" if dy < 0 else "swipe_down"

        return None

    def _detect_swipe(self, current_time: float, face_center: Optional[Tuple[float, float]] = None) -> Optional[str]:
        """
        Detects directional movement across recent frames.
        """
        # Search backwards within configured duration window
        pts = list(self.trajectory)
        latest_x, latest_y, latest_t = pts[-1]

        if face_center is not None and math.hypot(latest_x - face_center[0], latest_y - face_center[1]) < 0.14:
            return None

        for i in range(len(pts) - 2, -1, -1):
            past_x, past_y, past_t = pts[i]
            if face_center is not None and math.hypot(past_x - face_center[0], past_y - face_center[1]) < 0.14:
                continue
            dt = latest_t - past_t
            if dt < 0.035:
                continue
            if dt > self.max_duration_seconds:
                break

            dx = latest_x - past_x
            dy = latest_y - past_y
            D = math.hypot(dx, dy)
            if D < self.min_displacement:
                continue

            dist_x = abs(dx)
            dist_y = abs(dy)
            vx = dist_x / dt
            vy = dist_y / dt

            # 1. Starting Position / Rest Anchor Check
            if self.require_starting_anchor:
                pre_pts = pts[max(0, i - 4) : i + 1]
                if len(pre_pts) < 2:
                    continue  # Stroke did not originate from a steady starting position
                pre_speeds = [
                    math.hypot(pre_pts[k][0] - pre_pts[k - 1][0], pre_pts[k][1] - pre_pts[k - 1][1])
                    / max(pre_pts[k][2] - pre_pts[k - 1][2], 1e-4)
                    for k in range(1, len(pre_pts))
                ]
                avg_pre_speed = sum(pre_speeds) / len(pre_speeds)
                if avg_pre_speed > (self.max_anchor_speed * 1.5):
                    continue  # Hand was moving too fast prior to stroke
            elif i > 0:
                prev_x, prev_y, prev_t = pts[i - 1]
                dt_pre = past_t - prev_t
                if dt_pre > 1e-4:
                    dx_pre = past_x - prev_x
                    dy_pre = past_y - prev_y
                    speed_pre = math.hypot(dx_pre, dy_pre) / dt_pre
                    if speed_pre > 0.25:
                        cos_sim = (dx_pre * dx + dy_pre * dy) / (math.hypot(dx_pre, dy_pre) * D)
                        if cos_sim < 0.70:
                            continue

            # Straightness check: allows natural human arm arcs (up to 35% perpendicular deviation)
            sub_pts = pts[i:]
            max_dev = 0.0
            for p in sub_pts:
                dev = abs(dy * p[0] - dx * p[1] + latest_x * past_y - latest_y * past_x) / D
                if dev > max_dev:
                    max_dev = dev
            if (max_dev / D) > 0.35:
                continue

            # Horizontal Swipes (natural arm arc: dx dominates dy by 1.08x)
            if dist_x >= self.min_displacement and vx >= self.min_velocity:
                if dist_x >= 1.08 * dist_y:
                    if self.last_trigger_dir == "right" and dx < 0:
                        continue
                    if self.last_trigger_dir == "left" and dx > 0:
                        continue
                    return "swipe_right" if dx > 0 else "swipe_left"

            # Vertical Swipes (disabled by default to focus exclusively on left/right flicks)
            if self.enable_vertical_swipes and dist_y >= self.min_displacement and vy >= self.min_velocity:
                if dist_y >= 1.08 * dist_x:
                    if dy < 0:
                        if self.last_trigger_dir == "down":
                            continue
                        return "swipe_up"
                    else:
                        if self.last_trigger_dir == "up":
                            continue
                        return "swipe_down"

        return None

    def _detect_circle(self, current_time: float) -> Optional[str]:
        """
        Detects circular motion by computing cumulative angular displacement
        around the trajectory centroid.
        """
        if len(self.trajectory) < 14:
            return None

        pts = list(self.trajectory)
        dt = pts[-1][2] - pts[0][2]
        if dt > 1.4 or dt < 0.25:
            return None

        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])

        # Centroid
        cx = np.mean(xs)
        cy = np.mean(ys)

        # Variance check (circles should span both X and Y dimensions reasonably)
        span_x = np.ptp(xs)
        span_y = np.ptp(ys)
        if span_x < 0.10 or span_y < 0.10:
            return None

        aspect = span_x / max(span_y, 1e-4)
        if aspect < 0.4 or aspect > 2.5:
            return None

        # Cumulative signed angle
        angles = np.arctan2(ys - cy, xs - cx)
        angle_diffs = np.diff(angles)
        # Unwrap differences to [-pi, pi]
        angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
        total_angle = float(np.sum(angle_diffs))

        # Full circle is 2*pi (~6.28 rad). Require ~4.5 rad (approx 260 degrees)
        if total_angle >= 4.5:
            return "circle_cw"
        elif total_angle <= -4.5:
            return "circle_ccw"

        return None

    def draw_trail(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 230, 255),
        max_thickness: int = 5,
    ) -> None:
        """
        Renders a glowing fading comet trail of recent motion on the OpenCV frame,
        plus a visual 'READY' target reticle when paused at a starting position.
        """
        h, w, _ = frame.shape

        # Render Starting Position Anchor Cue if poised and ready
        if self.is_anchored and self.anchor_position is not None:
            ax, ay = int(self.anchor_position[0] * w), int(self.anchor_position[1] * h)
            # Outer subtle pulse ring
            cv2.circle(frame, (ax, ay), 18, (0, 255, 180), 2, cv2.LINE_AA)
            cv2.circle(frame, (ax, ay), 4, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                "READY",
                (ax - 22, ay - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 180),
                1,
                cv2.LINE_AA,
            )

        pts = list(self.trajectory)
        n = len(pts)
        if n < 2:
            return

        for i in range(1, n):
            # Calculate progressive alpha / thickness
            progress = i / n
            thickness = max(1, int(max_thickness * progress))

            # Coordinates
            p1 = (int(pts[i - 1][0] * w), int(pts[i - 1][1] * h))
            p2 = (int(pts[i][0] * w), int(pts[i][1] * h))

            # Draw outer glow & core line
            cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
            if thickness > 2:
                cv2.circle(frame, p2, thickness - 1, (255, 255, 255), -1, cv2.LINE_AA)

        # Render Top Real-Time Kinetic Speedometer HUD
        stats = self.get_kinetic_stats()
        state = stats["state"]
        spd = stats["speed"]
        thresh = stats["speed_threshold"]

        hud_w, hud_h = 240, 28
        hx1 = (w - hud_w) // 2
        hy1 = 12
        hx2 = hx1 + hud_w
        hy2 = hy1 + hud_h

        # Translucent dark backing
        overlay = frame.copy()
        cv2.rectangle(overlay, (hx1, hy1), (hx2, hy2), (18, 18, 22), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        # Border color depending on state
        if state == "ARMED":
            border_col = (0, 255, 160)
            tag_text = "READY"
        elif state == "IMPULSE":
            border_col = (0, 220, 255)
            tag_text = "IMPULSE"
        elif state == "COOLDOWN":
            border_col = (0, 140, 255)
            tag_text = "COOLDOWN"
        else:
            border_col = (120, 120, 130)
            tag_text = "STEADY"

        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), border_col, 1, cv2.LINE_AA)

        # Mini speed bar inside HUD
        bar_x1 = hx1 + 80
        bar_y1 = hy1 + 8
        bar_w = 100
        bar_h = 12
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + bar_h), (40, 40, 50), -1)

        fill_ratio = min(1.0, spd / max(thresh * 2.0, 0.01))
        fill_w = int(bar_w * fill_ratio)
        bar_col = (0, 255, 120) if spd >= thresh else (0, 200, 240)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y1 + bar_h), bar_col, -1)

        # Trigger threshold notch on bar
        thresh_x = bar_x1 + int(bar_w * 0.5)
        cv2.line(frame, (thresh_x, bar_y1 - 2), (thresh_x, bar_y1 + bar_h + 2), (0, 255, 255), 2, cv2.LINE_AA)

        # Text labels
        cv2.putText(frame, tag_text, (hx1 + 8, hy1 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, border_col, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{spd:.2f}", (bar_x1 + bar_w + 8, hy1 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 240), 1, cv2.LINE_AA)

    def get_kinetic_stats(self) -> Dict[str, Any]:
        """Returns real-time kinetic telemetry for studio gauge & HUD."""
        now = time.time()
        state = "COOLDOWN" if (now - self.last_trigger_time) < self.cooldown_seconds else (
            "ARMED" if self.is_anchored else ("IMPULSE" if self.active_start_pos else "IDLE")
        )
        return {
            "state": state,
            "speed": float(self.current_speed),
            "speed_threshold": float(self.min_velocity),
            "peak_disp": float(self.peak_disp),
            "min_disp": float(self.min_displacement),
            "velocity": self.current_velocity,
            "is_anchored": self.is_anchored,
        }


class MotionCalibrator:
    """
    Step-by-step interactive calibrator for learning personal hand movement kinematics.
    Guides the user to swipe in 4 directions and computes customized thresholds.
    """
    DIRECTIONS = ["swipe_right", "swipe_left"]
    DIRECTION_LABELS = {
        "swipe_right": "Flick RIGHT (Alt+Tab)",
        "swipe_left": "Flick LEFT (Shift+Alt+Tab)",
    }
    TARGET_SAMPLES_PER_DIR = 3

    def __init__(self):
        self.is_calibrating = False
        self.current_dir_index = 0
        self.samples: Dict[str, List[Dict[str, float]]] = {d: [] for d in self.DIRECTIONS}
        self.last_record_time = 0.0

        # Hand-Reset Gate: generous cooldown to comfortably return hand to start position
        self.waiting_for_reset = False
        self.reset_start_time = 0.0
        self.min_reset_time = 1.8  # Generous mandatory pause (1.8s) to bring hand back
        self.steady_hold_time = 0.35  # Must hold still at start position for 350ms
        self.anchor_hold_start: Optional[float] = None

    def start(self):
        """Starts calibration sequence from the first direction."""
        self.is_calibrating = True
        self.current_dir_index = 0
        self.samples = {d: [] for d in self.DIRECTIONS}
        self.last_record_time = time.time()
        self.waiting_for_reset = False
        self.reset_start_time = 0.0
        self.anchor_hold_start = None

    def cancel(self):
        """Cancels calibration."""
        self.is_calibrating = False
        self.current_dir_index = 0
        self.samples = {d: [] for d in self.DIRECTIONS}
        self.waiting_for_reset = False
        self.anchor_hold_start = None

    def reset_time_remaining(self, timestamp: Optional[float] = None) -> float:
        """Returns seconds remaining in the mandatory reset phase."""
        if not self.waiting_for_reset:
            return 0.0
        now = timestamp if timestamp is not None else time.time()
        return max(0.0, self.min_reset_time - (now - self.reset_start_time))

    def update_state(self, is_anchored: bool, timestamp: Optional[float] = None) -> None:
        """
        Manages the Hand-Reset Gate:
        After a stroke is recorded, user must reset hand and hold still (is_anchored)
        for at least min_reset_time + steady_hold_time before the next stroke is unlocked.
        """
        if not self.is_calibrating:
            return
        now = timestamp if timestamp is not None else time.time()
        if self.waiting_for_reset:
            # 1. Mandatory reset timer must elapse first
            if (now - self.reset_start_time) < self.min_reset_time:
                self.anchor_hold_start = None
                return
            # 2. Must hold hand still at rest for steady_hold_time
            if is_anchored:
                if self.anchor_hold_start is None:
                    self.anchor_hold_start = now
                elif (now - self.anchor_hold_start) >= self.steady_hold_time:
                    self.waiting_for_reset = False
                    self.anchor_hold_start = None
            else:
                self.anchor_hold_start = None

    @property
    def current_direction(self) -> Optional[str]:
        if not self.is_calibrating or self.current_dir_index >= len(self.DIRECTIONS):
            return None
        return self.DIRECTIONS[self.current_dir_index]

    @property
    def current_label(self) -> str:
        d = self.current_direction
        return self.DIRECTION_LABELS.get(d, "") if d else ""

    @property
    def current_count(self) -> int:
        d = self.current_direction
        return len(self.samples[d]) if d else 0

    @property
    def is_finished(self) -> bool:
        return self.is_calibrating and self.current_dir_index >= len(self.DIRECTIONS)

    def check_and_record_stroke(
        self,
        pts: List[Tuple[float, float, float]],
        is_anchored: bool = False,
        active_anchor: Optional[Tuple[float, float]] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Inspects recent trajectory to see if a stroke in the current target direction occurred.
        Guarded by the Hand-Reset Gate to prevent capturing reset movements.
        """
        if not self.is_calibrating or self.waiting_for_reset or len(pts) < 2:
            return None
        target_dir = self.current_direction
        if not target_dir:
            return None

        latest_x, latest_y, latest_t = pts[-1]

        # 1. Primary: Evaluate displacement from Start Anchor to Peak End Position
        if active_anchor is not None:
            ax, ay = active_anchor
            peak_pt = max(pts, key=lambda p: math.hypot(p[0] - ax, p[1] - ay))
            dx = peak_pt[0] - ax
            dy = peak_pt[1] - ay
            disp = math.hypot(dx, dy)
            dt = max(peak_pt[2] - pts[0][2], 0.02)
            vel = disp / dt
            dist_x, dist_y = abs(dx), abs(dy)

            if disp >= 0.045:
                matched = False
                if target_dir == "swipe_right" and dx > 0 and dist_x > 1.10 * dist_y:
                    matched = True
                elif target_dir == "swipe_left" and dx < 0 and dist_x > 1.10 * dist_y:
                    matched = True
                elif target_dir == "swipe_up" and dy < 0 and dist_y > 1.10 * dist_x:
                    matched = True
                elif target_dir == "swipe_down" and dy > 0 and dist_y > 1.10 * dist_x:
                    matched = True

                if matched:
                    now = time.time()
                    self.last_record_time = now
                    self.waiting_for_reset = True
                    self.reset_start_time = now
                    self.anchor_hold_start = None
                    stroke_info = {
                        "displacement": float(disp),
                        "velocity": float(vel),
                        "duration": float(dt),
                        "anchor_speed": 0.15,
                    }
                    self.samples[target_dir].append(stroke_info)
                    if len(self.samples[target_dir]) >= self.TARGET_SAMPLES_PER_DIR:
                        self.current_dir_index += 1
                    return stroke_info

        # 2. Secondary fallback: check trajectory backwards across recent frames
        for i in range(len(pts) - 2, -1, -1):
            past_x, past_y, past_t = pts[i]
            dt = latest_t - past_t
            if dt < 0.028:
                continue
            if dt > 1.0:
                break
            dx = latest_x - past_x
            dy = latest_y - past_y
            disp = math.hypot(dx, dy)
            if disp < 0.055:
                continue
            dist_x, dist_y = abs(dx), abs(dy)
            vel = disp / dt

            matched = False
            if target_dir == "swipe_right" and dx > 0 and dist_x > 1.15 * dist_y:
                matched = True
            elif target_dir == "swipe_left" and dx < 0 and dist_x > 1.15 * dist_y:
                matched = True
            elif target_dir == "swipe_up" and dy < 0 and dist_y > 1.15 * dist_x:
                matched = True
            elif target_dir == "swipe_down" and dy > 0 and dist_y > 1.15 * dist_x:
                matched = True

            if matched:
                now = time.time()
                self.last_record_time = now
                self.waiting_for_reset = True
                self.reset_start_time = now
                self.anchor_hold_start = None
                stroke_info = {
                    "displacement": float(disp),
                    "velocity": float(vel),
                    "duration": float(dt),
                    "anchor_speed": 0.15,
                }
                self.samples[target_dir].append(stroke_info)
                if len(self.samples[target_dir]) >= self.TARGET_SAMPLES_PER_DIR:
                    self.current_dir_index += 1
                return stroke_info

        return None

    def record_stroke(
        self, displacement: float, velocity: float, duration: float, anchor_speed: float = 0.15
    ) -> bool:
        """
        Records one calibration stroke for the active direction.
        Returns True if stroke was accepted.
        """
        d = self.current_direction
        if not d:
            return False

        now = time.time()
        self.last_record_time = now
        self.waiting_for_reset = True
        self.reset_start_time = now
        self.anchor_hold_start = None

        self.samples[d].append({
            "displacement": float(displacement),
            "velocity": float(velocity),
            "duration": float(duration),
            "anchor_speed": float(anchor_speed),
        })

        if len(self.samples[d]) >= self.TARGET_SAMPLES_PER_DIR:
            self.current_dir_index += 1
        return True

    def compute_calibrated_profile(self) -> Dict[str, Any]:
        """
        Analyzes recorded samples and generates a tailored personal motion profile.
        """
        all_disps = [s["displacement"] for dir_samples in self.samples.values() for s in dir_samples]
        all_vels = [s["velocity"] for dir_samples in self.samples.values() for s in dir_samples]
        all_durs = [s["duration"] for dir_samples in self.samples.values() for s in dir_samples]
        all_anchors = [s["anchor_speed"] for dir_samples in self.samples.values() for s in dir_samples]

        if not all_disps:
            return load_motion_profile()

        # Minimum required displacement is set to 70% of user's typical 25th percentile stroke
        p_disp = float(np.percentile(all_disps, 25)) if len(all_disps) > 1 else all_disps[0]
        min_disp = max(0.050, min(0.20, round(p_disp * 0.70, 3)))

        # Velocity threshold set to 65% of user's typical stroke speed
        p_vel = float(np.percentile(all_vels, 25)) if len(all_vels) > 1 else all_vels[0]
        min_vel = max(0.12, min(0.35, round(p_vel * 0.65, 3)))

        # Duration bounds (generous to accommodate natural human variation)
        min_dur = min(0.08, max(0.025, round(float(np.min(all_durs)) * 0.75, 3)))
        max_dur = max(0.65, min(1.00, round(float(np.max(all_durs)) * 1.40, 3)))

        # Anchor speed threshold
        max_anchor = max(0.18, min(0.35, round(float(np.mean(all_anchors)) * 1.5, 3)))

        profile = {
            "min_displacement": min_disp,
            "min_velocity": min_vel,
            "min_duration": min_dur,
            "max_duration": max_dur,
            "max_anchor_speed": max_anchor,
            "smoothing_alpha": 0.75,
        }
        return profile

    def save_and_apply(self, tracker: MotionTracker) -> Dict[str, Any]:
        """Calculates, saves to disk, and applies profile to the tracker."""
        profile = self.compute_calibrated_profile()
        save_motion_profile(profile)
        tracker.apply_profile(profile)
        self.is_calibrating = False
        return profile


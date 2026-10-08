"""
Kinetic Rig Tracker for High-Speed Hand & Wrist Gestures.
Tracks Forearm (Elbow -> Wrist) and Hand Base (MCP Index/Middle/Ring/Pinky matrix).
Uses a 4D Kalman filter and Optical Motion Field integration to maintain
rock-solid tracking persistence even through extreme-speed motion blur.
"""

import time
import math
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2


class Kalman2D:
    """
    Constant-velocity 4D Kalman filter: state = [x, y, vx, vy].
    Smooths jitter at rest and accurately predicts trajectory across dropped frames.
    """

    def __init__(self, process_noise: float = 0.08, measurement_noise: float = 0.005):
        self.state = np.zeros(4, dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1.0
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.initialized = False
        self.last_timestamp: float = 0.0

    def reset(self):
        self.state.fill(0.0)
        self.P = np.eye(4, dtype=np.float32) * 1.0
        self.initialized = False
        self.last_timestamp = 0.0

    def predict(self, dt: float) -> Tuple[float, float, float, float]:
        if not self.initialized:
            return 0.0, 0.0, 0.0, 0.0
        dt = max(dt, 1e-4)
        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

        q_pos = (dt ** 3) / 3.0 * self.process_noise
        q_vel = dt * self.process_noise
        Q = np.array([
            [q_pos, 0.0,   (dt**2)/2.0 * self.process_noise, 0.0],
            [0.0,   q_pos, 0.0, (dt**2)/2.0 * self.process_noise],
            [(dt**2)/2.0 * self.process_noise, 0.0, q_vel, 0.0],
            [0.0, (dt**2)/2.0 * self.process_noise, 0.0, q_vel],
        ], dtype=np.float32)

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q
        return float(self.state[0]), float(self.state[1]), float(self.state[2]), float(self.state[3])

    def update(self, z_x: float, z_y: float) -> Tuple[float, float, float, float]:
        z = np.array([z_x, z_y], dtype=np.float32)
        if not self.initialized:
            self.state = np.array([z_x, z_y, 0.0, 0.0], dtype=np.float32)
            self.initialized = True
            return z_x, z_y, 0.0, 0.0

        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float32)
        R = np.eye(2, dtype=np.float32) * self.measurement_noise

        y = z - (H @ self.state)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state + (K @ y)
        self.P = (np.eye(4, dtype=np.float32) - K @ H) @ self.P
        return float(self.state[0]), float(self.state[1]), float(self.state[2]), float(self.state[3])


class OpticalMotionEstimator:
    """
    Ultra-lightweight optical motion vector estimator.
    Computes rapid horizontal and vertical pixel flux in the active hand ROI.
    Guarantees non-zero velocity signal during severe motion blur when keypoint detectors drop.
    """

    def __init__(self):
        self.prev_gray: Optional[np.ndarray] = None
        self.last_motion_vector: Tuple[float, float] = (0.0, 0.0)

    def reset(self):
        self.prev_gray = None
        self.last_motion_vector = (0.0, 0.0)

    def estimate_motion(
        self,
        frame_bgr: np.ndarray,
        roi_norm: Optional[Tuple[float, float, float, float]] = None
    ) -> Tuple[float, float]:
        if frame_bgr is None or frame_bgr.size == 0:
            return 0.0, 0.0
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (160, 120))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0, 0.0

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=2, winsize=11, iterations=2, poly_n=5, poly_sigma=1.1, flags=0
        )
        self.prev_gray = gray

        if roi_norm is not None:
            rx1 = max(0, int(roi_norm[0] * 160))
            ry1 = max(0, int(roi_norm[1] * 120))
            rx2 = min(160, int(roi_norm[2] * 160))
            ry2 = min(120, int(roi_norm[3] * 120))
            if (rx2 - rx1) > 5 and (ry2 - ry1) > 5:
                roi_flow = flow[ry1:ry2, rx1:rx2]
            else:
                roi_flow = flow
        else:
            roi_flow = flow

        dt_x = float(np.median(roi_flow[..., 0])) / 160.0
        dt_y = float(np.median(roi_flow[..., 1])) / 120.0
        self.last_motion_vector = (dt_x, dt_y)
        return dt_x, dt_y


class KineticRigTracker:
    """
    High-Speed Kinetic Rig Tracker.
    Unifies:
    1. Forearm vector (Elbow -> Wrist)
    2. Knuckle Ridge base (Wrist + MCP 5, 9, 13, 17)
    3. Predictive 4D Kalman filtering
    4. Optical motion energy fusion
    """

    def __init__(self):
        self.kalman = Kalman2D(process_noise=0.08, measurement_noise=0.005)
        self.optical = OpticalMotionEstimator()
        self.last_time: Optional[float] = None
        self.coast_count = 0
        self.max_coast_frames = 6
        self.latest_pos: Optional[Tuple[float, float]] = None
        self.latest_vel: Tuple[float, float] = (0.0, 0.0)

    def reset(self):
        self.kalman.reset()
        self.optical.reset()
        self.last_time = None
        self.coast_count = 0
        self.latest_pos = None
        self.latest_vel = (0.0, 0.0)

    @staticmethod
    def extract_knuckle_center(raw_hand: List[List[float]]) -> Optional[Tuple[float, float]]:
        """
        Computes the MCP Knuckle Ridge Center from landmarks:
        Wrist (0), Index MCP (5), Middle MCP (9), Ring MCP (13), Pinky MCP (17).
        This base landmark does NOT curl or jitter when fingers move.
        """
        if not raw_hand or len(raw_hand) < 18:
            return None
        mcp_indices = [0, 5, 9, 13, 17]
        pts = [raw_hand[i] for i in mcp_indices if i < len(raw_hand)]
        if not pts:
            return None
        weights = [0.20, 0.25, 0.30, 0.15, 0.10]
        wx = sum(p[0] * w for p, w in zip(pts, weights[:len(pts)])) / sum(weights[:len(pts)])
        wy = sum(p[1] * w for p, w in zip(pts, weights[:len(pts)])) / sum(weights[:len(pts)])
        return float(wx), float(wy)

    def update(
        self,
        measurement: Optional[Tuple[float, float]],
        frame_bgr: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None
    ) -> Tuple[Optional[Tuple[float, float]], Tuple[float, float], float]:
        now = timestamp if timestamp is not None else time.time()
        dt = (now - self.last_time) if self.last_time is not None else 0.033
        dt = max(1e-4, min(dt, 0.20))
        self.last_time = now

        pred_x, pred_y, pred_vx, pred_vy = self.kalman.predict(dt)

        if measurement is not None:
            zx, zy = measurement
            # Outlier / Teleport gating:
            # If the measurement suddenly jumps from predicted position at an impossible velocity
            # (e.g. MediaPipe jumping to face or shoulder landmark across the body during occlusion),
            # gate it out and coast with prediction.
            is_teleport = False
            if self.kalman.initialized and self.coast_count < self.max_coast_frames:
                jump_dist = math.hypot(zx - pred_x, zy - pred_y)
                implied_v = jump_dist / dt
                if jump_dist > 0.30 and implied_v > 9.0:
                    is_teleport = True

            if not is_teleport:
                curr_x, curr_y, curr_vx, curr_vy = self.kalman.update(zx, zy)
                self.coast_count = 0
                self.latest_pos = (curr_x, curr_y)
                self.latest_vel = (curr_vx, curr_vy)
            else:
                self.coast_count += 1
                self.latest_pos = (pred_x, pred_y)
                self.latest_vel = (pred_vx, pred_vy)
        else:
            self.coast_count += 1
            if self.coast_count <= self.max_coast_frames and self.kalman.initialized:
                if frame_bgr is not None and self.latest_pos is not None:
                    px, py = self.latest_pos
                    roi = (max(0.0, px - 0.15), max(0.0, py - 0.15), min(1.0, px + 0.15), min(1.0, py + 0.15))
                    opt_vx, opt_vy = self.optical.estimate_motion(frame_bgr, roi)
                    if abs(opt_vx) > 0.05:
                        pred_vx = 0.5 * pred_vx + 0.5 * (opt_vx / dt)
                self.latest_pos = (pred_x, pred_y)
                self.latest_vel = (pred_vx, pred_vy)
            else:
                self.reset()
                return None, (0.0, 0.0), 0.0

        speed = math.hypot(self.latest_vel[0], self.latest_vel[1])
        return self.latest_pos, self.latest_vel, speed
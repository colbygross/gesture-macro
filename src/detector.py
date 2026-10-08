"""
Upper-Body & Arm Rig Landmark Detector.
Integrates MediaPipe Pose Heavy (arms, shoulders, head, torso) and MediaPipe Hands (fingers).
Features Simultaneous Dual-Hand Tracking & Instant Dual-Path Acquisition:
- Tracks BOTH hands simultaneously (Left & Right) with multi-color skeletons.
- Uses Pose wrists to acquire hands instantly on the first frame even if clenched in fists.
Cross-platform support for Windows and macOS.
"""

import sys
import math
import time
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
import numpy as np
import cv2
import mediapipe as mp

from src.config import (
    MP_MIN_DETECTION_CONFIDENCE,
    MP_MIN_TRACKING_CONFIDENCE,
    HAND_LANDMARKER_MODEL_PATH,
    POSE_LANDMARKER_MODEL_PATH,
    MODELS_DIR,
)
from src.rig_tracker import KineticRigTracker

# Upper-body landmark indices from MediaPipe Pose (33 points)
UPPER_BODY_INDICES = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

# Connections between upper body landmark indices
BODY_CONNECTIONS = [
    (11, 12),  # Shoulder span
    (11, 13), (13, 15),  # Left arm (shoulder -> elbow -> wrist)
    (12, 14), (14, 16),  # Right arm (shoulder -> elbow -> wrist)
    (11, 23), (12, 24),  # Left and right torso sides
    (23, 24),  # Hip span
    (0, 11), (0, 12),   # Neck/head to shoulders
]

# Standard hand skeleton connections (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm knuckles
]

POSE_HEAVY_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


class UpperBodyRigDetector:
    """
    Combined Upper-Body (Pose Heavy) and Dual-Hand Rig Detector.
    Tracks both hands simultaneously and renders distinct colored skeletons.
    """

    def __init__(
        self,
        min_detection_confidence: float = MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MP_MIN_TRACKING_CONFIDENCE,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.last_timestamp_ms = 0

        self._ensure_models_downloaded()
        self._init_landmarkers()

    def _ensure_models_downloaded(self):
        """Ensures Pose Heavy and Hand models are cached locally."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if not POSE_LANDMARKER_MODEL_PATH.exists():
            print("[DOWNLOAD] Downloading Upper-Body Pose Heavy model (~30.6 MB)...")
            urllib.request.urlretrieve(POSE_HEAVY_URL, POSE_LANDMARKER_MODEL_PATH)
        if not HAND_LANDMARKER_MODEL_PATH.exists():
            print("[DOWNLOAD] Downloading Hand Landmark model (~7.8 MB)...")
            urllib.request.urlretrieve(HAND_MODEL_URL, HAND_LANDMARKER_MODEL_PATH)

    def _init_landmarkers(self):
        """Initializes Pose Heavy and Hand detectors."""
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        # Pose Heavy Detector
        pose_opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(POSE_LANDMARKER_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=self.min_tracking_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_opts)

        # Primary Hand Video Tracker - Tracks up to 2 hands simultaneously
        hand_opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(HAND_LANDMARKER_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_tracking_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_opts)

        # Auxiliary Wrist-Bootstrapped Acquisition Finder
        crop_opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(HAND_LANDMARKER_MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.12,
            min_hand_presence_confidence=0.15,
            min_tracking_confidence=0.20,
        )
        self.crop_detector = vision.HandLandmarker.create_from_options(crop_opts)
        self.active_hand_pos: Optional[Tuple[float, float]] = None

    @staticmethod
    def compute_fingertip_center(
        raw_coords: List[List[float]], pixel_pts: List[Tuple[int, int]]
    ) -> Tuple[Tuple[float, float], Tuple[int, int]]:
        """
        Computes the primary fingertip tracking point for wrist flicks and gesture navigation.
        Identifies extended fingertips (index, middle, ring, pinky) and returns their centroid,
        giving the maximum lever-arm displacement and speed when pivoting from the wrist.
        Falls back to the index fingertip (8) if fingers are curled.
        """
        if not raw_coords or len(raw_coords) < 9:
            return (0.5, 0.5), (0, 0)

        wrist = raw_coords[0]
        # (tip_idx, pip_idx) for index, middle, ring, pinky
        finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
        extended_indices = []

        for tip_idx, pip_idx in finger_pairs:
            if tip_idx < len(raw_coords) and pip_idx < len(raw_coords):
                tip = raw_coords[tip_idx]
                pip = raw_coords[pip_idx]
                dist_tip_sq = (tip[0] - wrist[0]) ** 2 + (tip[1] - wrist[1]) ** 2
                dist_pip_sq = (pip[0] - wrist[0]) ** 2 + (pip[1] - wrist[1]) ** 2
                if dist_tip_sq > dist_pip_sq * 1.10:
                    extended_indices.append(tip_idx)

        # Centroid of extended fingertips, or fallback to index fingertip (8)
        if extended_indices:
            target_indices = extended_indices
        else:
            target_indices = [8] if len(raw_coords) > 8 else [0]

        tip_raw = [raw_coords[i] for i in target_indices]
        tip_px = [pixel_pts[i] for i in target_indices]

        norm_center = (
            float(np.mean([p[0] for p in tip_raw])),
            float(np.mean([p[1] for p in tip_raw])),
        )
        px_center = (
            int(np.mean([p[0] for p in tip_px])),
            int(np.mean([p[1] for p in tip_px])),
        )
        return norm_center, px_center

    def find_rig_landmarks(
        self, frame: np.ndarray, draw: bool = True
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Extracts upper body and both hands. Tracks both hands simultaneously.
        """
        h, w, _ = frame.shape
        rgb_frame = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        now_ms = int(time.time() * 1000)
        if now_ms <= self.last_timestamp_ms:
            now_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = now_ms

        # Run Pose and Hand tracking
        pose_result = self.pose_detector.detect_for_video(mp_image, now_ms)
        hand_result = self.hand_detector.detect_for_video(mp_image, now_ms)

        body_detected = bool(pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0)
        detected_hands = hand_result.hand_landmarks if hand_result.hand_landmarks else []

        all_hands_info: List[Dict[str, Any]] = []

        # 1. Process all hands found by primary video tracker
        for hand_lms in detected_hands:
            raw_coords = [[lm.x, lm.y, lm.z] for lm in hand_lms]
            pixel_pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
            norm_hand = self.normalize_hand(np.array(raw_coords, dtype=np.float32))
            
            # Compute fingertip action center and knuckle ridge base center
            tip_norm, tip_px = self.compute_fingertip_center(raw_coords, pixel_pts)
            knuckle_norm = KineticRigTracker.extract_knuckle_center(raw_coords)
            if knuckle_norm is not None:
                knuckle_px = (int(knuckle_norm[0] * w), int(knuckle_norm[1] * h))
            else:
                knuckle_norm, knuckle_px = tip_norm, tip_px
            
            all_hands_info.append({
                "raw": raw_coords,
                "pixels": pixel_pts,
                "norm": norm_hand,
                "palm_center_norm": knuckle_norm,
                "palm_center_px": knuckle_px,
                "knuckle_center_norm": knuckle_norm,
                "knuckle_center_px": knuckle_px,
                "fingertip_norm": tip_norm,
                "fingertip_px": tip_px,
            })

        # 2. Wrist-bootstrapped acquisition ONLY if 0 hands found by primary detector
        if len(all_hands_info) == 0 and body_detected:
            pose_lms = pose_result.pose_landmarks[0]
            existing_wrist_pts = [h_info["pixels"][0] for h_info in all_hands_info]

            for w_idx in [16, 15]:  # Right wrist, Left wrist
                if len(all_hands_info) >= 1:
                    break
                if w_idx < len(pose_lms):
                    w_lm = pose_lms[w_idx]
                    if 0.05 < w_lm.x < 0.95 and 0.05 < w_lm.y < 0.95:
                        cx, cy = int(w_lm.x * w), int(w_lm.y * h)
                        # Avoid cropping on the face zone (if nose is detected)
                        if len(pose_lms) > 0:
                            nose_lm = pose_lms[0]
                            if math.hypot(w_lm.x - nose_lm.x, w_lm.y - nose_lm.y) < 0.15:
                                continue  # Hand/wrist is overlapping face, skip crop false positives

                        # Check if this wrist already has a tracked hand
                        already_tracked = any(
                            np.hypot(cx - ex_x, cy - ex_y) < (min(w, h) * 0.12)
                            for ex_x, ex_y in existing_wrist_pts
                        )
                        if not already_tracked:
                            side = int(min(w, h) * 0.44)
                            side = min(side, w, h)
                            half = side // 2
                            # Clamp center so crop stays in-bounds and remains perfectly square
                            clamped_cx = max(half, min(w - half, cx))
                            clamped_cy = max(half, min(h - half, cy))
                            x1 = clamped_cx - half
                            x2 = clamped_cx + half
                            y1 = clamped_cy - half
                            y2 = clamped_cy + half
                            crop = rgb_frame[y1:y2, x1:x2]
                            if crop.shape[0] > 40 and crop.shape[1] > 40:
                                try:
                                    crop_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
                                    crop_res = self.crop_detector.detect(crop_mp)
                                    if crop_res.hand_landmarks and len(crop_res.hand_landmarks) > 0:
                                        chosen = crop_res.hand_landmarks[0]
                                        cw, ch = crop.shape[1], crop.shape[0]
                                        raw_pts = []
                                        px_pts = []
                                        for lm in chosen:
                                             fx = (x1 + lm.x * cw) / w
                                             fy = (y1 + lm.y * ch) / h
                                             fz = lm.z * (cw / w)
                                             raw_pts.append([fx, fy, fz])
                                             px_pts.append((int(fx * w), int(fy * h)))
                                        norm_hand = self.normalize_hand(np.array(raw_pts, dtype=np.float32))
                                        
                                        # Compute fingertip action center and knuckle ridge base center
                                        tip_norm, tip_px = self.compute_fingertip_center(raw_pts, px_pts)
                                        knuckle_norm = KineticRigTracker.extract_knuckle_center(raw_pts)
                                        if knuckle_norm is not None:
                                            knuckle_px = (int(knuckle_norm[0] * w), int(knuckle_norm[1] * h))
                                        else:
                                            knuckle_norm, knuckle_px = tip_norm, tip_px
                                        
                                        all_hands_info.append({
                                            "raw": raw_pts,
                                            "pixels": px_pts,
                                            "norm": norm_hand,
                                            "palm_center_norm": knuckle_norm,
                                            "palm_center_px": knuckle_px,
                                            "knuckle_center_norm": knuckle_norm,
                                            "knuckle_center_px": knuckle_px,
                                            "fingertip_norm": tip_norm,
                                            "fingertip_px": tip_px,
                                        })
                                except Exception:
                                    pass

        hand_detected = len(all_hands_info) > 0

        if not body_detected and not hand_detected:
            return None, {"body_detected": False, "hand_detected": False, "hands": []}

        # 3. Normalize Upper Body (15 landmarks -> 45 features)
        body_pixels: Dict[int, Tuple[int, int]] = {}
        body_coords = np.zeros((15, 3), dtype=np.float32)

        if body_detected:
            pose_lms = pose_result.pose_landmarks[0]
            for idx, body_idx in enumerate(UPPER_BODY_INDICES):
                if body_idx < len(pose_lms):
                    lm = pose_lms[body_idx]
                    body_coords[idx] = [lm.x, lm.y, lm.z]
                    body_pixels[body_idx] = (int(lm.x * w), int(lm.y * h))
            norm_body = self.normalize_upper_body(body_coords)
        else:
            norm_body = np.zeros(45, dtype=np.float32)

        face_center = None
        if body_detected and 0 in body_pixels:
            face_center = (body_pixels[0][0] / max(w, 1), body_pixels[0][1] / max(h, 1))

        # 4. Primary and Secondary Hand Features with Sticky Temporal Association
        all_hand_pixels = [h["pixels"] for h in all_hands_info]
        all_norm_hands = [h["norm"] for h in all_hands_info]
        all_palm_centers_px = [h["palm_center_px"] for h in all_hands_info]

        primary_norm_hand = all_norm_hands[0] if all_norm_hands else np.zeros(63, dtype=np.float32)

        # Determine primary knuckle position for motion tracking with temporal continuity
        if all_hands_info:
            if len(all_hands_info) > 1 and self.active_hand_pos is not None:
                chosen_hand = min(
                    all_hands_info,
                    key=lambda h: math.hypot(
                        h["knuckle_center_norm"][0] - self.active_hand_pos[0],
                        h["knuckle_center_norm"][1] - self.active_hand_pos[1]
                    )
                )
            else:
                chosen_hand = all_hands_info[0]

            self.active_hand_pos = chosen_hand["knuckle_center_norm"]
            primary_palm_norm = chosen_hand["knuckle_center_norm"]
            primary_palm_px = chosen_hand["knuckle_center_px"]
            primary_tip_norm = chosen_hand["fingertip_norm"]
            primary_tip_px = chosen_hand["fingertip_px"]
        else:
            self.active_hand_pos = None
            primary_palm_norm = None
            primary_palm_px = None
            primary_tip_norm = None
            primary_tip_px = None
            # Append wrist to drawing list for visual skeleton rig without corrupting motion tracker
            if body_detected and (16 in body_pixels or 15 in body_pixels):
                w_idx = 16 if 16 in body_pixels else 15
                all_palm_centers_px.append(body_pixels[w_idx])

        # 5. Extract Forearm Rig Vector (Elbow -> Wrist) matched to the active hand
        forearm_vector = None
        forearm_pixels = None
        if body_detected and primary_palm_norm is not None:
            hx, hy = primary_palm_norm
            best_dist = float("inf")
            for e_idx, w_idx in [(14, 16), (13, 15)]:
                if e_idx in body_pixels and w_idx in body_pixels:
                    ep = body_pixels[e_idx]
                    wp = body_pixels[w_idx]
                    wn = (wp[0] / max(w, 1), wp[1] / max(h, 1))
                    dist = math.hypot(wn[0] - hx, wn[1] - hy)
                    # Must be anatomically contiguous with the active hand (< 0.25 screen height)
                    if dist < best_dist and dist < 0.25:
                        best_dist = dist
                        forearm_pixels = (ep, wp)
                        forearm_vector = (float(wp[0] - ep[0]) / max(w, 1), float(wp[1] - ep[1]) / max(h, 1))

        # Draw visual rig with BOTH hands & palm center reticles
        if draw:
            self._draw_rig(frame, body_pixels, all_hand_pixels, all_palm_centers_px)

        combined_features = np.concatenate([norm_body, primary_norm_hand])

        debug_info = {
            "body_detected": body_detected,
            "hand_detected": hand_detected,
            "hand_count": len(all_hands_info),
            "body_pixels": body_pixels,
            "all_hands_pixels": all_hand_pixels,
            "hand_pixels": all_hand_pixels[0] if all_hand_pixels else [],
            "palm_center": primary_palm_norm,
            "palm_center_px": primary_palm_px,
            "knuckle_center": primary_palm_norm,
            "knuckle_center_px": primary_palm_px,
            "fingertip_center": primary_tip_norm,
            "fingertip_center_px": primary_tip_px,
            "tracking_point": primary_palm_norm,
            "tracking_point_px": primary_palm_px,
            "forearm_vector": forearm_vector,
            "forearm_pixels": forearm_pixels,
            "face_center": face_center,
            "norm_hand": primary_norm_hand,
            "all_norm_hands": all_norm_hands,
            "norm_body": norm_body,
        }
        return combined_features, debug_info

    def find_landmarks(
        self, frame: np.ndarray, draw: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[Any]]:
        """Backward-compatible method."""
        combined_feats, debug_info = self.find_rig_landmarks(frame, draw=draw)
        if debug_info.get("hand_detected"):
            return debug_info["norm_hand"], debug_info.get("hand_pixels")
        elif debug_info.get("body_detected"):
            return combined_feats, list(debug_info.get("body_pixels", {}).values())
        return None, None

    def _draw_rig(
        self,
        frame: np.ndarray,
        body_pixels: Dict[int, Tuple[int, int]],
        all_hands_pixels: List[List[Tuple[int, int]]],
        all_palm_centers_px: Optional[List[Tuple[int, int]]] = None,
    ):
        """Draws multi-color skeleton for upper body, hands, and palm centroids."""
        # Upper Body Bones
        for p1, p2 in BODY_CONNECTIONS:
            if p1 in body_pixels and p2 in body_pixels:
                pt1 = body_pixels[p1]
                pt2 = body_pixels[p2]
                is_arm = (p1, p2) in [(11, 13), (13, 15), (12, 14), (14, 16)]
                color = (0, 255, 255) if is_arm else (255, 180, 0)
                cv2.line(frame, pt1, pt2, color, 3, cv2.LINE_AA)

        # Upper Body Joints
        for idx, pt in body_pixels.items():
            if idx in (11, 12):  # Shoulders
                cv2.circle(frame, pt, 7, (0, 120, 255), -1, cv2.LINE_AA)
            elif idx in (13, 14):  # Elbows
                cv2.circle(frame, pt, 6, (0, 255, 255), -1, cv2.LINE_AA)
            elif idx in (15, 16):  # Wrists
                cv2.circle(frame, pt, 7, (255, 0, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 4, (200, 200, 200), -1, cv2.LINE_AA)

        # Color palettes for up to two hands
        hand_palettes = [
            # Hand 1: Cyan bones, pink fingertips, green joints
            {"bone": (0, 230, 255), "tip": (255, 0, 180), "joint": (0, 255, 120)},
            # Hand 2: Magenta bones, cyan fingertips, orange joints
            {"bone": (255, 0, 255), "tip": (0, 255, 255), "joint": (0, 165, 255)},
        ]

        # Draw skeletons for BOTH hands
        for h_idx, hand_pixels in enumerate(all_hands_pixels[:2]):
            palette = hand_palettes[h_idx % len(hand_palettes)]

            # Draw bones
            for p1, p2 in HAND_CONNECTIONS:
                if p1 < len(hand_pixels) and p2 < len(hand_pixels):
                    cv2.line(frame, hand_pixels[p1], hand_pixels[p2], palette["bone"], 2, cv2.LINE_AA)

            # Draw joint nodes
            for i, pt in enumerate(hand_pixels):
                if i in (0, 4, 8, 12, 16, 20):  # Fingertips & wrist
                    cv2.circle(frame, pt, 5, palette["tip"], -1, cv2.LINE_AA)
                    cv2.circle(frame, pt, 6, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, pt, 3, palette["joint"], -1, cv2.LINE_AA)

        # Draw Fingertip Action Center Target (The primary gesture & flick tracking point)
        if all_palm_centers_px:
            for tip_pt in all_palm_centers_px:
                if tip_pt:
                    # Outer target ring
                    cv2.circle(frame, tip_pt, 12, (0, 255, 255), 2, cv2.LINE_AA)
                    # Inner solid core
                    cv2.circle(frame, tip_pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
                    # Crosshair ticks
                    px, py = tip_pt
                    cv2.line(frame, (px - 16, py), (px - 7, py), (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.line(frame, (px + 7, py), (px + 16, py), (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.line(frame, (px, py - 16), (px, py - 7), (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.line(frame, (px, py + 7), (px, py + 16), (0, 255, 255), 2, cv2.LINE_AA)

    @staticmethod
    def normalize_upper_body(body_coords: np.ndarray) -> np.ndarray:
        """Chest-centered, shoulder-width normalized body points."""
        pts = body_coords.copy()
        left_sh = pts[1]
        right_sh = pts[2]

        chest_center = (left_sh + right_sh) / 2.0
        relative_pts = pts - chest_center

        shoulder_width = np.linalg.norm(left_sh - right_sh)
        if shoulder_width < 1e-4:
            shoulder_width = 1.0

        scaled_pts = relative_pts / shoulder_width
        return scaled_pts.flatten()

    @staticmethod
    def normalize_hand(hand_coords: np.ndarray) -> np.ndarray:
        """Wrist-centered, palm-scaled hand points."""
        pts = hand_coords.copy()
        wrist = pts[0]
        relative_pts = pts - wrist

        ref_distance = np.linalg.norm(relative_pts[9])
        if ref_distance < 1e-4:
            ref_distance = 1.0

        scaled_pts = relative_pts / ref_distance
        return scaled_pts.flatten()

    # Alias for backward compatibility
    normalize_landmarks = normalize_hand

    def close(self):
        """Releases detector resources."""
        if hasattr(self, "pose_detector"):
            try:
                self.pose_detector.close()
            except Exception:
                pass
        if hasattr(self, "hand_detector"):
            try:
                self.hand_detector.close()
            except Exception:
                pass
        if hasattr(self, "crop_detector"):
            try:
                self.crop_detector.close()
            except Exception:
                pass


# Alias for backward compatibility
HandDetector = UpperBodyRigDetector

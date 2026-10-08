"""
Configuration and settings for the Gesture Macro Controller & Studio.
Cross-platform support for Windows and macOS.
Supports Upper-Body Rig and Hands-Only modes with persistent macro mappings.
"""

import os
import json
import platform
from pathlib import Path
from typing import Dict, Any

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

GESTURES_CSV_PATH = DATA_DIR / "gestures.csv"
MODEL_PATH = MODELS_DIR / "gesture_classifier.joblib"
MACROS_JSON_PATH = DATA_DIR / "macros.json"

# Dynamic Motion Training Paths
MOTION_DATA_CSV_PATH = DATA_DIR / "motion_gestures.csv"
MOTION_MODEL_PATH = MODELS_DIR / "motion_classifier.joblib"


# Models
HAND_LANDMARKER_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
POSE_LANDMARKER_MODEL_PATH = MODELS_DIR / "pose_landmarker_heavy.task"

# Camera Settings
CAMERA_INDEX = int(os.environ.get("GESTURE_CAMERA_INDEX", 0))
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 60

# Detection Settings (Ultra-responsive for instant hand acquisition)
MP_MIN_DETECTION_CONFIDENCE = 0.15
MP_MIN_TRACKING_CONFIDENCE = 0.25

# Gesture Debounce & Action Settings (Requires deliberate hold)
DEBOUNCE_CONSECUTIVE_FRAMES = 14  # At 60 FPS, ~230ms steady hold required
ACTION_COOLDOWN_SECONDS = 1.0
PREDICTION_CONFIDENCE_THRESHOLD = 0.85

# Static gestures to ignore / treat as idle (user requested open_palm and fist do nothing)
IGNORED_STATIC_GESTURES = {"open_palm", "fist"}

MOTION_PROFILE_PATH = DATA_DIR / "motion_profile.json"

# Motion Gesture Settings (Tuned for responsive wrist flicks & larger gestures with zero micro-drift false positives)
MOTION_BUFFER_FRAMES = 60             # Trajectory window (up to 1.0s at 60 FPS, allows full arm strokes)
SWIPE_MIN_DISPLACEMENT = 0.085        # Cleanly separates deliberate flicks/swipes (~55px) from small micro-motions
SWIPE_MIN_VELOCITY = 0.30             # Rejects casual slow hand drift (requires intentional flick)
SWIPE_MIN_DURATION_SECONDS = 0.015    # Fast wrist flick snaps down to 15ms (1-2 frames)
SWIPE_MAX_DURATION_SECONDS = 0.85     # Generous window for wrist flicks and larger arm gestures
MOTION_COOLDOWN_SECONDS = 0.80        # Responsive cooldown (prevents locking out subsequent gestures)
REQUIRE_STARTING_ANCHOR = True        # Must start from a momentary pause/starting position
MAX_ANCHOR_SPEED = 0.28               # Maximum allowed speed during starting anchor pause
SMOOTHING_ALPHA = 0.85                # Responsive EMA smoothing for rapid flicks

ENABLE_CIRCLES = False               # Disable rotary circles to keep system simple and reliable
ENABLE_VERTICAL_SWIPES = False       # Focus exclusively on Left & Right wrist flicks (no scroll misfires)

# Platform detection
IS_MAC = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"

# Default Macro Mappings (Streamlined to ONLY essential horizontal navigation gestures)
DEFAULT_GESTURE_ACTIONS: Dict[str, Dict[str, Any]] = {
    "swipe_right": {
        "description": "Switch App (Alt+Tab)",
        "type": "hotkey",
        "keys": ["cmd", "tab"] if IS_MAC else ["alt", "tab"]
    },
    "swipe_left": {
        "description": "Previous App (Shift+Alt+Tab)",
        "type": "hotkey",
        "keys": ["shift", "cmd", "tab"] if IS_MAC else ["shift", "alt", "tab"]
    }
}


DEFAULT_MACRO_ACTIONS = DEFAULT_GESTURE_ACTIONS


def load_macro_actions() -> Dict[str, Dict[str, Any]]:
    """Loads macro mappings from data/macros.json or initializes with defaults."""
    if MACROS_JSON_PATH.exists():
        try:
            with open(MACROS_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load {MACROS_JSON_PATH}, using defaults: {e}")
    return DEFAULT_GESTURE_ACTIONS.copy()


def save_macro_actions(actions: Dict[str, Dict[str, Any]]) -> None:
    """Saves macro mappings to data/macros.json."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(MACROS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(actions, f, indent=2)
    print(f"[CONFIG] Saved macro mappings to {MACROS_JSON_PATH}")


def load_motion_profile() -> Dict[str, Any]:
    """Loads calibrated personal motion profile or returns default values."""
    defaults = {
        "min_displacement": SWIPE_MIN_DISPLACEMENT,
        "min_velocity": SWIPE_MIN_VELOCITY,
        "min_duration": SWIPE_MIN_DURATION_SECONDS,
        "max_duration": SWIPE_MAX_DURATION_SECONDS,
        "max_anchor_speed": MAX_ANCHOR_SPEED,
        "smoothing_alpha": SMOOTHING_ALPHA,
    }
    if MOTION_PROFILE_PATH.exists():
        try:
            with open(MOTION_PROFILE_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
                defaults.update(saved)
        except Exception as e:
            print(f"[WARN] Failed to load {MOTION_PROFILE_PATH}: {e}")
    return defaults


def save_motion_profile(profile: Dict[str, Any]) -> None:
    """Saves calibrated personal motion profile to data/motion_profile.json."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(MOTION_PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    print(f"[CONFIG] Saved motion profile to {MOTION_PROFILE_PATH}")


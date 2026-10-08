"""
Cross-platform camera capture wrapper.
Automatically selects and tries backends for Windows (MSMF, DSHOW), macOS (AVFOUNDATION),
and Linux (V4L2). Includes safe property assignment and auto-detection fallback.
"""

import sys
import platform
import cv2
from typing import Optional, Tuple, List
from src.config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS_TARGET


class Camera:
    """
    Cross-platform webcam wrapper with graceful fallbacks and safe configuration.
    """

    def __init__(self, camera_index: int = CAMERA_INDEX, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.os_name = platform.system()
        self.active_backend: Optional[int] = None

    def _get_candidate_backends(self) -> List[Tuple[str, int]]:
        """Returns prioritized list of (name, backend_id) based on current OS."""
        if self.os_name == "Windows":
            return [
                ("DSHOW (DirectShow)", cv2.CAP_DSHOW),
                ("MSMF (Media Foundation)", cv2.CAP_MSMF),
                ("ANY (Default)", cv2.CAP_ANY),
            ]
        elif self.os_name == "Darwin":
            return [
                ("AVFOUNDATION", cv2.CAP_AVFOUNDATION),
                ("ANY (Default)", cv2.CAP_ANY),
            ]
        else:
            return [
                ("V4L2", cv2.CAP_V4L2),
                ("ANY (Default)", cv2.CAP_ANY),
            ]

    def open(self) -> bool:
        """
        Attempts to open the camera using prioritized backends.
        If the configured index fails, tries other camera indices automatically.
        """
        # First attempt: user-selected camera_index
        if self._try_open_index(self.camera_index):
            return True

        # Second attempt: check other camera indices (0..4) if configured index failed
        print(f"[INFO] Camera index {self.camera_index} failed. Searching for other working cameras...")
        for alt_idx in range(5):
            if alt_idx == self.camera_index:
                continue
            if self._try_open_index(alt_idx):
                print(f"[INFO] Successfully switched to working camera index {alt_idx}!")
                self.camera_index = alt_idx
                return True

        self._print_troubleshooting_tips()
        return False

    def _try_open_index(self, index: int) -> bool:
        """Attempts to open a specific camera index with each available backend."""
        backends = self._get_candidate_backends()

        for name, backend in backends:
            # Always ensure previous capture is closed before creating a new one
            self.release()

            try:
                cap = cv2.VideoCapture(index, backend)
                if cap is not None and cap.isOpened():
                    # Apply MJPEG and resolution BEFORE the first read
                    self._apply_safe_properties(cap)

                    # Attempt a test read to ensure frames can actually be grabbed
                    try:
                        ret, test_frame = cap.read()
                    except Exception:
                        ret, test_frame = False, None

                    if ret and test_frame is not None:
                        self.cap = cap
                        self.active_backend = backend
                        print(f"[CAMERA] Connected to camera {index} using backend {name}")
                        return True
                    else:
                        cap.release()
                elif cap is not None:
                    cap.release()
            except Exception as e:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass

        return False

    def _apply_safe_properties(self, cap: Optional[cv2.VideoCapture] = None) -> None:
        """Sets frame dimensions, MJPEG codec, and target FPS before starting stream."""
        target_cap = cap if cap is not None else self.cap
        if target_cap is None:
            return

        # Prefer MJPG on Windows to avoid uncompressed stride assertion errors (_step >= minstep)
        if self.os_name == "Windows":
            try:
                target_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass

        for prop_id, val, desc in [
            (cv2.CAP_PROP_FRAME_WIDTH, self.width, "width"),
            (cv2.CAP_PROP_FRAME_HEIGHT, self.height, "height"),
            (cv2.CAP_PROP_FPS, FPS_TARGET, "fps"),
            (cv2.CAP_PROP_BUFFERSIZE, 1, "buffersize"),
        ]:
            try:
                target_cap.set(prop_id, val)
            except Exception as err:
                pass

    def read_frame(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        Reads and returns a frame safely.
        Flips horizontally for a natural mirror view.
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return False, None

            # Flip horizontally for intuitive selfie-view mirror effect
            frame = cv2.flip(frame, 1)
            return True, frame
        except Exception as e:
            # Catch transient driver/stride glitches without crashing the thread
            return False, None

    def release(self) -> None:
        """Releases the webcam resources."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _print_troubleshooting_tips(self) -> None:
        print("\n" + "=" * 60)
        print(f"[ERROR] Could not open any working camera (tried index {self.camera_index} and fallbacks).")
        if self.os_name == "Darwin":
            print("macOS Troubleshooting:")
            print(" 1. Check Camera Permission: System Settings > Privacy & Security > Camera")
            print("    Ensure your terminal or IDE is allowed.")
            print(" 2. Make sure no other app (FaceTime, Zoom, Photo Booth) is locking the camera.")
        elif self.os_name == "Windows":
            print("Windows Troubleshooting:")
            print(" 1. Check Settings > Privacy & security > Camera > 'Let desktop apps access your camera'.")
            print(" 2. Make sure another app (Teams, Zoom, Discord, OBS) isn't exclusively locking the camera.")
            print(" 3. Pass a specific camera index with: --camera 0 (or --camera 1, 2)")
        print("=" * 60 + "\n")

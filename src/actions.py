"""
Cross-platform macro and hotkey dispatcher.
Handles key translations between Windows and macOS, action debounce, and cooldowns.
"""

import time
import platform
import subprocess
from typing import Dict, Any, Optional
import pyautogui
from src.config import IS_MAC, ACTION_COOLDOWN_SECONDS

# Configure PyAutoGUI safety settings
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05


class ActionDispatcher:
    """
    Executes macros and hotkeys across Windows and macOS with built-in cooldown tracking.
    """

    def __init__(
        self,
        cooldown_seconds: float = ACTION_COOLDOWN_SECONDS,
        global_cooldown: float = 0.40,
        dry_run: bool = False
    ):
        self.cooldown_seconds = cooldown_seconds
        self.global_cooldown = 0.0 if dry_run else global_cooldown
        self.dry_run = dry_run
        self.last_action_time: Dict[str, float] = {}
        self.last_global_action_time: float = 0.0
        self.is_mac = platform.system() == "Darwin"

    def can_trigger(self, action_id: str) -> bool:
        """
        Checks if the given action ID is past its cooldown period.
        Enforces:
        1. Global minimum debounce to prevent double-firing macros.
        2. Per-gesture cooldown.
        3. Opposing-gesture lockout (e.g. swipe_right suppresses swipe_left return stroke).
        """
        now = time.time()

        # 1. Global action cooldown
        if self.global_cooldown > 0 and (now - self.last_global_action_time) < self.global_cooldown:
            return False

        # 2. Per-action cooldown
        last_time = self.last_action_time.get(action_id, 0.0)
        if (now - last_time) < self.cooldown_seconds:
            return False

        # 3. Opposing action lockout (e.g. moving hand back to center after swipe_right)
        opposing = {
            "swipe_right": "swipe_left",
            "swipe_left": "swipe_right",
            "swipe_up": "swipe_down",
            "swipe_down": "swipe_up",
        }
        opp_id = opposing.get(action_id)
        if opp_id and (now - self.last_action_time.get(opp_id, 0.0)) < self.cooldown_seconds:
            return False

        return True

    def execute(self, gesture_name: str, action_config: Dict[str, Any]) -> bool:
        """
        Executes an action configuration dictionary if cooldown permits.

        Returns:
            True if action was fired, False if blocked by cooldown or invalid config.
        """
        if not self.can_trigger(gesture_name):
            return False

        action_type = action_config.get("type", "hotkey")
        description = action_config.get("description", gesture_name)

        print(f"[ACTION] Triggering '{description}' for gesture '{gesture_name}'...")

        now_t = time.time()
        if self.dry_run:
            print(f"[DRY-RUN] Executed: {action_config}")
            self.last_action_time[gesture_name] = now_t
            self.last_global_action_time = now_t
            return True

        success = False
        try:
            if action_type == "hotkey":
                keys = action_config.get("keys", [])
                translated_keys = [self._translate_key(k) for k in keys]
                pyautogui.hotkey(*translated_keys)
                success = True

            elif action_type == "media":
                key = action_config.get("key", "")
                pyautogui.press(key)
                success = True

            elif action_type == "scroll":
                amount = action_config.get("amount", 300)
                # PyAutoGUI on macOS uses smaller scroll increments
                if self.is_mac and abs(amount) > 50:
                    amount = int(amount / 30)
                pyautogui.scroll(amount)
                success = True

            elif action_type == "command":
                cmd = action_config.get("command", "")
                if cmd:
                    subprocess.Popen(cmd, shell=True)
                    success = True

            else:
                print(f"[WARN] Unknown action type: {action_type}")

        except Exception as e:
            print(f"[ERROR] Failed to execute action for '{gesture_name}': {e}")
            success = False

        if success:
            finish_time = time.time()
            self.last_action_time[gesture_name] = finish_time
            self.last_global_action_time = finish_time

        return success

    def _translate_key(self, key: str) -> str:
        """
        Translates generic modifier keys to platform-specific key names.
        """
        key_lower = key.lower()
        if self.is_mac:
            # Map 'ctrl' to 'command' when standard OS shortcut is expected
            if key_lower == "cmd" or key_lower == "command":
                return "command"
            if key_lower == "win":
                return "command"
        else:
            if key_lower == "cmd" or key_lower == "command":
                return "ctrl"
            if key_lower == "win":
                return "win"

        return key_lower

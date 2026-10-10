# Gesture Macro Studio (Upper-Body & Hands)

Control your computer, media, and hotkeys using custom body and hand gestures detected in real-time from your webcam. Runs 100% locally at high frame rates on both **macOS** and **Windows** without requiring a GPU.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![PySide6](https://img.shields.io/badge/PySide6_(Qt)-41CD52?logo=qt&logoColor=white)

## 🧠 How It Works

```
Webcam ─► MediaPipe pose + hand landmarks ─► chest-relative normalization
       ─► Random Forest classifier (poses) / motion learner (swipes, circles)
       ─► debounce + cooldown ─► keystroke, media key, or shell command
```

You record your own samples for each gesture in the Studio, retrain in about a second, and map each gesture to an action. The default model and mappings ship in `models/` and `data/` so it works out of the box.

---

## 📦 Installation

Requires Python 3.10+ and a webcam.

```bash
git clone https://github.com/colbygross/gesture-macro.git
cd gesture-macro
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or use the launchers, which set up the environment for you: `./run_mac.sh` on macOS, `run_windows.bat` on Windows.

---

## ⚡ Key Highlights
- **Upper-Body & Arm Rig**: Tracks head, shoulders, elbows, wrists, and torso alongside detailed 21-point finger joints.
- **Chest-Relative Normalization**: Arm gestures (e.g. *arms crossed*, *hand to chin*, *raising hand*, *pointing*) are scale- and distance-invariant.
- **Gesture Macro Studio GUI**: Complete desktop application built with PySide6 for live preview, 3-2-1 sample recording, 1-click model retraining, and visual macro configuration.
- **Debouncing & Safety**: Configurable frame-hold threshold and cooldowns prevent accidental triggers.
- **Cross-Platform**: Automatic camera backend selection (MSMF/DirectShow on Windows, AVFoundation on macOS) and smart modifier key translation (`Cmd` on macOS, `Ctrl` on Windows).

---

## 🚀 Launching the Gesture Macro Studio (Recommended)

Start the unified desktop application:

```bash
python -m src.studio
```

### What You Can Do in the Studio:
1. **Live Rig Preview**: See your shoulders, arms, elbows, and fingers color-coded in real time.
2. **Gesture Library**: View your recorded dataset with sample counts per class.
3. **Record Gestures**: Type a gesture name (e.g., `arms_crossed`, `hand_to_chin`, `raise_hand`, `fist`), click **Record**, and a 3-2-1 on-screen countdown captures 100 samples.
4. **1-Click Retrain**: Click **Retrain Model** to train an updated classifier in ~1 second. The live model hot-reloads instantly without restarting the app.
5. **Macro Mappings Matrix**: Edit and save custom keystrokes (`Ctrl+Shift+M`), media keys (`playpause`, `volumeup`), or shell commands.

---

## 💻 Command Line Tools (Alternative)

If you prefer using the terminal directly:

### 1. Record Gestures
```bash
# Record upper-body gesture
python -m src.collect_data --gesture arms_crossed --samples 100

# Record hand gesture
python -m src.collect_data --gesture open_palm --samples 100
```

### 2. Train Model
```bash
python -m src.train
```

### 3. Run Real-Time Headless / Overlay
```bash
# Dry-run mode (safely logs actions to terminal without firing simulated keys)
python -m src.main --dry-run

# Live detection mode
python -m src.main
```

---

## ⚙️ Default Macro Actions

Stored in `data/macros.json` and configurable in the Studio:

### Static Poses (Upper-Body & Hands)
| Gesture | Type | Key / Action | Description |
| :--- | :--- | :--- | :--- |
| **`open_palm`** | media | `playpause` | Play / Pause Media |
| **`fist`** | hotkey | `Ctrl+Shift+M` (`Cmd+Shift+M`) | Mute / Unmute Microphone |
| **`arms_crossed`** | hotkey | `Win+D` (`Cmd+F3`) | Show Desktop / Minimize All |
| **`hand_to_chin`** | hotkey | `Ctrl+Shift+M` (`Cmd+Shift+M`) | Shh / Mute Mic |
| **`raise_hand`** | hotkey | `Ctrl+Shift+K` (`Cmd+Shift+K`) | Raise Hand (Teams/Meet) |
| **`thumbs_up`** | media | `volumeup` | Increase System Volume |
| **`thumbs_down`** | media | `volumedown` | Decrease System Volume |
| **`peace`** | hotkey | `Win+Shift+S` (`Cmd+Shift+4`) | Snipping Tool / Screenshot |

### Dynamic Motion Gestures (Kinematic Swipes & Circles)
| Gesture | Type | Key / Action | Description |
| :--- | :--- | :--- | :--- |
| **`swipe_right`** | hotkey | `Alt+Tab` (`Cmd+Tab`) | Switch App / Window |
| **`swipe_left`** | hotkey | `Shift+Alt+Tab` (`Shift+Cmd+Tab`) | Previous App / Window |
| **`swipe_up`** | scroll | `scroll 300` | Smooth Scroll Up |
| **`swipe_down`** | scroll | `scroll -300` | Smooth Scroll Down |
| **`circle_cw`** | media | `volumeup` | Rotary Dial Volume Up (Clockwise) |
| **`circle_ccw`** | media | `volumedown` | Rotary Dial Volume Down (Counter-Clockwise) |

---

## 🧪 Tests

```bash
python -m unittest discover tests
```

The suite covers the motion tracker, the motion learner, upper-body normalization, and the full detection pipeline.

---

## 🛠️ How It Was Built

I built this project with AI coding agents, directing the work feature by feature (rig tracking, motion learning, the Studio GUI) and using the test suite to check each change before moving on.

---

## 🍎 macOS Permissions Note
When running on macOS, make sure your terminal or IDE has:
1. **Camera Permission**: *System Settings > Privacy & Security > Camera*
2. **Accessibility Permission**: *System Settings > Privacy & Security > Accessibility* (required to simulate keypresses).

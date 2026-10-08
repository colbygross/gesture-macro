"""
Unified launcher for Gesture Macro Controller and Studio.
Usage:
    python main.py          -> Launches Gesture Studio GUI (default)
    python main.py studio   -> Launches Gesture Studio GUI
    python main.py run      -> Runs OpenCV CLI HUD runner
    python main.py train    -> Runs model trainer
    python main.py record   -> Runs dataset recorder
"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))


def main():
    mode = "studio"
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        mode = sys.argv[1].lower()
        sys.argv.pop(1)

    if mode in ("studio", "gui"):
        from src.studio import main as studio_main
        studio_main()
    elif mode in ("run", "cli", "app"):
        from src.main import main as app_main
        app_main()
    elif mode == "train":
        from src.train import train_gesture_classifier
        train_gesture_classifier()
    elif mode == "record":
        from src.collect_data import main as record_main
        record_main()
    else:
        from src.studio import main as studio_main
        studio_main()


if __name__ == "__main__":
    main()

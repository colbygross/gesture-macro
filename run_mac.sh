#!/bin/bash
# ==============================================================================
# Setup and Launch script for Gesture Macro Studio on macOS (Apple Silicon / Intel)
# ==============================================================================

set -e

# Change to the script's directory
cd "$(dirname "$0")"

echo "============================================================"
echo "  Gesture Macro Studio - macOS Launcher"
echo "============================================================"

# Check for python3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found."
    echo "Please install Python 3.10+ from python.org or via Homebrew: brew install python"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[SETUP] Creating isolated Python virtual environment (venv)..."
    python3 -m venv venv
    source venv/bin/activate
    echo "[SETUP] Upgrading pip..."
    pip install --upgrade pip
    echo "[SETUP] Installing dependencies (MediaPipe, OpenCV, PySide6, scikit-learn)..."
    pip install -r requirements.txt
    echo "[SETUP] Dependencies installed successfully!"
else
    source venv/bin/activate
fi

echo ""
echo "============================================================"
echo " IMPORTANT macOS PERMISSIONS NOTICE:"
echo " 1. Camera Permission: If prompted, allow Terminal / VS Code to access Camera."
echo "    (System Settings > Privacy & Security > Camera)"
echo " 2. Accessibility Permission: Required to trigger macro keyboard shortcuts."
echo "    (System Settings > Privacy & Security > Accessibility)"
echo "============================================================"
echo ""

echo "[LAUNCH] Starting Gesture Macro Studio..."
python -m src.studio

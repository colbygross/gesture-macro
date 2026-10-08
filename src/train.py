"""
Machine learning classifier trainer for gesture recognition.
Trains on normalized landmark features (supports Upper-Body Rig and Hands-Only).
Provides programmatic training functions for GUI integration and CLI execution.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GESTURES_CSV_PATH, MODEL_PATH, MODELS_DIR


def train_gesture_classifier(
    csv_path: Path = GESTURES_CSV_PATH,
    output_model_path: Path = MODEL_PATH,
    model_type: str = "rf",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Loads dataset, trains classifier, prints evaluation metrics, and saves the trained model.
    Returns a dictionary with training results for CLI and GUI display.
    """
    if not csv_path.exists():
        err = f"Dataset not found at: {csv_path}. Please record gestures first."
        print(f"[ERROR] {err}")
        return {"success": False, "error": err}

    print(f"[DATA] Loading dataset from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        err = f"Failed to read dataset: {e}"
        print(f"[ERROR] {err}")
        return {"success": False, "error": err}

    if len(df) == 0:
        err = "Dataset is empty."
        print(f"[ERROR] {err}")
        return {"success": False, "error": err}

    labels = df["label"]
    features = df.drop(columns=["label"])

    unique_classes = sorted(labels.unique())
    print(f"[DATA] Found {len(unique_classes)} unique classes: {unique_classes}")
    print(f"[DATA] Total samples: {len(df)}")
    class_counts = dict(labels.value_counts())
    print("\nSamples per class:")
    for cls_name, count in class_counts.items():
        print(f"  - {cls_name}: {count} samples")

    if len(unique_classes) < 2:
        err = f"Need at least 2 distinct gestures in dataset (currently have {len(unique_classes)}: {unique_classes}). Record another gesture first!"
        print(f"\n[ERROR] {err}")
        return {"success": False, "error": err, "classes": unique_classes, "class_counts": class_counts}

    X = features.values
    y = labels.values

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Initialize model
    if model_type == "mlp":
        print("\n[TRAIN] Training Multi-Layer Perceptron (MLP) Classifier...")
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=random_state)
    else:
        print("\n[TRAIN] Training Random Forest Classifier (100 trees)...")
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)

    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    print("\n" + "=" * 60)
    print(f"  TRAINING RESULTS ({model_type.upper()})")
    print(f"  Overall Test Accuracy: {acc * 100:.2f}%")
    print("=" * 60)
    report_str = classification_report(y_test, y_pred)
    print("\nClassification Report:\n")
    print(report_str)

    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    cm_df = pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)
    print("Confusion Matrix:")
    print(cm_df)

    # Save model and metadata
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": clf,
        "classes": list(clf.classes_),
        "feature_count": X.shape[1],
        "model_type": model_type,
    }
    joblib.dump(payload, output_model_path)
    print(f"\n[SAVE] Model successfully saved to: {output_model_path}")

    return {
        "success": True,
        "accuracy": acc,
        "classes": list(clf.classes_),
        "class_counts": class_counts,
        "total_samples": len(df),
        "feature_count": X.shape[1],
        "report": report_str,
        "model": clf,
        "model_path": str(output_model_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Train a gesture classifier on landmark coordinates.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(GESTURES_CSV_PATH),
        help="Path to the gestures CSV dataset file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(MODEL_PATH),
        help="Path to save the trained model .joblib file.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["rf", "mlp"],
        default="rf",
        help="Classifier type: 'rf' (Random Forest) or 'mlp' (Neural Network). Default: rf.",
    )
    args = parser.parse_args()

    result = train_gesture_classifier(
        csv_path=Path(args.dataset),
        output_model_path=Path(args.output),
        model_type=args.model_type,
    )

    if not result.get("success"):
        sys.exit(1)


if __name__ == "__main__":
    main()

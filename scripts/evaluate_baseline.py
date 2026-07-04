from __future__ import annotations

import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)

from src.config import (
    EMOTION_NAMES,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
)
from src.processed_loader import load_processed_data


def evaluate_baseline_model() -> None:
    model_path = MODELS_DIR / "baseline_cnn_best.keras"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Baseline model was not found at: {model_path}\n"
            "Run this command first:\n"
            "python -m scripts.train_baseline --epochs 25 --batch-size 64"
        )

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print("Loading processed FER2013 data...")
    splits = load_processed_data()

    print("Evaluating model on the test set...")
    test_loss, test_accuracy = model.evaluate(
        splits.X_test,
        splits.y_test,
        batch_size=64,
        verbose=1,
    )

    print("Generating predictions...")
    probabilities = model.predict(
        splits.X_test,
        batch_size=64,
        verbose=1,
    )

    y_pred = np.argmax(probabilities, axis=1)
    y_true = splits.y_test

    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy_from_keras": float(test_accuracy),
        "test_accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }

    metrics_path = METRICS_DIR / "baseline_cnn_test_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"Test metrics saved to: {metrics_path}")

    report = classification_report(
        y_true,
        y_pred,
        target_names=EMOTION_NAMES,
        zero_division=0,
    )

    report_path = METRICS_DIR / "baseline_cnn_classification_report.txt"

    with report_path.open("w", encoding="utf-8") as file:
        file.write(report)

    print(f"Classification report saved to: {report_path}")

    print("\nClassification report:")
    print(report)

    cm = confusion_matrix(y_true, y_pred)

    cm_path = FIGURES_DIR / "baseline_cnn_confusion_matrix.png"

    fig, ax = plt.subplots(figsize=(9, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=EMOTION_NAMES,
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d",
        colorbar=True,
    )

    ax.set_title("Baseline CNN Confusion Matrix on FER2013 Test Set")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Confusion matrix saved to: {cm_path}")

    print("\nFinal test metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


def main() -> None:
    evaluate_baseline_model()


if __name__ == "__main__":
    main()
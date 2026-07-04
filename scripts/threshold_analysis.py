from __future__ import annotations

import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
)
from src.processed_loader import load_processed_data
from src.thresholding import (
    calculate_threshold_metrics,
    threshold_metrics_to_dict,
)


def load_selected_model() -> tf.keras.Model:
    """
    Load the selected final model for threshold analysis.

    The selected model is the improved CNN without class weights because it
    achieved the best test accuracy and weighted F1-score.
    """

    model_path = MODELS_DIR / "improved_cnn_no_class_weights_best.keras"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Selected model was not found at: {model_path}\n"
            "Run this command first:\n"
            "python -m scripts.train_improved_no_class_weights --epochs 25 --batch-size 64"
        )

    print(f"Loading selected model from: {model_path}")

    return tf.keras.models.load_model(model_path)


def run_threshold_analysis() -> pd.DataFrame:
    """
    Run confidence-threshold analysis on the validation set.
    """

    model = load_selected_model()

    print("Loading processed FER2013 data...")
    splits = load_processed_data()

    print("Generating validation probabilities...")
    probabilities = model.predict(
        splits.X_val,
        batch_size=64,
        verbose=1,
    )

    thresholds = np.round(
        np.arange(0.40, 0.91, 0.05),
        2,
    )

    rows = []

    for threshold in thresholds:
        metrics = calculate_threshold_metrics(
            y_true=splits.y_val,
            probabilities=probabilities,
            threshold=float(threshold),
        )

        rows.append(threshold_metrics_to_dict(metrics))

    results = pd.DataFrame(rows)

    return results


def select_recommended_threshold(
    results: pd.DataFrame,
    minimum_accepted_accuracy: float = 0.80,
    minimum_coverage: float = 0.40,
) -> dict:
    """
    Select a practical recommended threshold.

    Strategy:
        1. Keep thresholds with accepted_accuracy >= minimum_accepted_accuracy.
        2. Keep thresholds with coverage >= minimum_coverage.
        3. Among valid candidates, choose the one with highest coverage.
        4. If tied, choose the one with higher accepted accuracy.

    This avoids selecting overly strict thresholds such as 0.90,
    which may have very high accepted accuracy but reject too many samples.
    """

    candidates = results[
        (results["accepted_accuracy"] >= minimum_accepted_accuracy)
        & (results["coverage"] >= minimum_coverage)
    ].copy()

    if candidates.empty:
        best_row = results.sort_values(
            by=["accepted_accuracy", "coverage"],
            ascending=[False, False],
        ).iloc[0]

        selection_note = (
            "No threshold satisfied both constraints. "
            "Fallback selected the highest accepted accuracy."
        )
    else:
        best_row = candidates.sort_values(
            by=["coverage", "accepted_accuracy"],
            ascending=[False, False],
        ).iloc[0]

        selection_note = (
            "Selected highest-coverage threshold satisfying "
            "minimum accepted accuracy and minimum coverage."
        )

    recommended = {
        "recommended_threshold": float(best_row["threshold"]),
        "accepted_accuracy": float(best_row["accepted_accuracy"]),
        "coverage": float(best_row["coverage"]),
        "rejection_rate": float(best_row["rejection_rate"]),
        "raw_accuracy": float(best_row["raw_accuracy"]),
        "minimum_accepted_accuracy_rule": float(minimum_accepted_accuracy),
        "minimum_coverage_rule": float(minimum_coverage),
        "selection_note": selection_note,
    }

    return recommended


def save_threshold_results(
    results: pd.DataFrame,
    recommended: dict,
) -> None:
    """
    Save threshold analysis results.
    """

    csv_path = METRICS_DIR / "threshold_analysis_validation.csv"
    json_path = METRICS_DIR / "threshold_analysis_validation.json"
    recommended_path = METRICS_DIR / "recommended_threshold.json"

    results.to_csv(csv_path, index=False)

    results.to_json(
        json_path,
        orient="records",
        indent=4,
    )

    with recommended_path.open("w", encoding="utf-8") as file:
        json.dump(recommended, file, indent=4)

    print(f"Threshold analysis CSV saved to: {csv_path}")
    print(f"Threshold analysis JSON saved to: {json_path}")
    print(f"Recommended threshold saved to: {recommended_path}")


def plot_threshold_analysis(results: pd.DataFrame) -> None:
    """
    Plot threshold, accepted accuracy, coverage, and rejection rate.
    """

    output_path = FIGURES_DIR / "threshold_analysis_validation.png"

    plt.figure(figsize=(10, 6))

    plt.plot(
        results["threshold"],
        results["accepted_accuracy"],
        marker="o",
        label="Accepted Accuracy",
    )

    plt.plot(
        results["threshold"],
        results["coverage"],
        marker="o",
        label="Coverage",
    )

    plt.plot(
        results["threshold"],
        results["rejection_rate"],
        marker="o",
        label="Rejection Rate",
    )

    plt.title("Confidence Threshold Analysis on Validation Set")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Threshold analysis chart saved to: {output_path}")


def main() -> None:
    results = run_threshold_analysis()

    recommended = select_recommended_threshold(
    results,
    minimum_accepted_accuracy=0.80,
    minimum_coverage=0.40,
)

    rounded_results = results.copy()

    numeric_columns = [
        "threshold",
        "coverage",
        "rejection_rate",
        "accepted_accuracy",
        "raw_accuracy",
    ]

    rounded_results[numeric_columns] = rounded_results[numeric_columns].round(4)

    print("\nThreshold analysis results:")
    print(rounded_results)

    print("\nRecommended threshold:")
    for key, value in recommended.items():
        if isinstance(value, str):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.4f}")

    save_threshold_results(
        results=results,
        recommended=recommended,
    )

    plot_threshold_analysis(results)


if __name__ == "__main__":
    main()
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, METRICS_DIR


MODEL_CONFIGS = {
    "Baseline CNN": {
        "metrics_file": "baseline_cnn_test_metrics.json",
    },
    "Improved CNN + Class Weights": {
        "metrics_file": "improved_cnn_test_metrics.json",
    },
    "Improved CNN No Class Weights": {
        "metrics_file": "improved_cnn_no_class_weights_test_metrics.json",
    },
}


def load_metrics_file(file_name: str) -> dict:
    """
    Load a test metrics JSON file from the metrics directory.
    """

    metrics_path = METRICS_DIR / file_name

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file was not found: {metrics_path}"
        )

    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    return metrics


def build_comparison_table() -> pd.DataFrame:
    """
    Build a comparison table for all trained models.
    """

    rows = []

    for model_name, config in MODEL_CONFIGS.items():
        metrics = load_metrics_file(config["metrics_file"])

        rows.append(
            {
                "Model": model_name,
                "Test Accuracy": metrics["test_accuracy"],
                "Macro Precision": metrics["macro_precision"],
                "Macro Recall": metrics["macro_recall"],
                "Macro F1": metrics["macro_f1"],
                "Weighted F1": metrics["weighted_f1"],
                "Test Loss": metrics["test_loss"],
            }
        )

    comparison = pd.DataFrame(rows)

    return comparison


def save_comparison_outputs(comparison: pd.DataFrame) -> None:
    """
    Save comparison results as CSV and JSON.
    """

    csv_path = METRICS_DIR / "model_comparison.csv"
    json_path = METRICS_DIR / "model_comparison.json"

    comparison.to_csv(csv_path, index=False)

    comparison.to_json(
        json_path,
        orient="records",
        indent=4,
    )

    print(f"Comparison CSV saved to: {csv_path}")
    print(f"Comparison JSON saved to: {json_path}")


def plot_metric_comparison(comparison: pd.DataFrame) -> None:
    """
    Plot grouped bar chart for key evaluation metrics.
    """

    output_path = FIGURES_DIR / "model_comparison_metrics.png"

    plot_df = comparison[
        [
            "Model",
            "Test Accuracy",
            "Macro F1",
            "Weighted F1",
        ]
    ].copy()

    ax = plot_df.set_index("Model").plot(
        kind="bar",
        figsize=(11, 6),
        rot=20,
    )

    ax.set_title("Model Performance Comparison on FER2013 Test Set")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.75)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Model comparison chart saved to: {output_path}")


def print_best_models(comparison: pd.DataFrame) -> None:
    """
    Print best model according to each main metric.
    """

    metrics = [
        "Test Accuracy",
        "Macro F1",
        "Weighted F1",
    ]

    print("\nBest model by metric:")

    for metric in metrics:
        best_row = comparison.loc[comparison[metric].idxmax()]

        print(
            f"{metric}: {best_row['Model']} "
            f"({best_row[metric]:.4f})"
        )


def main() -> None:
    comparison = build_comparison_table()

    rounded_comparison = comparison.copy()

    numeric_columns = [
        "Test Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Weighted F1",
        "Test Loss",
    ]

    rounded_comparison[numeric_columns] = rounded_comparison[
        numeric_columns
    ].round(4)

    print("\nModel comparison:")
    print(rounded_comparison)

    save_comparison_outputs(comparison)
    plot_metric_comparison(comparison)
    print_best_models(comparison)


if __name__ == "__main__":
    main()
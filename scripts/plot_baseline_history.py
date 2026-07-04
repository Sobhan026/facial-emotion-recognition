from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR, METRICS_DIR


def plot_accuracy_curve(history: pd.DataFrame) -> None:
    """
    Plot training and validation accuracy curves.
    """

    output_path = FIGURES_DIR / "baseline_cnn_accuracy_curve.png"

    epochs = range(1, len(history) + 1)

    plt.figure(figsize=(9, 6))

    plt.plot(
        epochs,
        history["accuracy"],
        marker="o",
        label="Training Accuracy",
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        marker="o",
        label="Validation Accuracy",
    )

    plt.title("Baseline CNN Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(list(epochs))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Accuracy curve saved to: {output_path}")


def plot_loss_curve(history: pd.DataFrame) -> None:
    """
    Plot training and validation loss curves.
    """

    output_path = FIGURES_DIR / "baseline_cnn_loss_curve.png"

    epochs = range(1, len(history) + 1)

    plt.figure(figsize=(9, 6))

    plt.plot(
        epochs,
        history["loss"],
        marker="o",
        label="Training Loss",
    )

    plt.plot(
        epochs,
        history["val_loss"],
        marker="o",
        label="Validation Loss",
    )

    plt.title("Baseline CNN Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(list(epochs))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Loss curve saved to: {output_path}")


def print_training_summary(history: pd.DataFrame) -> None:
    """
    Print a concise summary of the baseline training run.
    """

    best_val_accuracy_epoch = int(history["val_accuracy"].idxmax() + 1)
    best_val_accuracy = float(history["val_accuracy"].max())

    best_val_loss_epoch = int(history["val_loss"].idxmin() + 1)
    best_val_loss = float(history["val_loss"].min())

    final_epoch = len(history)
    final_train_accuracy = float(history.iloc[-1]["accuracy"])
    final_val_accuracy = float(history.iloc[-1]["val_accuracy"])
    final_train_loss = float(history.iloc[-1]["loss"])
    final_val_loss = float(history.iloc[-1]["val_loss"])

    print("\nBaseline CNN training summary:")
    print(f"Total epochs completed: {final_epoch}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f} at epoch {best_val_accuracy_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_val_loss_epoch}")
    print(f"Final training accuracy: {final_train_accuracy:.4f}")
    print(f"Final validation accuracy: {final_val_accuracy:.4f}")
    print(f"Final training loss: {final_train_loss:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}")


def main() -> None:
    history_path = METRICS_DIR / "baseline_cnn_history.csv"

    if not history_path.exists():
        raise FileNotFoundError(
            f"Training history was not found at: {history_path}\n"
            "Run this command first:\n"
            "python -m scripts.train_baseline --epochs 25 --batch-size 64"
        )

    history = pd.read_csv(history_path)

    required_columns = {
        "accuracy",
        "loss",
        "val_accuracy",
        "val_loss",
    }

    missing_columns = required_columns.difference(history.columns)

    if missing_columns:
        raise ValueError(
            f"Training history is missing required columns: {sorted(missing_columns)}"
        )

    print_training_summary(history)
    plot_accuracy_curve(history)
    plot_loss_curve(history)


if __name__ == "__main__":
    main()
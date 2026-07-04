from __future__ import annotations

import argparse
import json
import os

# Reduce TensorFlow log noise.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
)
from src.models import build_improved_cnn_model
from src.processed_loader import load_class_weights, load_processed_data


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set random seeds for reproducible experiments.
    """

    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_callbacks(
    model_name: str,
) -> list[tf.keras.callbacks.Callback]:
    """
    Create training callbacks for the improved model.
    """

    best_model_path = MODELS_DIR / f"{model_name}_best.keras"
    csv_log_path = METRICS_DIR / f"{model_name}_training_log.csv"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=csv_log_path,
            append=False,
        ),
    ]

    return callbacks


def save_training_history(
    history: tf.keras.callbacks.History,
    model_name: str,
) -> None:
    """
    Save training history as CSV and JSON.
    """

    history_dict = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    history_csv_path = METRICS_DIR / f"{model_name}_history.csv"
    history_json_path = METRICS_DIR / f"{model_name}_history.json"

    history_df = pd.DataFrame(history_dict)
    history_df.to_csv(history_csv_path, index=False)

    with history_json_path.open("w", encoding="utf-8") as file:
        json.dump(history_dict, file, indent=4)

    print(f"Training history saved to: {history_csv_path}")
    print(f"Training history JSON saved to: {history_json_path}")


def save_model_summary(
    model: tf.keras.Model,
    model_name: str,
) -> None:
    """
    Save model architecture summary to a text file.
    """

    summary_path = METRICS_DIR / f"{model_name}_summary.txt"

    lines: list[str] = []
    model.summary(print_fn=lambda line: lines.append(line))

    with summary_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    print(f"Model summary saved to: {summary_path}")


def train_improved_model(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    use_class_weights: bool = True,
) -> tf.keras.Model:
    """
    Train the improved CNN model on the processed FER2013 dataset.
    """

    model_name = "improved_cnn"

    print("Setting random seed...")
    set_global_seed(RANDOM_SEED)

    print("Loading processed FER2013 data...")
    splits = load_processed_data()

    print("Building improved CNN model...")
    model = build_improved_cnn_model()

    save_model_summary(model, model_name)

    callbacks = create_callbacks(model_name)

    class_weight = None

    if use_class_weights:
        print("Loading class weights...")
        class_weight = load_class_weights()
        print(f"Class weights: {class_weight}")
    else:
        print("Class weights are disabled for this improved run.")

    print("\nTraining configuration:")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Use class weights: {use_class_weights}")
    print("Data augmentation: enabled inside the model during training")

    history = model.fit(
        splits.X_train,
        splits.y_train,
        validation_data=(splits.X_val, splits.y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    final_model_path = MODELS_DIR / f"{model_name}_final.keras"
    model.save(final_model_path)

    print(f"\nFinal model saved to: {final_model_path}")

    save_training_history(history, model_name)

    print("\nEvaluating final model on the validation set...")
    val_loss, val_accuracy = model.evaluate(
        splits.X_val,
        splits.y_val,
        batch_size=batch_size,
        verbose=1,
    )

    validation_results = {
        "val_loss": float(val_loss),
        "val_accuracy": float(val_accuracy),
        "epochs_requested": int(epochs),
        "batch_size": int(batch_size),
        "use_class_weights": bool(use_class_weights),
        "data_augmentation": True,
    }

    validation_results_path = METRICS_DIR / f"{model_name}_validation_results.json"

    with validation_results_path.open("w", encoding="utf-8") as file:
        json.dump(validation_results, file, indent=4)

    print(f"Validation results saved to: {validation_results_path}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")

    return model


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Train the improved CNN model on FER2013."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Training batch size.",
    )

    parser.add_argument(
        "--disable-class-weights",
        action="store_true",
        help="Disable class weights during improved model training.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    train_improved_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_class_weights=not args.disable_class_weights,
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import (
    IMAGE_SHAPE,
    NUMBER_OF_CLASSES,
    PROCESSED_DATA_DIR,
)
from src.dataset import FER2013Splits


PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "fer2013_processed.npz"
CLASS_WEIGHTS_PATH = PROCESSED_DATA_DIR / "class_weights.json"
PROCESSED_SUMMARY_PATH = PROCESSED_DATA_DIR / "processed_summary.csv"


def load_processed_data(
    processed_data_path=PROCESSED_DATA_PATH,
) -> FER2013Splits:
    """
    Load preprocessed FER2013 arrays from the compressed .npz file.

    Returns
    -------
    FER2013Splits
        Train, validation, and test arrays.
    """

    if not processed_data_path.exists():
        raise FileNotFoundError(
            f"Processed dataset was not found at: {processed_data_path}\n"
            "Run this command first:\n"
            "python -m scripts.save_processed_data"
        )

    data = np.load(processed_data_path)

    splits = FER2013Splits(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_val"],
        y_val=data["y_val"],
        X_test=data["X_test"],
        y_test=data["y_test"],
    )

    validate_processed_splits(splits)

    return splits


def load_class_weights(
    class_weights_path=CLASS_WEIGHTS_PATH,
) -> dict[int, float]:
    """
    Load class weights from JSON.

    JSON stores dictionary keys as strings, so this function converts them
    back to integers.
    """

    if not class_weights_path.exists():
        raise FileNotFoundError(
            f"Class weights file was not found at: {class_weights_path}\n"
            "Run this command first:\n"
            "python -m scripts.save_processed_data"
        )

    with class_weights_path.open("r", encoding="utf-8") as file:
        raw_class_weights = json.load(file)

    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in raw_class_weights.items()
    }

    return class_weights


def load_processed_summary(
    summary_path=PROCESSED_SUMMARY_PATH,
) -> pd.DataFrame:
    """
    Load the saved processed dataset summary.
    """

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Processed summary file was not found at: {summary_path}\n"
            "Run this command first:\n"
            "python -m scripts.save_processed_data"
        )

    summary = pd.read_csv(summary_path)

    return summary


def validate_processed_splits(splits: FER2013Splits) -> None:
    """
    Validate shapes, dtypes, and value ranges of processed arrays.
    """

    split_items = {
        "train": (splits.X_train, splits.y_train),
        "validation": (splits.X_val, splits.y_val),
        "test": (splits.X_test, splits.y_test),
    }

    for split_name, (X, y) in split_items.items():
        if X.ndim != 4:
            raise ValueError(
                f"{split_name} X must be 4D, but received shape {X.shape}."
            )

        if X.shape[1:] != IMAGE_SHAPE:
            raise ValueError(
                f"{split_name} X has invalid image shape. "
                f"Expected {IMAGE_SHAPE}, but received {X.shape[1:]}."
            )

        if y.ndim != 1:
            raise ValueError(
                f"{split_name} y must be 1D, but received shape {y.shape}."
            )

        if len(X) != len(y):
            raise ValueError(
                f"{split_name} X and y length mismatch: "
                f"{len(X)} images vs {len(y)} labels."
            )

        if X.dtype != np.float32:
            raise ValueError(
                f"{split_name} X must be float32, but received {X.dtype}."
            )

        if X.min() < 0 or X.max() > 1:
            raise ValueError(
                f"{split_name} X values must be normalized to [0, 1]."
            )

        unique_labels = set(np.unique(y).tolist())

        if not unique_labels.issubset(set(range(NUMBER_OF_CLASSES))):
            raise ValueError(
                f"{split_name} contains invalid labels: {sorted(unique_labels)}"
            )


def print_processed_data_report() -> None:
    """
    Print a short report about the processed dataset.
    """

    splits = load_processed_data()
    class_weights = load_class_weights()
    summary = load_processed_summary()

    print("Processed dataset loaded successfully.")
    print("\nSummary:")
    print(summary)

    print("\nArray ranges:")
    print(f"X_train: min={splits.X_train.min():.4f}, max={splits.X_train.max():.4f}")
    print(f"X_val:   min={splits.X_val.min():.4f}, max={splits.X_val.max():.4f}")
    print(f"X_test:  min={splits.X_test.min():.4f}, max={splits.X_test.max():.4f}")

    print("\nClass weights:")
    print(class_weights)
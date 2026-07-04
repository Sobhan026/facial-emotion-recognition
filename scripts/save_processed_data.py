from __future__ import annotations

import json

import numpy as np

from src.config import (
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    USE_HISTOGRAM_EQUALIZATION,
)
from src.dataset import (
    calculate_class_weights,
    load_fer2013_splits,
    summarize_splits,
)


def main() -> None:
    print("Loading and preprocessing FER2013 dataset...")

    splits = load_fer2013_splits(
        use_histogram_equalization=USE_HISTOGRAM_EQUALIZATION,
        one_hot=False,
    )

    processed_file_path = PROCESSED_DATA_DIR / "fer2013_processed.npz"
    class_weights_path = PROCESSED_DATA_DIR / "class_weights.json"
    summary_path = PROCESSED_DATA_DIR / "processed_summary.csv"

    print("Saving processed arrays...")

    np.savez_compressed(
        processed_file_path,
        X_train=splits.X_train,
        y_train=splits.y_train,
        X_val=splits.X_val,
        y_val=splits.y_val,
        X_test=splits.X_test,
        y_test=splits.y_test,
    )

    print("Calculating class weights...")

    class_weights = calculate_class_weights(splits.y_train)

    with class_weights_path.open("w", encoding="utf-8") as file:
        json.dump(class_weights, file, indent=4)

    summary = summarize_splits(splits)
    summary.to_csv(summary_path, index=False)

    print("\nProcessing completed successfully.")
    print(f"Processed dataset saved to: {processed_file_path}")
    print(f"Class weights saved to: {class_weights_path}")
    print(f"Summary saved to: {summary_path}")

    print("\nDataset summary:")
    print(summary)

    print("\nClass weights:")
    print(class_weights)

    print(f"\nRandom seed: {RANDOM_SEED}")
    print(f"Histogram equalization used: {USE_HISTOGRAM_EQUALIZATION}")


if __name__ == "__main__":
    main()
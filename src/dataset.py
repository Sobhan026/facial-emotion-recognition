from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    EMOTION_LABELS,
    FER2013_CSV_PATH,
    NUMBER_OF_CLASSES,
    RANDOM_SEED,
    TEST_SPLIT_NAME,
    TRAIN_SPLIT_NAME,
    VALID_USAGE_VALUES,
    VALIDATION_SPLIT_NAME,
    USE_HISTOGRAM_EQUALIZATION,
)
from src.preprocessing import preprocess_fer2013_pixel_string


@dataclass
class FER2013Splits:
    """
    Container for FER2013 train, validation, and test arrays.
    """

    X_train: np.ndarray
    y_train: np.ndarray

    X_val: np.ndarray
    y_val: np.ndarray

    X_test: np.ndarray
    y_test: np.ndarray


def load_fer2013_dataframe(csv_path=FER2013_CSV_PATH) -> pd.DataFrame:
    """
    Load the FER2013 CSV file as a pandas DataFrame.
    """

    dataframe = pd.read_csv(csv_path)

    return dataframe


def validate_fer2013_dataframe(dataframe: pd.DataFrame) -> None:
    """
    Validate the required FER2013 dataframe structure.
    """

    required_columns = {"emotion", "pixels", "Usage"}
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    invalid_emotions = set(dataframe["emotion"].unique()).difference(
        EMOTION_LABELS.keys()
    )

    if invalid_emotions:
        raise ValueError(
            f"Invalid emotion labels found: {sorted(invalid_emotions)}"
        )

    invalid_usage_values = set(dataframe["Usage"].unique()).difference(
        VALID_USAGE_VALUES
    )

    if invalid_usage_values:
        raise ValueError(
            f"Invalid Usage values found: {sorted(invalid_usage_values)}"
        )

    if dataframe["pixels"].isna().any():
        raise ValueError("Missing pixel strings found in the dataset.")


def add_emotion_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add a human-readable emotion_name column.
    """

    dataframe = dataframe.copy()
    dataframe["emotion_name"] = dataframe["emotion"].map(EMOTION_LABELS)

    return dataframe


def build_image_array(
    dataframe: pd.DataFrame,
    use_histogram_equalization: bool = USE_HISTOGRAM_EQUALIZATION,
) -> np.ndarray:
    """
    Convert FER2013 pixel strings into a NumPy image array.

    Output shape:
        (number_of_samples, 48, 48, 1)
    """

    images = [
        preprocess_fer2013_pixel_string(
            pixel_string,
            use_histogram_equalization=use_histogram_equalization,
        )
        for pixel_string in dataframe["pixels"]
    ]

    X = np.stack(images).astype(np.float32)

    return X


def build_label_array(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Convert emotion labels into an integer NumPy array.
    """

    y = dataframe["emotion"].to_numpy(dtype=np.int64)

    return y


def one_hot_encode_labels(labels: np.ndarray) -> np.ndarray:
    """
    Convert integer labels to one-hot encoded labels.

    Example:
        3 -> [0, 0, 0, 1, 0, 0, 0]
    """

    labels = np.asarray(labels, dtype=np.int64)

    one_hot = np.eye(NUMBER_OF_CLASSES, dtype=np.float32)[labels]

    return one_hot


def split_dataframe_by_usage(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split FER2013 dataframe using the official Usage column.

    Training    -> train set
    PublicTest  -> validation set
    PrivateTest -> test set
    """

    train_df = dataframe[dataframe["Usage"] == TRAIN_SPLIT_NAME].copy()
    val_df = dataframe[dataframe["Usage"] == VALIDATION_SPLIT_NAME].copy()
    test_df = dataframe[dataframe["Usage"] == TEST_SPLIT_NAME].copy()

    return train_df, val_df, test_df


def load_fer2013_splits(
    csv_path=FER2013_CSV_PATH,
    use_histogram_equalization: bool = USE_HISTOGRAM_EQUALIZATION,
    one_hot: bool = False,
) -> FER2013Splits:
    """
    Load FER2013 and return preprocessed train, validation, and test arrays.
    """

    dataframe = load_fer2013_dataframe(csv_path)
    validate_fer2013_dataframe(dataframe)

    train_df, val_df, test_df = split_dataframe_by_usage(dataframe)

    X_train = build_image_array(
        train_df,
        use_histogram_equalization=use_histogram_equalization,
    )
    y_train = build_label_array(train_df)

    X_val = build_image_array(
        val_df,
        use_histogram_equalization=use_histogram_equalization,
    )
    y_val = build_label_array(val_df)

    X_test = build_image_array(
        test_df,
        use_histogram_equalization=use_histogram_equalization,
    )
    y_test = build_label_array(test_df)

    if one_hot:
        y_train = one_hot_encode_labels(y_train)
        y_val = one_hot_encode_labels(y_val)
        y_test = one_hot_encode_labels(y_test)

    return FER2013Splits(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )


def calculate_class_weights(labels: np.ndarray) -> dict[int, float]:
    """
    Calculate class weights for handling class imbalance.

    These weights can later be passed to model.fit(..., class_weight=...).
    """

    labels = np.asarray(labels, dtype=np.int64)

    class_ids = np.arange(NUMBER_OF_CLASSES)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=class_ids,
        y=labels,
    )

    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(class_ids, weights)
    }

    return class_weights


def summarize_splits(splits: FER2013Splits) -> pd.DataFrame:
    """
    Create a simple summary table for train, validation, and test arrays.
    """

    summary = pd.DataFrame(
        {
            "Split": ["Train", "Validation", "Test"],
            "X shape": [
                str(splits.X_train.shape),
                str(splits.X_val.shape),
                str(splits.X_test.shape),
            ],
            "y shape": [
                str(splits.y_train.shape),
                str(splits.y_val.shape),
                str(splits.y_test.shape),
            ],
            "X dtype": [
                str(splits.X_train.dtype),
                str(splits.X_val.dtype),
                str(splits.X_test.dtype),
            ],
            "y dtype": [
                str(splits.y_train.dtype),
                str(splits.y_val.dtype),
                str(splits.y_test.dtype),
            ],
        }
    )

    return summary


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set NumPy random seed for reproducibility.
    """

    np.random.seed(seed)
import numpy as np

from src.config import IMAGE_SHAPE, NUMBER_OF_CLASSES
from src.processed_loader import (
    load_class_weights,
    load_processed_data,
    load_processed_summary,
)


def test_load_processed_data_shapes():
    splits = load_processed_data()

    assert splits.X_train.shape[1:] == IMAGE_SHAPE
    assert splits.X_val.shape[1:] == IMAGE_SHAPE
    assert splits.X_test.shape[1:] == IMAGE_SHAPE

    assert splits.X_train.ndim == 4
    assert splits.X_val.ndim == 4
    assert splits.X_test.ndim == 4

    assert splits.y_train.ndim == 1
    assert splits.y_val.ndim == 1
    assert splits.y_test.ndim == 1


def test_load_processed_data_lengths_match():
    splits = load_processed_data()

    assert len(splits.X_train) == len(splits.y_train)
    assert len(splits.X_val) == len(splits.y_val)
    assert len(splits.X_test) == len(splits.y_test)


def test_load_processed_data_dtype():
    splits = load_processed_data()

    assert splits.X_train.dtype == np.float32
    assert splits.X_val.dtype == np.float32
    assert splits.X_test.dtype == np.float32


def test_load_processed_data_value_range():
    splits = load_processed_data()

    assert splits.X_train.min() >= 0
    assert splits.X_train.max() <= 1

    assert splits.X_val.min() >= 0
    assert splits.X_val.max() <= 1

    assert splits.X_test.min() >= 0
    assert splits.X_test.max() <= 1


def test_labels_are_valid():
    splits = load_processed_data()

    valid_labels = set(range(NUMBER_OF_CLASSES))

    assert set(np.unique(splits.y_train)).issubset(valid_labels)
    assert set(np.unique(splits.y_val)).issubset(valid_labels)
    assert set(np.unique(splits.y_test)).issubset(valid_labels)


def test_load_class_weights():
    class_weights = load_class_weights()

    assert isinstance(class_weights, dict)
    assert set(class_weights.keys()) == set(range(NUMBER_OF_CLASSES))

    for class_id, weight in class_weights.items():
        assert isinstance(class_id, int)
        assert isinstance(weight, float)
        assert weight > 0


def test_load_processed_summary():
    summary = load_processed_summary()

    expected_columns = {
        "Split",
        "X shape",
        "y shape",
        "X dtype",
        "y dtype",
    }

    assert expected_columns.issubset(set(summary.columns))
    assert len(summary) == 3
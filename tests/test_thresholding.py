import numpy as np
import pytest

from src.thresholding import (
    calculate_threshold_metrics,
    get_top_k_predictions,
    predict_with_threshold,
    validate_probability_matrix,
    validate_probability_vector,
)


def test_validate_probability_vector_accepts_valid_probabilities():
    probabilities = np.array(
        [0.05, 0.02, 0.03, 0.70, 0.05, 0.10, 0.05],
        dtype=np.float32,
    )

    validated = validate_probability_vector(probabilities)

    assert validated.shape == (7,)
    assert np.isclose(validated.sum(), 1.0)


def test_validate_probability_vector_rejects_invalid_shape():
    probabilities = np.array(
        [
            [0.05, 0.02, 0.03, 0.70, 0.05, 0.10, 0.05],
        ],
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        validate_probability_vector(probabilities)


def test_validate_probability_vector_rejects_invalid_sum():
    probabilities = np.array(
        [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        validate_probability_vector(probabilities)


def test_predict_with_threshold_accepts_high_confidence_prediction():
    probabilities = np.array(
        [0.05, 0.02, 0.03, 0.70, 0.05, 0.10, 0.05],
        dtype=np.float32,
    )

    result = predict_with_threshold(
        probabilities,
        threshold=0.60,
    )

    assert result.predicted_class_id == 3
    assert result.predicted_emotion == "Happy"
    assert result.is_rejected is False
    assert result.confidence >= 0.60


def test_predict_with_threshold_rejects_low_confidence_prediction():
    probabilities = np.array(
        [0.20, 0.05, 0.10, 0.30, 0.10, 0.15, 0.10],
        dtype=np.float32,
    )

    result = predict_with_threshold(
        probabilities,
        threshold=0.60,
    )

    assert result.predicted_class_id == 3
    assert result.predicted_emotion == "Uncertain"
    assert result.is_rejected is True
    assert result.confidence < 0.60


def test_get_top_k_predictions_returns_sorted_predictions():
    probabilities = np.array(
        [0.05, 0.02, 0.03, 0.70, 0.05, 0.10, 0.05],
        dtype=np.float32,
    )

    top_predictions = get_top_k_predictions(
        probabilities,
        k=3,
    )

    assert len(top_predictions) == 3
    assert top_predictions[0][1] == "Happy"
    assert top_predictions[1][1] == "Surprise"

    confidences = [
        prediction[2]
        for prediction in top_predictions
    ]

    assert confidences == sorted(confidences, reverse=True)


def test_validate_probability_matrix_accepts_valid_matrix():
    probabilities = np.array(
        [
            [0.05, 0.02, 0.03, 0.70, 0.05, 0.10, 0.05],
            [0.20, 0.05, 0.10, 0.30, 0.10, 0.15, 0.10],
        ],
        dtype=np.float32,
    )

    validated = validate_probability_matrix(probabilities)

    assert validated.shape == (2, 7)
    assert np.allclose(validated.sum(axis=1), 1.0)


def test_calculate_threshold_metrics():
    y_true = np.array([3, 3, 5], dtype=np.int64)

    probabilities = np.array(
        [
            [0.05, 0.02, 0.03, 0.70, 0.05, 0.10, 0.05],
            [0.20, 0.05, 0.10, 0.30, 0.10, 0.15, 0.10],
            [0.05, 0.02, 0.03, 0.10, 0.05, 0.70, 0.05],
        ],
        dtype=np.float32,
    )

    metrics = calculate_threshold_metrics(
        y_true=y_true,
        probabilities=probabilities,
        threshold=0.60,
    )

    assert metrics.total_samples == 3
    assert metrics.accepted_samples == 2
    assert metrics.rejected_samples == 1
    assert np.isclose(metrics.coverage, 2 / 3)
    assert np.isclose(metrics.rejection_rate, 1 / 3)
    assert np.isclose(metrics.accepted_accuracy, 1.0)
    assert np.isclose(metrics.raw_accuracy, 1.0)
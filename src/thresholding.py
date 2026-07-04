from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import (
    CONFIDENCE_THRESHOLD,
    EMOTION_NAMES,
    REJECTION_LABEL,
)


@dataclass
class ThresholdPrediction:
    """
    Result of applying confidence threshold to one prediction.
    """

    predicted_class_id: int
    predicted_emotion: str
    confidence: float
    is_rejected: bool


@dataclass
class ThresholdMetrics:
    """
    Metrics for one confidence-threshold value.
    """

    threshold: float
    total_samples: int
    accepted_samples: int
    rejected_samples: int
    coverage: float
    rejection_rate: float
    accepted_accuracy: float
    raw_accuracy: float


def validate_probability_vector(probabilities: np.ndarray) -> np.ndarray:
    """
    Validate and return a single probability vector.

    Expected shape:
        (number_of_classes,)
    """

    probabilities = np.asarray(probabilities, dtype=np.float32)

    if probabilities.ndim != 1:
        raise ValueError(
            f"Expected a 1D probability vector, but received shape {probabilities.shape}."
        )

    if probabilities.size != len(EMOTION_NAMES):
        raise ValueError(
            f"Expected {len(EMOTION_NAMES)} probabilities, "
            f"but received {probabilities.size}."
        )

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must be between 0 and 1.")

    probability_sum = float(probabilities.sum())

    if not np.isclose(probability_sum, 1.0, atol=1e-4):
        raise ValueError(
            f"Probabilities must sum to 1. Received sum={probability_sum:.6f}."
        )

    return probabilities


def validate_probability_matrix(probabilities: np.ndarray) -> np.ndarray:
    """
    Validate and return a batch of probability vectors.

    Expected shape:
        (number_of_samples, number_of_classes)
    """

    probabilities = np.asarray(probabilities, dtype=np.float32)

    if probabilities.ndim != 2:
        raise ValueError(
            f"Expected a 2D probability matrix, but received shape {probabilities.shape}."
        )

    if probabilities.shape[1] != len(EMOTION_NAMES):
        raise ValueError(
            f"Expected {len(EMOTION_NAMES)} class probabilities, "
            f"but received {probabilities.shape[1]}."
        )

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must be between 0 and 1.")

    probability_sums = probabilities.sum(axis=1)

    if not np.allclose(probability_sums, 1.0, atol=1e-4):
        raise ValueError("Each probability row must sum to 1.")

    return probabilities


def predict_with_threshold(
    probabilities: np.ndarray,
    threshold: float = CONFIDENCE_THRESHOLD,
    rejection_label: str = REJECTION_LABEL,
) -> ThresholdPrediction:
    """
    Apply confidence threshold to a single prediction.

    If the highest class probability is below the threshold, the prediction
    is rejected and assigned the rejection label.
    """

    probabilities = validate_probability_vector(probabilities)

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    predicted_class_id = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class_id])

    is_rejected = confidence < threshold

    if is_rejected:
        predicted_emotion = rejection_label
    else:
        predicted_emotion = EMOTION_NAMES[predicted_class_id]

    return ThresholdPrediction(
        predicted_class_id=predicted_class_id,
        predicted_emotion=predicted_emotion,
        confidence=confidence,
        is_rejected=is_rejected,
    )


def get_top_k_predictions(
    probabilities: np.ndarray,
    k: int = 3,
) -> list[tuple[int, str, float]]:
    """
    Return top-k predicted classes with class id, emotion name, and confidence.
    """

    probabilities = validate_probability_vector(probabilities)

    if k <= 0:
        raise ValueError("k must be positive.")

    k = min(k, len(EMOTION_NAMES))

    top_indices = np.argsort(probabilities)[::-1][:k]

    top_predictions = [
        (
            int(class_id),
            EMOTION_NAMES[int(class_id)],
            float(probabilities[int(class_id)]),
        )
        for class_id in top_indices
    ]

    return top_predictions


def calculate_threshold_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> ThresholdMetrics:
    """
    Calculate threshold metrics for a batch of predictions.

    Definitions:
        raw_accuracy:
            Accuracy before applying rejection.

        coverage:
            Percentage of samples accepted by the threshold.

        rejection_rate:
            Percentage of samples rejected by the threshold.

        accepted_accuracy:
            Accuracy only among accepted predictions.
    """

    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = validate_probability_matrix(probabilities)

    if y_true.ndim != 1:
        raise ValueError(
            f"Expected y_true to be 1D, but received shape {y_true.shape}."
        )

    if len(y_true) != len(probabilities):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples, "
            f"but probabilities has {len(probabilities)} samples."
        )

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    predicted_classes = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)

    correct_predictions = predicted_classes == y_true

    raw_accuracy = float(np.mean(correct_predictions))

    accepted_mask = confidences >= threshold

    total_samples = int(len(y_true))
    accepted_samples = int(np.sum(accepted_mask))
    rejected_samples = total_samples - accepted_samples

    coverage = accepted_samples / total_samples
    rejection_rate = rejected_samples / total_samples

    if accepted_samples == 0:
        accepted_accuracy = 0.0
    else:
        accepted_accuracy = float(
            np.mean(correct_predictions[accepted_mask])
        )

    return ThresholdMetrics(
        threshold=float(threshold),
        total_samples=total_samples,
        accepted_samples=accepted_samples,
        rejected_samples=rejected_samples,
        coverage=float(coverage),
        rejection_rate=float(rejection_rate),
        accepted_accuracy=float(accepted_accuracy),
        raw_accuracy=raw_accuracy,
    )


def threshold_metrics_to_dict(metrics: ThresholdMetrics) -> dict[str, float | int]:
    """
    Convert ThresholdMetrics dataclass to a dictionary.
    """

    return {
        "threshold": metrics.threshold,
        "total_samples": metrics.total_samples,
        "accepted_samples": metrics.accepted_samples,
        "rejected_samples": metrics.rejected_samples,
        "coverage": metrics.coverage,
        "rejection_rate": metrics.rejection_rate,
        "accepted_accuracy": metrics.accepted_accuracy,
        "raw_accuracy": metrics.raw_accuracy,
    }
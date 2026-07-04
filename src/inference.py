from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.config import (
    CONFIDENCE_THRESHOLD,
    METRICS_DIR,
    MODELS_DIR,
    REJECTION_LABEL,
)
from src.preprocessing import preprocess_face_image
from src.thresholding import (
    ThresholdPrediction,
    get_top_k_predictions,
    predict_with_threshold,
)


DEFAULT_MODEL_PATH = MODELS_DIR / "improved_cnn_no_class_weights_best.keras"
RECOMMENDED_THRESHOLD_PATH = METRICS_DIR / "recommended_threshold.json"


@dataclass
class EmotionPredictionResult:
    """
    Complete emotion prediction result for one face image.
    """

    predicted_class_id: int
    predicted_emotion: str
    confidence: float
    is_rejected: bool
    threshold: float
    top_predictions: list[tuple[int, str, float]]


def load_emotion_model(
    model_path: Path = DEFAULT_MODEL_PATH,
) -> tf.keras.Model:
    """
    Load a trained emotion recognition model.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model was not found at: {model_path}"
        )

    model = tf.keras.models.load_model(model_path)

    return model


def load_recommended_threshold(
    threshold_path: Path = RECOMMENDED_THRESHOLD_PATH,
    fallback_threshold: float = CONFIDENCE_THRESHOLD,
) -> float:
    """
    Load the recommended confidence threshold.

    If the file does not exist, use the fallback threshold from config.py.
    """

    if not threshold_path.exists():
        return float(fallback_threshold)

    with threshold_path.open("r", encoding="utf-8") as file:
        threshold_data = json.load(file)

    threshold = float(
        threshold_data.get(
            "recommended_threshold",
            fallback_threshold,
        )
    )

    return threshold


def predict_emotion_from_face(
    model: tf.keras.Model,
    face_image: np.ndarray,
    threshold: float | None = None,
    rejection_label: str = REJECTION_LABEL,
    top_k: int = 3,
    use_histogram_equalization: bool = True,
) -> EmotionPredictionResult:
    """
    Predict emotion for one detected face image.

    Parameters
    ----------
    model:
        Trained Keras emotion recognition model.

    face_image:
        A cropped face image. It can be grayscale or BGR.

    threshold:
        Confidence threshold. If None, the recommended threshold is loaded.

    rejection_label:
        Label used when confidence is below threshold.

    top_k:
        Number of top predictions to return.

    use_histogram_equalization:
        Whether to apply histogram equalization during face preprocessing.
    """

    if threshold is None:
        threshold = load_recommended_threshold()

    processed_face = preprocess_face_image(
        face_image,
        use_histogram_equalization=use_histogram_equalization,
    )

    batch = np.expand_dims(
        processed_face,
        axis=0,
    )

    probabilities = model.predict(
        batch,
        verbose=0,
    )[0]

    threshold_result: ThresholdPrediction = predict_with_threshold(
        probabilities=probabilities,
        threshold=threshold,
        rejection_label=rejection_label,
    )

    top_predictions = get_top_k_predictions(
        probabilities=probabilities,
        k=top_k,
    )

    return EmotionPredictionResult(
        predicted_class_id=threshold_result.predicted_class_id,
        predicted_emotion=threshold_result.predicted_emotion,
        confidence=threshold_result.confidence,
        is_rejected=threshold_result.is_rejected,
        threshold=float(threshold),
        top_predictions=top_predictions,
    )


def format_prediction_label(
    prediction: EmotionPredictionResult,
) -> str:
    """
    Format prediction result for display on images or webcam frames.
    """

    confidence_percent = prediction.confidence * 100

    if prediction.is_rejected:
        label = (
            f"{prediction.predicted_emotion} "
            f"({confidence_percent:.1f}% < {prediction.threshold:.2f})"
        )
    else:
        label = (
            f"{prediction.predicted_emotion}: "
            f"{confidence_percent:.1f}%"
        )

    return label


def format_top_predictions(
    prediction: EmotionPredictionResult,
) -> list[str]:
    """
    Format top-k predictions as readable strings.
    """

    formatted = [
        f"{emotion}: {confidence * 100:.1f}%"
        for _, emotion, confidence in prediction.top_predictions
    ]

    return formatted
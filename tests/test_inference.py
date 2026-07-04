import numpy as np

from src.config import IMAGE_SHAPE
from src.inference import (
    EmotionPredictionResult,
    format_prediction_label,
    format_top_predictions,
    load_recommended_threshold,
    predict_emotion_from_face,
)


class DummyModel:
    """
    Simple fake model for testing inference logic without loading TensorFlow model.
    """

    def predict(self, batch, verbose=0):
        batch_size = batch.shape[0]

        probabilities = np.array(
            [
                [0.05, 0.02, 0.03, 0.70, 0.05, 0.10, 0.05],
            ],
            dtype=np.float32,
        )

        return np.repeat(
            probabilities,
            repeats=batch_size,
            axis=0,
        )


class LowConfidenceDummyModel:
    """
    Fake model that returns low-confidence probabilities.
    """

    def predict(self, batch, verbose=0):
        batch_size = batch.shape[0]

        probabilities = np.array(
            [
                [0.20, 0.05, 0.10, 0.30, 0.10, 0.15, 0.10],
            ],
            dtype=np.float32,
        )

        return np.repeat(
            probabilities,
            repeats=batch_size,
            axis=0,
        )


def test_load_recommended_threshold_returns_float():
    threshold = load_recommended_threshold()

    assert isinstance(threshold, float)
    assert 0 <= threshold <= 1


def test_predict_emotion_from_face_accepts_high_confidence_prediction():
    model = DummyModel()
    face_image = np.random.rand(48, 48).astype(np.float32)

    result = predict_emotion_from_face(
        model=model,
        face_image=face_image,
        threshold=0.60,
        use_histogram_equalization=False,
    )

    assert isinstance(result, EmotionPredictionResult)
    assert result.predicted_class_id == 3
    assert result.predicted_emotion == "Happy"
    assert result.is_rejected is False
    assert result.confidence >= 0.60
    assert result.threshold == 0.60
    assert len(result.top_predictions) == 3


def test_predict_emotion_from_face_rejects_low_confidence_prediction():
    model = LowConfidenceDummyModel()
    face_image = np.random.rand(48, 48).astype(np.float32)

    result = predict_emotion_from_face(
        model=model,
        face_image=face_image,
        threshold=0.60,
        use_histogram_equalization=False,
    )

    assert isinstance(result, EmotionPredictionResult)
    assert result.predicted_class_id == 3
    assert result.predicted_emotion == "Uncertain"
    assert result.is_rejected is True
    assert result.confidence < 0.60
    assert result.threshold == 0.60
    assert len(result.top_predictions) == 3


def test_predict_emotion_from_face_accepts_bgr_image():
    model = DummyModel()
    face_image = np.random.randint(
        low=0,
        high=255,
        size=(48, 48, 3),
        dtype=np.uint8,
    )

    result = predict_emotion_from_face(
        model=model,
        face_image=face_image,
        threshold=0.60,
        use_histogram_equalization=True,
    )

    assert isinstance(result, EmotionPredictionResult)
    assert result.predicted_emotion == "Happy"


def test_format_prediction_label_for_accepted_prediction():
    result = EmotionPredictionResult(
        predicted_class_id=3,
        predicted_emotion="Happy",
        confidence=0.876,
        is_rejected=False,
        threshold=0.55,
        top_predictions=[
            (3, "Happy", 0.876),
            (6, "Neutral", 0.080),
            (5, "Surprise", 0.044),
        ],
    )

    label = format_prediction_label(result)

    assert label == "Happy: 87.6%"


def test_format_prediction_label_for_rejected_prediction():
    result = EmotionPredictionResult(
        predicted_class_id=3,
        predicted_emotion="Uncertain",
        confidence=0.338,
        is_rejected=True,
        threshold=0.55,
        top_predictions=[
            (3, "Happy", 0.338),
            (6, "Neutral", 0.200),
            (5, "Surprise", 0.100),
        ],
    )

    label = format_prediction_label(result)

    assert label == "Uncertain (33.8% < 0.55)"


def test_format_top_predictions():
    result = EmotionPredictionResult(
        predicted_class_id=3,
        predicted_emotion="Happy",
        confidence=0.876,
        is_rejected=False,
        threshold=0.55,
        top_predictions=[
            (3, "Happy", 0.876),
            (6, "Neutral", 0.080),
            (5, "Surprise", 0.044),
        ],
    )

    formatted = format_top_predictions(result)

    assert formatted == [
        "Happy: 87.6%",
        "Neutral: 8.0%",
        "Surprise: 4.4%",
    ]
import pytest

from src.temporal_smoothing import (
    PredictionSmoother,
    SmoothedPrediction,
)


def test_prediction_smoother_rejects_invalid_window_size():
    with pytest.raises(ValueError):
        PredictionSmoother(window_size=0)


def test_empty_smoother_returns_uncertain():
    smoother = PredictionSmoother(window_size=3)

    result = smoother.get_smoothed_prediction()

    assert isinstance(result, SmoothedPrediction)
    assert result.emotion == "Uncertain"
    assert result.confidence == 0.0
    assert result.frame_count == 0


def test_prediction_smoother_returns_single_prediction():
    smoother = PredictionSmoother(window_size=3)

    result = smoother.update(
        emotion="Happy",
        confidence=0.70,
    )

    assert result.emotion == "Happy"
    assert result.confidence == 0.70
    assert result.frame_count == 1


def test_prediction_smoother_returns_majority_prediction():
    smoother = PredictionSmoother(window_size=3)

    smoother.update("Happy", 0.70)
    smoother.update("Happy", 0.80)
    result = smoother.update("Sad", 0.60)

    assert result.emotion == "Happy"
    assert result.confidence == 0.75
    assert result.frame_count == 3


def test_prediction_smoother_respects_window_size():
    smoother = PredictionSmoother(window_size=3)

    smoother.update("Happy", 0.70)
    smoother.update("Happy", 0.80)
    smoother.update("Sad", 0.60)
    result = smoother.update("Sad", 0.90)

    assert result.emotion == "Sad"
    assert result.confidence == 0.75
    assert result.frame_count == 3


def test_prediction_smoother_rejects_invalid_confidence():
    smoother = PredictionSmoother(window_size=3)

    with pytest.raises(ValueError):
        smoother.update(
            emotion="Happy",
            confidence=1.5,
        )


def test_prediction_smoother_reset_clears_history():
    smoother = PredictionSmoother(window_size=3)

    smoother.update("Happy", 0.70)
    smoother.update("Happy", 0.80)

    smoother.reset()

    result = smoother.get_smoothed_prediction()

    assert result.emotion == "Uncertain"
    assert result.confidence == 0.0
    assert result.frame_count == 0
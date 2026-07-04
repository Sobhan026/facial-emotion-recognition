from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass


@dataclass
class SmoothedPrediction:
    """
    Smoothed prediction output.
    """

    emotion: str
    confidence: float
    frame_count: int


class PredictionSmoother:
    """
    Temporal smoother for real-time emotion predictions.

    It keeps the last N accepted/rejected predictions and returns the most
    frequent emotion label. Confidence is averaged over the frames that match
    the selected emotion.
    """

    def __init__(
        self,
        window_size: int = 7,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive.")

        self.window_size = window_size
        self.predictions: deque[tuple[str, float]] = deque(
            maxlen=window_size,
        )

    def update(
        self,
        emotion: str,
        confidence: float,
    ) -> SmoothedPrediction:
        """
        Add a new prediction and return the smoothed result.
        """

        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1.")

        self.predictions.append(
            (
                emotion,
                float(confidence),
            )
        )

        return self.get_smoothed_prediction()

    def get_smoothed_prediction(self) -> SmoothedPrediction:
        """
        Return the most frequent emotion in the smoothing window.
        """

        if not self.predictions:
            return SmoothedPrediction(
                emotion="Uncertain",
                confidence=0.0,
                frame_count=0,
            )

        emotion_counts = Counter(
            emotion
            for emotion, _ in self.predictions
        )

        selected_emotion = emotion_counts.most_common(1)[0][0]

        matching_confidences = [
            confidence
            for emotion, confidence in self.predictions
            if emotion == selected_emotion
        ]

        average_confidence = sum(matching_confidences) / len(matching_confidences)

        return SmoothedPrediction(
            emotion=selected_emotion,
            confidence=float(average_confidence),
            frame_count=len(self.predictions),
        )

    def reset(self) -> None:
        """
        Clear the smoothing window.
        """

        self.predictions.clear()
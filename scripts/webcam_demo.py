from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
from datetime import datetime

import cv2

from src.config import DEMO_DIR
from src.face_detection import (
    crop_face,
    detect_faces,
    load_face_detector,
    select_largest_face,
)
from src.inference import (
    format_top_predictions,
    load_emotion_model,
    load_recommended_threshold,
    predict_emotion_from_face,
)
from src.temporal_smoothing import PredictionSmoother


WINDOW_NAME = "Facial Emotion Recognition - Webcam Demo"
SMOOTHING_WINDOW_SIZE = 7


def draw_text_lines(
    frame,
    lines: list[str],
    x: int,
    y: int,
    line_height: int = 24,
    font_scale: float = 0.6,
    thickness: int = 2,
) -> None:
    """
    Draw multiple text lines on a frame.
    """

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def draw_face_prediction(
    frame,
    face_box,
    label: str,
    top_predictions: list[str],
) -> None:
    """
    Draw face box, main label, and top predictions on the frame.
    """

    x, y, w, h = face_box.as_tuple

    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2,
    )

    label_y = max(y - 12, 25)

    cv2.putText(
        frame,
        label,
        (x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    panel_x = x
    panel_y = y + h + 25

    if panel_y + 80 > frame.shape[0]:
        panel_y = y - 95

    for index, prediction_text in enumerate(top_predictions):
        cv2.putText(
            frame,
            prediction_text,
            (panel_x, panel_y + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def save_webcam_snapshot(frame) -> None:
    """
    Save current webcam frame to outputs/demo.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = DEMO_DIR / f"webcam_prediction_snapshot_{timestamp}.png"

    cv2.imwrite(
        str(output_path),
        frame,
    )

    print(f"Snapshot saved to: {output_path}")


def build_smoothed_label(
    emotion: str,
    confidence: float,
    frame_count: int,
) -> str:
    """
    Build display label for smoothed webcam prediction.
    """

    confidence_percent = confidence * 100

    label = (
        f"{emotion}: "
        f"{confidence_percent:.1f}% "
        f"(smooth {frame_count})"
    )

    return label


def run_webcam_demo(
    camera_index: int = 0,
) -> None:
    """
    Run real-time webcam facial emotion recognition demo.

    Controls:
        q or ESC: quit
        s: save current frame snapshot
    """

    print("Loading selected emotion model...")
    model = load_emotion_model()

    threshold = load_recommended_threshold()

    print(f"Using confidence threshold: {threshold:.2f}")

    print("Loading face detector...")
    detector = load_face_detector()

    smoother = PredictionSmoother(
        window_size=SMOOTHING_WINDOW_SIZE,
    )

    print(f"Using temporal smoothing window: {SMOOTHING_WINDOW_SIZE}")

    print(f"Opening webcam with camera index: {camera_index}")
    capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open webcam with camera index {camera_index}."
        )

    previous_time = time.time()

    print("\nWebcam demo started.")
    print("Controls:")
    print("  q or ESC = quit")
    print("  s        = save snapshot")

    while True:
        success, frame = capture.read()

        if not success:
            print("Could not read frame from webcam.")
            break

        frame = cv2.flip(
            frame,
            1,
        )

        current_time = time.time()
        elapsed_time = current_time - previous_time
        previous_time = current_time

        fps = 1.0 / elapsed_time if elapsed_time > 0 else 0.0

        face_boxes = detect_faces(
            frame=frame,
            detector=detector,
            scale_factor=1.1,
            min_neighbors=5,
            min_size=(60, 60),
        )

        primary_face = select_largest_face(face_boxes)

        if primary_face is None:
            smoother.reset()
        else:
            try:
                face_crop = crop_face(
                    frame=frame,
                    face_box=primary_face,
                    margin_ratio=0.15,
                )

                prediction = predict_emotion_from_face(
                    model=model,
                    face_image=face_crop,
                    threshold=threshold,
                    use_histogram_equalization=True,
                )

                smoothed_prediction = smoother.update(
                    emotion=prediction.predicted_emotion,
                    confidence=prediction.confidence,
                )

                label = build_smoothed_label(
                    emotion=smoothed_prediction.emotion,
                    confidence=smoothed_prediction.confidence,
                    frame_count=smoothed_prediction.frame_count,
                )

                top_predictions = format_top_predictions(prediction)

                draw_face_prediction(
                    frame=frame,
                    face_box=primary_face,
                    label=label,
                    top_predictions=top_predictions,
                )

            except Exception as error:
                print(f"Prediction error: {error}")
                smoother.reset()

        status_lines = [
            f"Faces detected: {len(face_boxes)}",
            f"Primary face mode: largest face",
            f"Threshold: {threshold:.2f}",
            f"Smoothing window: {SMOOTHING_WINDOW_SIZE}",
            f"FPS: {fps:.1f}",
            "Press q/ESC to quit | s to save",
        ]

        draw_text_lines(
            frame=frame,
            lines=status_lines,
            x=10,
            y=25,
        )

        cv2.imshow(
            WINDOW_NAME,
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

        if key == ord("s"):
            save_webcam_snapshot(frame)

    capture.release()
    cv2.destroyAllWindows()

    print("Webcam demo finished.")


def main() -> None:
    run_webcam_demo(camera_index=0)


if __name__ == "__main__":
    main()
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FaceBox:
    """
    Bounding box for one detected face.
    """

    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


def get_default_face_cascade_path() -> str:
    """
    Return OpenCV's default Haar cascade path for frontal face detection.
    """

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    return cascade_path


def load_face_detector(
    cascade_path: str | None = None,
) -> cv2.CascadeClassifier:
    """
    Load OpenCV Haar cascade face detector.
    """

    if cascade_path is None:
        cascade_path = get_default_face_cascade_path()

    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise FileNotFoundError(
            f"Could not load face cascade from: {cascade_path}"
        )

    return detector


def convert_to_grayscale(frame: np.ndarray) -> np.ndarray:
    """
    Convert a BGR or grayscale frame to grayscale.
    """

    if frame.ndim == 2:
        return frame

    if frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )
        return gray

    raise ValueError(
        f"Expected grayscale or BGR image, but received shape {frame.shape}."
    )


def detect_faces(
    frame: np.ndarray,
    detector: cv2.CascadeClassifier,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: tuple[int, int] = (30, 30),
) -> list[FaceBox]:
    """
    Detect faces in a frame.

    Returns a list of FaceBox objects.
    """

    gray = convert_to_grayscale(frame)

    detected = detector.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )

    face_boxes = [
        FaceBox(
            x=int(x),
            y=int(y),
            width=int(w),
            height=int(h),
        )
        for x, y, w, h in detected
    ]

    return face_boxes


def select_largest_face(
    face_boxes: list[FaceBox],
) -> FaceBox | None:
    """
    Select the largest detected face.

    This is useful for webcam mode when multiple faces are detected.
    """

    if not face_boxes:
        return None

    largest_face = max(
        face_boxes,
        key=lambda box: box.area,
    )

    return largest_face


def crop_face(
    frame: np.ndarray,
    face_box: FaceBox,
    margin_ratio: float = 0.15,
) -> np.ndarray:
    """
    Crop a face region from the frame with optional margin.

    The margin helps keep more facial context around the detected face.
    """

    if not 0 <= margin_ratio <= 1:
        raise ValueError("margin_ratio must be between 0 and 1.")

    image_height, image_width = frame.shape[:2]

    margin_x = int(face_box.width * margin_ratio)
    margin_y = int(face_box.height * margin_ratio)

    x1 = max(face_box.x - margin_x, 0)
    y1 = max(face_box.y - margin_y, 0)

    x2 = min(face_box.x + face_box.width + margin_x, image_width)
    y2 = min(face_box.y + face_box.height + margin_y, image_height)

    cropped = frame[y1:y2, x1:x2]

    if cropped.size == 0:
        raise ValueError("Cropped face is empty.")

    return cropped


def draw_face_box(
    frame: np.ndarray,
    face_box: FaceBox,
    label: str,
) -> np.ndarray:
    """
    Draw face bounding box and label on a frame.
    """

    output = frame.copy()

    x, y, w, h = face_box.as_tuple

    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2,
    )

    label_y = max(y - 10, 20)

    cv2.putText(
        output,
        label,
        (x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return output
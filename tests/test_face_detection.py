import numpy as np
import pytest

from src.face_detection import (
    FaceBox,
    convert_to_grayscale,
    crop_face,
    get_default_face_cascade_path,
    load_face_detector,
    select_largest_face,
)


def test_face_box_area_and_tuple():
    face_box = FaceBox(
        x=10,
        y=20,
        width=30,
        height=40,
    )

    assert face_box.area == 1200
    assert face_box.as_tuple == (10, 20, 30, 40)


def test_get_default_face_cascade_path_returns_xml_path():
    cascade_path = get_default_face_cascade_path()

    assert cascade_path.endswith("haarcascade_frontalface_default.xml")


def test_load_face_detector_returns_non_empty_detector():
    detector = load_face_detector()

    assert detector.empty() is False


def test_convert_to_grayscale_accepts_grayscale_image():
    image = np.random.randint(
        low=0,
        high=255,
        size=(100, 100),
        dtype=np.uint8,
    )

    gray = convert_to_grayscale(image)

    assert gray.shape == (100, 100)
    assert gray.dtype == np.uint8


def test_convert_to_grayscale_accepts_bgr_image():
    image = np.random.randint(
        low=0,
        high=255,
        size=(100, 100, 3),
        dtype=np.uint8,
    )

    gray = convert_to_grayscale(image)

    assert gray.shape == (100, 100)
    assert gray.dtype == np.uint8


def test_convert_to_grayscale_rejects_invalid_shape():
    image = np.random.randint(
        low=0,
        high=255,
        size=(100, 100, 4),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):
        convert_to_grayscale(image)


def test_select_largest_face_returns_none_for_empty_list():
    largest = select_largest_face([])

    assert largest is None


def test_select_largest_face_returns_largest_area():
    face_boxes = [
        FaceBox(x=0, y=0, width=10, height=10),
        FaceBox(x=0, y=0, width=30, height=20),
        FaceBox(x=0, y=0, width=15, height=15),
    ]

    largest = select_largest_face(face_boxes)

    assert largest is not None
    assert largest.width == 30
    assert largest.height == 20


def test_crop_face_without_margin():
    frame = np.random.randint(
        low=0,
        high=255,
        size=(100, 100, 3),
        dtype=np.uint8,
    )

    face_box = FaceBox(
        x=20,
        y=30,
        width=40,
        height=30,
    )

    cropped = crop_face(
        frame,
        face_box,
        margin_ratio=0.0,
    )

    assert cropped.shape == (30, 40, 3)


def test_crop_face_with_margin_clips_to_image_boundaries():
    frame = np.random.randint(
        low=0,
        high=255,
        size=(100, 100, 3),
        dtype=np.uint8,
    )

    face_box = FaceBox(
        x=0,
        y=0,
        width=40,
        height=40,
    )

    cropped = crop_face(
        frame,
        face_box,
        margin_ratio=0.25,
    )

    assert cropped.shape[0] == 50
    assert cropped.shape[1] == 50


def test_crop_face_rejects_invalid_margin():
    frame = np.random.randint(
        low=0,
        high=255,
        size=(100, 100, 3),
        dtype=np.uint8,
    )

    face_box = FaceBox(
        x=20,
        y=20,
        width=40,
        height=40,
    )

    with pytest.raises(ValueError):
        crop_face(
            frame,
            face_box,
            margin_ratio=1.5,
        )
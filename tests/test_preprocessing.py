import numpy as np
import pandas as pd
import pytest

from src.config import FER2013_CSV_PATH, IMAGE_SHAPE, NUMBER_OF_PIXELS
from src.preprocessing import (
    add_channel_dimension,
    normalize_image,
    pixel_string_to_array,
    preprocess_fer2013_pixel_string,
    validate_image_range,
)


def test_pixel_string_to_array_shape():
    df = pd.read_csv(FER2013_CSV_PATH)
    pixel_string = df.iloc[0]["pixels"]

    image = pixel_string_to_array(pixel_string)

    assert image.shape == (48, 48)
    assert image.size == NUMBER_OF_PIXELS
    assert image.dtype == np.float32


def test_validate_image_range_accepts_valid_image():
    image = np.zeros((48, 48), dtype=np.float32)

    validate_image_range(image)


def test_validate_image_range_rejects_invalid_image():
    image = np.zeros((48, 48), dtype=np.float32)
    image[0, 0] = 300

    with pytest.raises(ValueError):
        validate_image_range(image)


def test_normalize_image_range():
    image = np.array(
        [
            [0, 127.5],
            [255, 64],
        ],
        dtype=np.float32,
    )

    normalized = normalize_image(image)

    assert normalized.min() >= 0
    assert normalized.max() <= 1
    assert normalized.dtype == np.float32


def test_add_channel_dimension_shape():
    image = np.zeros((48, 48), dtype=np.float32)

    image_with_channel = add_channel_dimension(image)

    assert image_with_channel.shape == IMAGE_SHAPE


def test_preprocess_fer2013_pixel_string_output():
    df = pd.read_csv(FER2013_CSV_PATH)
    pixel_string = df.iloc[0]["pixels"]

    processed_image = preprocess_fer2013_pixel_string(pixel_string)

    assert processed_image.shape == IMAGE_SHAPE
    assert processed_image.dtype == np.float32
    assert processed_image.min() >= 0
    assert processed_image.max() <= 1
from __future__ import annotations

import cv2
import numpy as np

from src.config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_CHANNELS,
    IMAGE_SHAPE,
    NUMBER_OF_PIXELS,
    NORMALIZATION_DIVISOR,
)


def pixel_string_to_array(pixel_string: str) -> np.ndarray:
    """
    Convert a FER2013 pixel string into a 48x48 grayscale image.

    Parameters
    ----------
    pixel_string:
        Space-separated pixel values from the FER2013 CSV file.

    Returns
    -------
    np.ndarray
        Grayscale image with shape (48, 48).
    """

    pixels = np.fromstring(str(pixel_string), sep=" ", dtype=np.float32)

    if pixels.size != NUMBER_OF_PIXELS:
        raise ValueError(
            f"Invalid pixel count. Expected {NUMBER_OF_PIXELS}, "
            f"but received {pixels.size}."
        )

    image = pixels.reshape(IMAGE_HEIGHT, IMAGE_WIDTH)

    return image


def validate_image_range(image: np.ndarray) -> None:
    """
    Validate that image pixel values are in the expected grayscale range.
    """

    if image.min() < 0 or image.max() > 255:
        raise ValueError(
            "Invalid pixel range. Expected values between 0 and 255."
        )


def resize_image(
    image: np.ndarray,
    target_size: tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT),
) -> np.ndarray:
    """
    Resize image to the target size.
    """

    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    return resized


def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to improve image contrast.

    The input image must be a grayscale image.
    """

    image_uint8 = image.astype(np.uint8)
    equalized = cv2.equalizeHist(image_uint8)

    return equalized.astype(np.float32)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values from [0, 255] to [0, 1].
    """

    normalized = image.astype(np.float32) / NORMALIZATION_DIVISOR

    return normalized


def add_channel_dimension(image: np.ndarray) -> np.ndarray:
    """
    Convert image shape from (48, 48) to (48, 48, 1).
    """

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale image, but received shape {image.shape}."
        )

    image_with_channel = np.expand_dims(image, axis=-1)

    if image_with_channel.shape != IMAGE_SHAPE:
        raise ValueError(
            f"Invalid image shape. Expected {IMAGE_SHAPE}, "
            f"but received {image_with_channel.shape}."
        )

    return image_with_channel


def preprocess_fer2013_pixel_string(
    pixel_string: str,
    use_histogram_equalization: bool = False,
) -> np.ndarray:
    """
    Full preprocessing pipeline for one FER2013 CSV row.

    Steps:
    1. Convert pixel string to 48x48 image.
    2. Validate pixel range.
    3. Optionally apply histogram equalization.
    4. Normalize image to [0, 1].
    5. Add channel dimension.

    Returns
    -------
    np.ndarray
        Preprocessed image with shape (48, 48, 1).
    """

    image = pixel_string_to_array(pixel_string)
    validate_image_range(image)

    if use_histogram_equalization:
        image = apply_histogram_equalization(image)

    image = normalize_image(image)
    image = add_channel_dimension(image)

    return image


def preprocess_face_image(
    face_image: np.ndarray,
    use_histogram_equalization: bool = True,
) -> np.ndarray:
    """
    Preprocess a detected face image for model inference.

    This function is intended for webcam or external image input.

    Parameters
    ----------
    face_image:
        Face crop as a grayscale or BGR image.

    Returns
    -------
    np.ndarray
        Preprocessed face image with shape (48, 48, 1).
    """

    if face_image is None:
        raise ValueError("face_image cannot be None.")

    if face_image.ndim == 3:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

    face_image = resize_image(face_image)

    if use_histogram_equalization:
        face_image = apply_histogram_equalization(face_image)

    face_image = normalize_image(face_image)
    face_image = add_channel_dimension(face_image)

    return face_image
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models

from src.config import (
    IMAGE_SHAPE,
    LEARNING_RATE,
    NUMBER_OF_CLASSES,
)


def build_baseline_cnn_model(
    input_shape: tuple[int, int, int] = IMAGE_SHAPE,
    number_of_classes: int = NUMBER_OF_CLASSES,
    learning_rate: float = LEARNING_RATE,
) -> tf.keras.Model:
    """
    Build and compile a baseline CNN model for FER2013.

    This model is intentionally simple. It will be used as the baseline
    for comparison with the improved CNN model later.

    Architecture:
        Input
        Conv2D -> MaxPooling -> Dropout
        Conv2D -> MaxPooling -> Dropout
        Conv2D -> MaxPooling -> Dropout
        Flatten
        Dense
        Dropout
        Softmax output
    """

    model = models.Sequential(
        [
            layers.Input(shape=input_shape, name="input_image"),

            layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
                name="conv_1",
            ),
            layers.MaxPooling2D(
                pool_size=(2, 2),
                name="pool_1",
            ),
            layers.Dropout(
                rate=0.25,
                name="dropout_1",
            ),

            layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
                name="conv_2",
            ),
            layers.MaxPooling2D(
                pool_size=(2, 2),
                name="pool_2",
            ),
            layers.Dropout(
                rate=0.25,
                name="dropout_2",
            ),

            layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
                name="conv_3",
            ),
            layers.MaxPooling2D(
                pool_size=(2, 2),
                name="pool_3",
            ),
            layers.Dropout(
                rate=0.30,
                name="dropout_3",
            ),

            layers.Flatten(name="flatten"),

            layers.Dense(
                units=128,
                activation="relu",
                name="dense_1",
            ),
            layers.Dropout(
                rate=0.50,
                name="dropout_4",
            ),

            layers.Dense(
                units=number_of_classes,
                activation="softmax",
                name="emotion_output",
            ),
        ],
        name="baseline_cnn",
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_improved_cnn_model(
    input_shape: tuple[int, int, int] = IMAGE_SHAPE,
    number_of_classes: int = NUMBER_OF_CLASSES,
    learning_rate: float = LEARNING_RATE,
) -> tf.keras.Model:
    """
    Build and compile an improved CNN model for FER2013.

    Improvements over the baseline model:
        - Data augmentation
        - Batch normalization
        - Deeper convolutional blocks
        - Global average pooling instead of flattening
        - Stronger dropout regularization
    """

    data_augmentation = models.Sequential(
        [
            layers.RandomFlip(
                mode="horizontal",
                name="random_horizontal_flip",
            ),
            layers.RandomRotation(
                factor=0.08,
                name="random_rotation",
            ),
            layers.RandomZoom(
                height_factor=0.10,
                width_factor=0.10,
                name="random_zoom",
            ),
            layers.RandomTranslation(
                height_factor=0.08,
                width_factor=0.08,
                name="random_translation",
            ),
        ],
        name="data_augmentation",
    )

    inputs = layers.Input(
        shape=input_shape,
        name="input_image",
    )

    x = data_augmentation(inputs)

    # Block 1
    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        name="block1_conv1",
    )(x)
    x = layers.BatchNormalization(name="block1_bn1")(x)
    x = layers.ReLU(name="block1_relu1")(x)

    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        name="block1_conv2",
    )(x)
    x = layers.BatchNormalization(name="block1_bn2")(x)
    x = layers.ReLU(name="block1_relu2")(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2),
        name="block1_pool",
    )(x)
    x = layers.Dropout(
        rate=0.25,
        name="block1_dropout",
    )(x)

    # Block 2
    x = layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        name="block2_conv1",
    )(x)
    x = layers.BatchNormalization(name="block2_bn1")(x)
    x = layers.ReLU(name="block2_relu1")(x)

    x = layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        name="block2_conv2",
    )(x)
    x = layers.BatchNormalization(name="block2_bn2")(x)
    x = layers.ReLU(name="block2_relu2")(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2),
        name="block2_pool",
    )(x)
    x = layers.Dropout(
        rate=0.30,
        name="block2_dropout",
    )(x)

    # Block 3
    x = layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        name="block3_conv1",
    )(x)
    x = layers.BatchNormalization(name="block3_bn1")(x)
    x = layers.ReLU(name="block3_relu1")(x)

    x = layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        name="block3_conv2",
    )(x)
    x = layers.BatchNormalization(name="block3_bn2")(x)
    x = layers.ReLU(name="block3_relu2")(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2),
        name="block3_pool",
    )(x)
    x = layers.Dropout(
        rate=0.40,
        name="block3_dropout",
    )(x)

    # Block 4
    x = layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        name="block4_conv1",
    )(x)
    x = layers.BatchNormalization(name="block4_bn1")(x)
    x = layers.ReLU(name="block4_relu1")(x)

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling",
    )(x)

    x = layers.Dense(
        units=128,
        activation="relu",
        name="dense_1",
    )(x)
    x = layers.Dropout(
        rate=0.50,
        name="dense_dropout",
    )(x)

    outputs = layers.Dense(
        units=number_of_classes,
        activation="softmax",
        name="emotion_output",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="improved_cnn",
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_model_parameter_count(model: tf.keras.Model) -> int:
    """
    Return the total number of trainable and non-trainable parameters.
    """

    return int(model.count_params())
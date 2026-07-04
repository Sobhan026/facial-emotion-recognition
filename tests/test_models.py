import numpy as np

from src.config import IMAGE_SHAPE, NUMBER_OF_CLASSES
from src.models import (
    build_baseline_cnn_model,
    build_improved_cnn_model,
    get_model_parameter_count,
)


def test_baseline_model_input_output_shape():
    model = build_baseline_cnn_model()

    assert model.input_shape == (None, *IMAGE_SHAPE)
    assert model.output_shape == (None, NUMBER_OF_CLASSES)


def test_baseline_model_parameter_count_is_positive():
    model = build_baseline_cnn_model()

    parameter_count = get_model_parameter_count(model)

    assert isinstance(parameter_count, int)
    assert parameter_count > 0


def test_baseline_model_prediction_shape():
    model = build_baseline_cnn_model()

    dummy_batch = np.random.rand(4, *IMAGE_SHAPE).astype(np.float32)

    predictions = model.predict(dummy_batch, verbose=0)

    assert predictions.shape == (4, NUMBER_OF_CLASSES)


def test_baseline_model_softmax_output():
    model = build_baseline_cnn_model()

    dummy_batch = np.random.rand(4, *IMAGE_SHAPE).astype(np.float32)

    predictions = model.predict(dummy_batch, verbose=0)

    assert np.all(predictions >= 0)
    assert np.all(predictions <= 1)

    probability_sums = predictions.sum(axis=1)

    assert np.allclose(probability_sums, 1.0, atol=1e-5)


def test_improved_model_input_output_shape():
    model = build_improved_cnn_model()

    assert model.input_shape == (None, *IMAGE_SHAPE)
    assert model.output_shape == (None, NUMBER_OF_CLASSES)


def test_improved_model_parameter_count_is_positive():
    model = build_improved_cnn_model()

    parameter_count = get_model_parameter_count(model)

    assert isinstance(parameter_count, int)
    assert parameter_count > 0


def test_improved_model_prediction_shape():
    model = build_improved_cnn_model()

    dummy_batch = np.random.rand(4, *IMAGE_SHAPE).astype(np.float32)

    predictions = model.predict(dummy_batch, verbose=0)

    assert predictions.shape == (4, NUMBER_OF_CLASSES)


def test_improved_model_softmax_output():
    model = build_improved_cnn_model()

    dummy_batch = np.random.rand(4, *IMAGE_SHAPE).astype(np.float32)

    predictions = model.predict(dummy_batch, verbose=0)

    assert np.all(predictions >= 0)
    assert np.all(predictions <= 1)

    probability_sums = predictions.sum(axis=1)

    assert np.allclose(probability_sums, 1.0, atol=1e-5)


def test_improved_model_contains_data_augmentation_layer():
    model = build_improved_cnn_model()

    layer_names = [layer.name for layer in model.layers]

    assert "data_augmentation" in layer_names


def test_improved_model_contains_global_average_pooling():
    model = build_improved_cnn_model()

    layer_names = [layer.name for layer in model.layers]

    assert "global_average_pooling" in layer_names
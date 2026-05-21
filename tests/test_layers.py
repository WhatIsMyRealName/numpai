from __future__ import annotations

import contextlib
import io
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from numpai import Network
from numpai.activations import tanh, tanh_prime
from numpai.layers import (
    ActivationLayer,
    BatchNormalization,
    Conv2D,
    Dropout,
    Embedding,
    FCLayer,
    FlattenLayer,
    GlobalAvgPool2D,
    Layer,
    LSTM,
    MaxPooling,
)
from numpai.layers.activation import ActivationLayer as ActivationLayerFromModule
from numpai.layers.convolution import Conv2D as Conv2DFromModule
from numpai.layers.dense import FCLayer as FCLayerFromModule
from numpai.layers.embedding import Embedding as EmbeddingFromModule
from numpai.layers.pooling import GlobalAvgPool2D as GlobalAvgPool2DFromModule
from numpai.lossfunction import mse, mse_prime


class LayerSplitTests(unittest.TestCase):
    def test_layers_are_exported_from_package_modules(self) -> None:
        self.assertIs(FCLayer, FCLayerFromModule)
        self.assertIs(Conv2D, Conv2DFromModule)
        self.assertIs(ActivationLayer, ActivationLayerFromModule)
        self.assertIs(Embedding, EmbeddingFromModule)
        self.assertIs(GlobalAvgPool2D, GlobalAvgPool2DFromModule)

        exported_layers = [
            ActivationLayer,
            BatchNormalization,
            Conv2D,
            Dropout,
            Embedding,
            FCLayer,
            FlattenLayer,
            GlobalAvgPool2D,
            LSTM,
            MaxPooling,
        ]
        for layer_type in exported_layers:
            self.assertTrue(issubclass(layer_type, Layer))

    def test_xor_network_can_predict_and_train_one_epoch(self) -> None:
        np.random.seed(0)
        x_train = np.array([[[0.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]], [[1.0, 1.0]]])
        y_train = np.array([[[0.0]], [[1.0]], [[1.0]], [[0.0]]])

        net = Network()
        net.add(FCLayer(2, 2, init_method="xavier"))
        net.add(ActivationLayer(tanh, tanh_prime))
        net.add(FCLayer(2, 1, init_method="xavier"))
        net.use(mse, mse_prime)

        before_training = net.predict(x_train)
        self.assertEqual(len(before_training), 4)
        self.assertEqual(before_training[0].shape, (1, 1))

        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            net.fit(x_train, y_train, epochs=1, learning_rate=0.1)

        after_training = net.predict(x_train)
        self.assertEqual(len(after_training), 4)
        self.assertEqual(after_training[0].shape, (1, 1))

    def test_convolution_pooling_and_flatten_stack_shapes(self) -> None:
        np.random.seed(0)
        input_data = np.random.randn(2, 1, 4, 4)

        conv = Conv2D(input_shape=(1, 4, 4), num_filters=3, filter_size=3, stride=1, padding=1)
        conv_output = conv.forward_propagation(input_data)
        self.assertEqual(conv_output.shape, (2, 3, 4, 4))

        pool = MaxPooling(pool_size=2, stride=2)
        pooled_output = pool.forward_propagation(conv_output)
        self.assertEqual(pooled_output.shape, (2, 3, 2, 2))

        global_pool = GlobalAvgPool2D()
        global_output = global_pool.forward_propagation(pooled_output)
        self.assertEqual(global_output.shape, (2, 3))

        flatten = FlattenLayer()
        flattened_output = flatten.forward_propagation(pooled_output)
        self.assertEqual(flattened_output.shape, (2, 12))

    def test_conv2d_padding_zero_keeps_valid_shapes(self) -> None:
        input_data = np.arange(9, dtype=float).reshape(1, 1, 3, 3)
        conv = Conv2D(input_shape=(1, 3, 3), num_filters=1, filter_size=2, stride=1, padding=0)

        output = conv.forward_propagation(input_data)
        input_error = conv.backward_propagation(np.ones_like(output), learning_rate=0.0, epoch=0)

        self.assertEqual(output.shape, (1, 1, 2, 2))
        self.assertEqual(input_error.shape, input_data.shape)

    def test_conv2d_input_gradient_matches_numeric_gradient(self) -> None:
        input_data = np.arange(9, dtype=float).reshape(1, 1, 3, 3) / 10
        conv = Conv2D(input_shape=(1, 3, 3), num_filters=1, filter_size=2, stride=1, padding=0)
        conv.filters = np.array([[[[0.2, -0.3], [0.4, 0.1]]]])
        conv.biases = np.zeros((1, 1, 1))
        upstream_gradient = np.array([[[[1.0, -0.5], [0.25, 0.75]]]])

        conv.forward_propagation(input_data)
        input_error = conv.backward_propagation(upstream_gradient, learning_rate=0.0, epoch=0)

        epsilon = 1e-6
        expected = np.zeros_like(input_data)
        for index in np.ndindex(input_data.shape):
            input_plus = input_data.copy()
            input_minus = input_data.copy()
            input_plus[index] += epsilon
            input_minus[index] -= epsilon
            loss_plus = np.sum(conv.forward_propagation(input_plus) * upstream_gradient)
            loss_minus = np.sum(conv.forward_propagation(input_minus) * upstream_gradient)
            expected[index] = (loss_plus - loss_minus) / (2 * epsilon)

        np.testing.assert_allclose(input_error, expected, rtol=1e-6, atol=1e-6)

    def test_pooling_layers_preserve_explicit_batch_dimension_of_one(self) -> None:
        input_data = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])

        max_pool = MaxPooling(pool_size=2, stride=2)
        max_output = max_pool.forward_propagation(input_data)
        max_input_error = max_pool.backward_propagation(np.ones_like(max_output), learning_rate=0.0, epoch=0)

        global_pool = GlobalAvgPool2D()
        global_output = global_pool.forward_propagation(input_data)
        global_input_error = global_pool.backward_propagation(np.ones_like(global_output), learning_rate=0.0, epoch=0)

        self.assertEqual(max_output.shape, (1, 1, 1, 1))
        self.assertEqual(max_input_error.shape, input_data.shape)
        self.assertEqual(global_output.shape, (1, 1))
        self.assertEqual(global_input_error.shape, input_data.shape)

    def test_dropout_backward_uses_inverted_dropout_scale(self) -> None:
        np.random.seed(0)
        layer = Dropout(dropout_rate=0.5)
        input_data = np.ones((2, 4))

        layer.forward_propagation(input_data, training=True)
        mask = layer.mask.copy()
        input_error = layer.backward_propagation(np.ones_like(input_data), learning_rate=0.0, epoch=0)

        np.testing.assert_array_equal(input_error, mask / 0.5)

    def test_dropout_rejects_full_dropout_rate(self) -> None:
        with self.assertRaises(ValueError):
            Dropout(dropout_rate=1.0)

    def test_batch_normalization_forward_shape(self) -> None:
        layer = BatchNormalization(input_size=3)
        input_data = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])

        output = layer.forward_propagation(input_data, training=True)

        self.assertEqual(output.shape, input_data.shape)

    def test_batch_normalization_single_image_backward_shape(self) -> None:
        layer = BatchNormalization(input_size=(2, 2, 2))
        input_data = np.ones((2, 2, 2))

        output = layer.forward_propagation(input_data, training=True)
        input_error = layer.backward_propagation(np.ones_like(output), learning_rate=0.1, epoch=0)

        self.assertEqual(output.shape, input_data.shape)
        self.assertEqual(input_error.shape, input_data.shape)

    def test_embedding_backward_accumulates_repeated_indices(self) -> None:
        layer = Embedding(vocab_size=3, embed_dim=2)
        layer.embeddings = np.zeros((3, 2))

        layer.forward_propagation(np.array([1, 1, 2]))
        layer.backward_propagation(np.ones((3, 2)), learning_rate=0.1, epoch=0)

        np.testing.assert_allclose(layer.embeddings[0], np.array([0.0, 0.0]))
        np.testing.assert_allclose(layer.embeddings[1], np.array([-0.2, -0.2]))
        np.testing.assert_allclose(layer.embeddings[2], np.array([-0.1, -0.1]))

    def test_lstm_forward_supports_batched_sequences(self) -> None:
        layer = LSTM(input_dim=3, hidden_dim=5)
        input_data = np.ones((2, 4, 3))

        output = layer.forward_propagation(input_data)

        self.assertEqual(output.shape, (2, 4, 5))


if __name__ == "__main__":
    unittest.main()

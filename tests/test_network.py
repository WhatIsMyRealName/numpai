from __future__ import annotations

import contextlib
import importlib
import io
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from numpai import Network
from numpai.activations import relu, relu_prime, softmax, softmax_prime_approx
from numpai.layers import (
    ActivationLayer,
    BatchNormalization,
    Conv2D,
    Dropout,
    Embedding,
    FCLayer,
    FlattenLayer,
    GlobalAvgPool2D,
    MaxPooling,
)
from numpai.lossfunction import mse, mse_prime


def build_linear_network() -> Network:
    net = Network()
    net.add(FCLayer(2, 1, init_method="xavier"))
    net.use(mse, mse_prime)
    return net


def import_onnx_or_skip():
    try:
        return importlib.import_module("onnx")
    except ModuleNotFoundError as exc:
        raise unittest.SkipTest("onnx n'est pas installe") from exc


class NetworkTests(unittest.TestCase):
    def test_add_requires_layer_instances(self) -> None:
        net = Network()

        with self.assertRaises(TypeError):
            net.add(object())

    def test_predict_returns_one_output_per_sample(self) -> None:
        np.random.seed(0)
        net = build_linear_network()
        x = np.array([[0.0, 1.0], [1.0, 0.0]])

        output = net.predict(x)

        self.assertEqual(len(output), 2)
        self.assertEqual(output[0].shape, (1,))

    def test_fit_emits_deprecation_warning(self) -> None:
        np.random.seed(0)
        net = build_linear_network()
        x = np.array([[0.0, 1.0], [1.0, 0.0]])
        y = np.array([[1.0], [1.0]])

        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertWarnsRegex(DeprecationWarning, "fit"):
                net.fit(x, y, epochs=1, learning_rate=0.01)

    def test_fit2_emits_deprecation_warning(self) -> None:
        np.random.seed(0)
        net = build_linear_network()
        x = np.array([[0.0, 1.0], [1.0, 0.0]])
        y = np.array([[1.0], [1.0]])

        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertWarnsRegex(DeprecationWarning, "fit2"):
                net.fit2(x, y, epochs=1, learning_rate=0.01, batch_size=2)

        self.assertEqual(len(net.logs), 1)

    def test_fit3_records_training_and_validation_logs(self) -> None:
        np.random.seed(0)
        net = build_linear_network()
        x = np.array([[0.0, 1.0], [1.0, 0.0]])
        y = np.array([[1.0], [1.0]])

        with contextlib.redirect_stdout(io.StringIO()):
            net.fit3(x, y, x, y, epochs=1, learning_rate=0.01, batch_size=2)

        self.assertEqual(len(net.logs), 1)
        self.assertEqual(len(net.logs_test), 1)

    def test_training_requires_loss_configuration(self) -> None:
        net = Network()
        net.add(FCLayer(2, 1, init_method="xavier"))

        with self.assertRaises(ValueError):
            net.fit3(
                np.array([[0.0, 1.0]]),
                np.array([[1.0]]),
                np.array([[0.0, 1.0]]),
                np.array([[1.0]]),
                epochs=1,
                learning_rate=0.01,
            )

    def test_summary_reports_parameter_counts(self) -> None:
        np.random.seed(0)
        net = Network()
        net.add(FCLayer(2, 3, init_method="xavier"))

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            net.summary()

        summary = output.getvalue()
        self.assertIn("Layer 1: FCLayer", summary)
        self.assertIn("Parameters: 9", summary)
        self.assertIn("Total Parameters: 9", summary)

    def test_save_and_load_round_trip(self) -> None:
        np.random.seed(0)
        net = build_linear_network()
        x = np.array([[0.0, 1.0]])
        filename = Path("tests") / "_tmp_network.pkl"

        try:
            net.save(filename, debug=False)
            loaded = Network.load(filename, debug=False)
        finally:
            if filename.exists():
                filename.unlink()

        np.testing.assert_allclose(loaded.predict(x), net.predict(x))

    def test_export_fuses_batchnorm_and_disables_training(self) -> None:
        net = Network()
        fc = FCLayer(2, 2, init_method="random")
        fc.weights = np.array([[1.0, 2.0], [3.0, 4.0]])
        fc.biases = np.array([[0.5, -0.5]])

        batchnorm = BatchNormalization(input_size=2, epsilon=0.0)
        batchnorm.running_mean = np.array([[0.5, 1.0]])
        batchnorm.running_var = np.array([[3.0, 8.0]])
        batchnorm.gamma = np.array([[2.0, 3.0]])
        batchnorm.beta = np.array([[1.0, -1.0]])

        net.add(fc)
        net.add(batchnorm)
        net.add(Dropout(0.25))
        net.use(mse, mse_prime)

        net.export()

        self.assertEqual(len(net.layers), 1)
        self.assertIs(net.layers[0], fc)
        self.assertFalse(hasattr(fc, "m_weights"))
        self.assertFalse(hasattr(net, "loss"))
        self.assertTrue(net._inference_only)
        np.testing.assert_allclose(
            fc.weights,
            np.array([[2 / np.sqrt(3), 6 / np.sqrt(8)], [6 / np.sqrt(3), 12 / np.sqrt(8)]]),
        )
        np.testing.assert_allclose(fc.biases, np.array([[1.0, (-1.5 * 3 / np.sqrt(8)) - 1.0]]))

        with self.assertRaises(RuntimeError):
            net.add(FCLayer(2, 1, init_method="xavier"))

    def test_export_to_onnx_rejects_unsupported_layers(self) -> None:
        import_onnx_or_skip()
        net = Network()
        net.add(Embedding(vocab_size=10, embed_dim=3))

        with self.assertRaises(NotImplementedError):
            net.export_to_onnx(Path("tests") / "_tmp_unsupported.onnx", input_shape=(1,))

    def test_export_to_onnx_writes_checked_model_when_onnx_is_available(self) -> None:
        onnx = import_onnx_or_skip()
        filename = Path("tests") / "_tmp_network.onnx"
        net = Network()
        net.add(Conv2D(input_shape=(1, 4, 4), num_filters=2, filter_size=3, padding=1))
        net.add(BatchNormalization(input_size=(2, 4, 4)))
        net.add(ActivationLayer(relu, relu_prime))
        net.add(MaxPooling(pool_size=2, stride=2))
        net.add(FlattenLayer())
        net.add(FCLayer(8, 3, init_method="xavier"))
        net.add(ActivationLayer(softmax, softmax_prime_approx))

        try:
            net.export_to_onnx(filename, input_shape=(1, 1, 4, 4))
            model = onnx.load(filename)
            onnx.checker.check_model(model)
        finally:
            if filename.exists():
                filename.unlink()

        op_types = [node.op_type for node in model.graph.node]
        self.assertEqual(
            op_types,
            ["Conv", "BatchNormalization", "Relu", "MaxPool", "Flatten", "Gemm", "Softmax"],
        )

    def test_export_to_onnx_supports_global_average_pool_when_onnx_is_available(self) -> None:
        onnx = import_onnx_or_skip()
        filename = Path("tests") / "_tmp_global_pool.onnx"
        net = Network()
        net.add(Conv2D(input_shape=(1, 4, 4), num_filters=2, filter_size=3, padding=1))
        net.add(GlobalAvgPool2D())
        net.add(FCLayer(2, 1, init_method="xavier"))

        try:
            net.export_to_onnx(filename, input_shape=(1, 1, 4, 4))
            model = onnx.load(filename)
            onnx.checker.check_model(model)
        finally:
            if filename.exists():
                filename.unlink()

        self.assertEqual([node.op_type for node in model.graph.node], ["Conv", "GlobalAveragePool", "Flatten", "Gemm"])


if __name__ == "__main__":
    unittest.main()

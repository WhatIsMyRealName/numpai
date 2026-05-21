from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from numpai import Network
from numpai.activations import tanh, tanh_prime
from numpai.layers import ActivationLayer, FCLayer
from numpai.lossfunction import mse, mse_prime


def build_network() -> Network:
    net = Network()
    net.add(FCLayer(2, 4, init_method="xavier"))
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(4, 1, init_method="xavier"))
    net.add(ActivationLayer(tanh, tanh_prime))
    net.use(mse, mse_prime)
    return net


def predictions(net: Network, x_train: np.ndarray) -> np.ndarray:
    return np.array(net.predict(x_train)).reshape(-1)


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true.reshape(-1)) ** 2))


def print_state(epoch: int, y_train: np.ndarray, y_pred: np.ndarray) -> None:
    loss = mean_squared_error(y_train, y_pred)
    formatted_predictions = " ".join(f"{value:.4f}" for value in y_pred)
    print(f"epoch={epoch:4d} mse={loss:.6f} predictions=[{formatted_predictions}]")


def main() -> None:
    np.random.seed(0)

    x_train = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y_train = np.array([[0.0], [1.0], [1.0], [0.0]])

    net = build_network()

    print("XOR targets:      [0.0000 1.0000 1.0000 0.0000]")
    print_state(0, y_train, predictions(net, x_train))

    for epoch in range(100, 1001, 100):
        with contextlib.redirect_stdout(io.StringIO()):
            net.fit2(x_train, y_train, epochs=100, learning_rate=0.1, batch_size=4)
        print_state(epoch, y_train, predictions(net, x_train))


if __name__ == "__main__":
    main()

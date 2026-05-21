from __future__ import annotations

import numpy as np

from .base import Layer


class FCLayer(Layer):
    def __init__(self, input_size: int, output_size: int, init_method: str = "he") -> None:
        super().__init__()
        if init_method not in {"he", "xavier", "random"}:
            raise ValueError(f"init_method inconnu: {init_method}")

        self.biases = np.random.randn(1, output_size) * 0.1
        if init_method == "he":
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.5 / input_size)
        elif init_method == "xavier":
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1.5 / input_size)
        else:
            self.weights = np.random.randn(input_size, output_size) * 0.1

        self.m_weights = np.zeros((input_size, output_size))
        self.v_weights = np.zeros((input_size, output_size))
        self.m_biases = np.zeros((1, output_size))
        self.v_biases = np.zeros((1, output_size))
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 1

    def forward_propagation(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = np.atleast_2d(input_data)
        self.output = self.input @ self.weights + self.biases
        return self.output if input_data.ndim > 1 else self.output.flatten()

    def backward_propagation(self, output_error: np.ndarray, learning_rate: float, epoch: int) -> np.ndarray:
        output_error = np.atleast_2d(output_error)
        batch_size = self.input.shape[0]

        d_weights = self.input.T @ output_error / batch_size
        d_biases = np.mean(output_error, axis=0, keepdims=True)
        d_weights = np.clip(d_weights, -1.0, 1.0)
        d_biases = np.clip(d_biases, -1.0, 1.0)

        input_error = output_error @ self.weights.T

        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * d_weights
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (d_weights**2)
        m_hat_weights = self.m_weights / (1 - self.beta1**self.t)
        v_hat_weights = self.v_weights / (1 - self.beta2**self.t)
        self.weights -= learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)

        self.m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * d_biases
        self.v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * (d_biases**2)
        m_hat_biases = self.m_biases / (1 - self.beta1**self.t)
        v_hat_biases = self.v_biases / (1 - self.beta2**self.t)
        self.biases -= learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)

        self.t += 1
        return input_error

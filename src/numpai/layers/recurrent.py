from __future__ import annotations

import numpy as np

from .base import Layer


class LSTM(Layer):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W = np.random.randn(4, input_dim + hidden_dim, hidden_dim)
        self.b = np.random.randn(4, hidden_dim)

    def forward_propagation(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        batch_size = input_data.shape[0]
        self.h = np.zeros((batch_size, self.hidden_dim))
        self.c = np.zeros((batch_size, self.hidden_dim))
        self.outputs = []

        for t in range(input_data.shape[1]):
            x_t = input_data[:, t, :]
            combined = np.concatenate((x_t, self.h), axis=1)
            i_t = self._sigmoid(combined @ self.W[0] + self.b[0])
            f_t = self._sigmoid(combined @ self.W[1] + self.b[1])
            o_t = self._sigmoid(combined @ self.W[2] + self.b[2])
            g_t = np.tanh(combined @ self.W[3] + self.b[3])
            self.c = f_t * self.c + i_t * g_t
            self.h = o_t * np.tanh(self.c)
            self.outputs.append(self.h)

        return np.stack(self.outputs, axis=1)

    def backward_propagation(self, output_error: np.ndarray, learning_rate: float, epoch: int) -> np.ndarray:
        # TODO: implement backpropagation through time before using LSTM for training.
        raise NotImplementedError

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

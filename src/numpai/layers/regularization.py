from __future__ import annotations

import numpy as np

from .base import Layer


class Dropout(Layer):
    def __init__(self, dropout_rate: float) -> None:
        super().__init__()
        if not 0 <= dropout_rate < 1:
            raise ValueError("dropout_rate doit etre dans [0, 1)")
        self.dropout_rate = dropout_rate
        self.mask: np.ndarray | None = None

    def forward_propagation(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        if not training or self.dropout_rate == 0:
            self.mask = None
            return input_data

        keep_probability = 1 - self.dropout_rate
        self.mask = np.random.binomial(1, keep_probability, size=input_data.shape)
        return input_data * self.mask / keep_probability

    def backward_propagation(self, output_error: np.ndarray, learning_rate: float, epoch: int) -> np.ndarray:
        if self.mask is None:
            return output_error

        keep_probability = 1 - self.dropout_rate
        input_error = output_error * self.mask / keep_probability
        self.mask = None
        return input_error

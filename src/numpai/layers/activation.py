from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .base import Layer


class ActivationLayer(Layer):
    def __init__(
        self,
        activation: Callable[[np.ndarray], np.ndarray],
        activation_prime: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = input_data
        self.output = self.activation(input_data)
        return self.output

    def backward_propagation(self, output_error: np.ndarray, learning_rate: float, epoch: int) -> np.ndarray:
        return self.activation_prime(self.input) * output_error

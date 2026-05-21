from __future__ import annotations

import numpy as np

from .base import Layer


class FlattenLayer(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.input_shape: tuple[int, ...] | None = None

    def forward_propagation(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        self.input_shape = input_data.shape
        if input_data.ndim == 3:
            return input_data.reshape(-1)
        if input_data.ndim == 4:
            return input_data.reshape(input_data.shape[0], -1)
        raise ValueError(f"FlattenLayer attend une entree 3D ou 4D, recu {input_data.ndim}D")

    def backward_propagation(self, output_error: np.ndarray, learning_rate: float, epoch: int) -> np.ndarray:
        return output_error.reshape(self.input_shape)

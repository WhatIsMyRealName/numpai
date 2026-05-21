from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import as_strided

from .base import Layer


class MaxPooling(Layer):
    def __init__(self, pool_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        if pool_size <= 0:
            raise ValueError("pool_size doit etre strictement positif")
        if stride <= 0:
            raise ValueError("stride doit etre strictement positif")

        self.pool_size = pool_size
        self.stride = stride
        self.input_had_batch = True
        self.max_indices: np.ndarray | None = None

    def forward_propagation(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        self.input_had_batch = input_data.ndim == 4
        input_batch = input_data if self.input_had_batch else input_data[np.newaxis, ...]
        self.input = input_batch
        batch_size, channels, height, width = input_batch.shape

        output_height = (height - self.pool_size) // self.stride + 1
        output_width = (width - self.pool_size) // self.stride + 1
        shape = (batch_size, channels, output_height, output_width, self.pool_size, self.pool_size)
        strides = (
            input_batch.strides[0],
            input_batch.strides[1],
            self.stride * input_batch.strides[2],
            self.stride * input_batch.strides[3],
            input_batch.strides[2],
            input_batch.strides[3],
        )
        windows = as_strided(input_batch, shape=shape, strides=strides, writeable=False)

        flattened_windows = windows.reshape(batch_size, channels, output_height, output_width, -1)
        self.max_indices = np.argmax(flattened_windows, axis=-1)
        self.output = np.max(windows, axis=(-2, -1))
        return self.output if self.input_had_batch else self.output[0]

    def backward_propagation(self, output_error: np.ndarray, learning_rate: float, epoch: int) -> np.ndarray:
        output_error_batch = output_error if output_error.ndim == 4 else output_error[np.newaxis, ...]
        input_error = np.zeros_like(self.input)

        batch_size, channels, output_height, output_width = output_error_batch.shape
        max_i, max_j = np.unravel_index(self.max_indices, (self.pool_size, self.pool_size))

        batch_idx = np.arange(batch_size)[:, None, None, None]
        channel_idx = np.arange(channels)[None, :, None, None]
        out_y = np.arange(output_height)[None, None, :, None] * self.stride
        out_x = np.arange(output_width)[None, None, None, :] * self.stride

        input_error[batch_idx, channel_idx, out_y + max_i, out_x + max_j] = output_error_batch
        return input_error if self.input_had_batch else input_error[0]


class GlobalAvgPool2D(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.input_shape: tuple[int, ...] | None = None
        self.input_had_batch = True

    def forward_propagation(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        self.input_had_batch = input_data.ndim == 4
        input_batch = input_data if self.input_had_batch else input_data[np.newaxis, ...]
        self.input_shape = input_batch.shape
        self.output = input_batch.mean(axis=(2, 3))
        return self.output if self.input_had_batch else self.output[0]

    def backward_propagation(self, output_error: np.ndarray, learning_rate: float, epoch: int) -> np.ndarray:
        output_error_batch = output_error if output_error.ndim == 2 else output_error[np.newaxis, ...]
        _, _, height, width = self.input_shape
        input_error = np.ones(self.input_shape) * output_error_batch[:, :, None, None] / (height * width)
        return input_error if self.input_had_batch else input_error[0]

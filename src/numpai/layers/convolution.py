from __future__ import annotations

import numpy as np

from .base import Layer


class Conv2D(Layer):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_filters: int,
        filter_size: int,
        stride: int = 1,
        padding: int = 0,
        init_method: str = "he",
    ) -> None:
        super().__init__()
        if init_method not in {"he", "xavier", "random"}:
            raise ValueError(f"init_method inconnu: {init_method}")
        if filter_size <= 0:
            raise ValueError("filter_size doit etre strictement positif")
        if stride <= 0:
            raise ValueError("stride doit etre strictement positif")
        if padding < 0:
            raise ValueError("padding doit etre positif")

        self.input_shape = input_shape
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        channels = input_shape[0]
        self.biases = np.random.uniform(-0.1, 0.1, size=(num_filters, 1, 1))
        if init_method == "he":
            scale = np.sqrt(2 / (channels * filter_size * filter_size))
            self.filters = np.random.randn(num_filters, channels, filter_size, filter_size) * scale
        elif init_method == "xavier":
            scale = np.sqrt(1 / (channels * filter_size * filter_size))
            self.filters = np.random.randn(num_filters, channels, filter_size, filter_size) * scale
        else:
            self.filters = np.random.uniform(-0.1, 0.1, size=(num_filters, channels, filter_size, filter_size))

        self.input_had_batch = True
        self.padded_input: np.ndarray | None = None

    def _pad(self, input_data: np.ndarray) -> np.ndarray:
        if self.padding == 0:
            return input_data
        return np.pad(
            input_data,
            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            mode="constant",
        )

    def forward_propagation(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        self.input_had_batch = input_data.ndim == 4
        input_batch = input_data if self.input_had_batch else input_data[np.newaxis, ...]
        self.input = input_batch
        self.padded_input = self._pad(input_batch)

        batch_size, channels, height, width = self.padded_input.shape
        filter_size = self.filter_size
        output_height = (height - filter_size) // self.stride + 1
        output_width = (width - filter_size) // self.stride + 1
        output = np.empty((batch_size, self.num_filters, output_height, output_width))

        for y in range(output_height):
            y_start = y * self.stride
            y_end = y_start + filter_size
            for x in range(output_width):
                x_start = x * self.stride
                x_end = x_start + filter_size
                window = self.padded_input[:, :, y_start:y_end, x_start:x_end]
                output[:, :, y, x] = np.einsum("bchw,fchw->bf", window, self.filters, optimize=True)

        self.output = output + self.biases
        return self.output if self.input_had_batch else self.output[0]

    def backward_propagation(self, output_error: np.ndarray, learning_rate: float, epoch: int) -> np.ndarray:
        output_error_batch = output_error if output_error.ndim == 4 else output_error[np.newaxis, ...]
        filter_size = self.filter_size
        _, _, output_height, output_width = output_error_batch.shape

        d_filters = np.zeros_like(self.filters)
        d_biases = np.sum(output_error_batch, axis=(0, 2, 3))
        padded_input_error = np.zeros_like(self.padded_input)

        for y in range(output_height):
            y_start = y * self.stride
            y_end = y_start + filter_size
            for x in range(output_width):
                x_start = x * self.stride
                x_end = x_start + filter_size
                error_slice = output_error_batch[:, :, y, x]
                input_window = self.padded_input[:, :, y_start:y_end, x_start:x_end]
                d_filters += np.einsum("bf,bchw->fchw", error_slice, input_window, optimize=True)
                padded_input_error[:, :, y_start:y_end, x_start:x_end] += np.einsum(
                    "bf,fchw->bchw",
                    error_slice,
                    self.filters,
                    optimize=True,
                )

        input_error = (
            padded_input_error[:, :, self.padding : -self.padding, self.padding : -self.padding]
            if self.padding > 0
            else padded_input_error
        )

        self.filters -= learning_rate * d_filters
        self.biases -= learning_rate * d_biases[:, np.newaxis, np.newaxis]

        return input_error if self.input_had_batch else input_error[0]

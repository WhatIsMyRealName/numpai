from __future__ import annotations

import numpy as np

from .base import Layer


class BatchNormalization(Layer):
    def __init__(self, input_size: int | tuple[int, int, int], momentum: float = 0.9, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.momentum = momentum
        self.epsilon = epsilon
        self.input_size = input_size

        if isinstance(input_size, int):
            feature_shape = (1, input_size)
        elif len(input_size) == 3:
            feature_shape = (1, input_size[0], 1, 1)
        else:
            raise ValueError(f"BatchNormalization ne supporte pas input_size={input_size}")

        self.gamma = np.ones(feature_shape)
        self.beta = np.zeros(feature_shape)
        self.running_mean = np.zeros(feature_shape)
        self.running_var = np.ones(feature_shape)
        self.normalized_input: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.variance: np.ndarray | None = None
        self.reduce_axes: tuple[int, ...] | None = None
        self.uses_batch_stats = False

    def forward_propagation(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = input_data
        self.normalized_input = None
        self.reduce_axes = None
        self.uses_batch_stats = False

        if input_data.ndim == 1 or input_data.ndim == 3:
            return input_data
        if input_data.ndim == 2:
            self.reduce_axes = (0,)
        elif input_data.ndim == 4:
            self.reduce_axes = (0, 2, 3)
        else:
            raise ValueError(f"BatchNormalization ne supporte pas input_data.ndim={input_data.ndim}")

        batch_size = input_data.shape[0]
        if training and batch_size > 1:
            self.mean = np.mean(input_data, axis=self.reduce_axes, keepdims=True)
            self.variance = np.var(input_data, axis=self.reduce_axes, keepdims=True)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.variance
            self.uses_batch_stats = True
        else:
            self.mean = self.running_mean
            self.variance = self.running_var

        self.normalized_input = (input_data - self.mean) / np.sqrt(self.variance + self.epsilon)
        self.output = self.gamma * self.normalized_input + self.beta
        return self.output

    def backward_propagation(self, dout: np.ndarray, learning_rate: float, epoch: int) -> np.ndarray:
        if self.normalized_input is None or self.reduce_axes is None:
            return dout

        dgamma = np.sum(dout * self.normalized_input, axis=self.reduce_axes, keepdims=True)
        dbeta = np.sum(dout, axis=self.reduce_axes, keepdims=True)
        dx_hat = dout * self.gamma

        if self.uses_batch_stats:
            elements_count = np.prod([self.input.shape[axis] for axis in self.reduce_axes])
            inv_std = 1 / np.sqrt(self.variance + self.epsilon)
            centered_input = self.input - self.mean
            dvar = np.sum(dx_hat * centered_input * -0.5 * inv_std**3, axis=self.reduce_axes, keepdims=True)
            dmean = np.sum(-dx_hat * inv_std, axis=self.reduce_axes, keepdims=True)
            dmean += dvar * np.mean(-2 * centered_input, axis=self.reduce_axes, keepdims=True)
            input_error = dx_hat * inv_std + dvar * 2 * centered_input / elements_count + dmean / elements_count
        else:
            input_error = dx_hat / np.sqrt(self.variance + self.epsilon)

        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta
        return input_error

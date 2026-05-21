from __future__ import annotations

import numpy as np
import scipy.special


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_prime(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x) ** 2


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_prime(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1.0, 0.0)


def leakyrelu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)


def leakyrelu_prime(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, 1.0, alpha)


def gelu_exact(x: np.ndarray) -> np.ndarray:
    return x * 0.5 * (1 + scipy.special.erf(x / np.sqrt(2)))


def gelu_exact_prime(x: np.ndarray) -> np.ndarray:
    phi_x = 0.5 * (1 + scipy.special.erf(x / np.sqrt(2)))
    pdf_x = np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)
    return phi_x + x * pdf_x


def gelu_approx1(x: np.ndarray) -> np.ndarray:
    c = np.sqrt(2 / np.pi)
    return 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * x**3)))


def gelu_approx1_prime(x: np.ndarray) -> np.ndarray:
    c = np.sqrt(2 / np.pi)
    tanh_argument = c * (x + 0.044715 * x**3)
    tanh_term = np.tanh(tanh_argument)
    sech2_term = 1 - tanh_term**2
    argument_prime = c * (1 + 3 * 0.044715 * x**2)
    return 0.5 * (1 + tanh_term) + 0.5 * x * sech2_term * argument_prime


def gelu_approx2(x: np.ndarray) -> np.ndarray:
    return x / (1 + np.exp(-1.702 * x))


def gelu_approx2_prime(x: np.ndarray) -> np.ndarray:
    sig = 1 / (1 + np.exp(-1.702 * x))
    return sig + 1.702 * x * sig * (1 - sig)


def swish(x: np.ndarray) -> np.ndarray:
    return x / (1 + np.exp(-x))


def swish_prime(x: np.ndarray) -> np.ndarray:
    sig = 1 / (1 + np.exp(-x))
    return sig + x * sig * (1 - sig)


def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def softmax_prime_approx(x: np.ndarray) -> np.ndarray:
    s = softmax(x)
    return s * (1 - s)


def softmax_prime(x: np.ndarray, dL_dS: np.ndarray) -> np.ndarray:
    s = softmax(x)
    dot = np.sum(dL_dS * s, axis=-1, keepdims=True)
    return s * (dL_dS - dot)


__all__ = [
    "gelu_approx1",
    "gelu_approx1_prime",
    "gelu_approx2",
    "gelu_approx2_prime",
    "gelu_exact",
    "gelu_exact_prime",
    "leakyrelu",
    "leakyrelu_prime",
    "relu",
    "relu_prime",
    "softmax",
    "softmax_prime",
    "softmax_prime_approx",
    "swish",
    "swish_prime",
    "tanh",
    "tanh_prime",
]

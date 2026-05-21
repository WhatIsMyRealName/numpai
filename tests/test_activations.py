from __future__ import annotations

import sys
import unittest
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from numpai.activations import (
    gelu_approx1,
    gelu_approx1_prime,
    gelu_approx2,
    gelu_approx2_prime,
    gelu_exact,
    gelu_exact_prime,
    leakyrelu,
    leakyrelu_prime,
    relu,
    relu_prime,
    softmax,
    softmax_prime,
    softmax_prime_approx,
    swish,
    swish_prime,
    tanh,
    tanh_prime,
)


def numerical_prime(function: Callable[[np.ndarray], np.ndarray], x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    return (function(x + epsilon) - function(x - epsilon)) / (2 * epsilon)


class ActivationTests(unittest.TestCase):
    def test_elementwise_activation_primes_match_numeric_gradients(self) -> None:
        x = np.array([[-2.0, -0.7, 0.3, 1.5]])
        checks = [
            (tanh, tanh_prime, 1e-6),
            (relu, relu_prime, 1e-6),
            (leakyrelu, leakyrelu_prime, 1e-6),
            (gelu_exact, gelu_exact_prime, 1e-6),
            (gelu_approx1, gelu_approx1_prime, 1e-6),
            (gelu_approx2, gelu_approx2_prime, 1e-6),
            (swish, swish_prime, 1e-6),
        ]

        for function, prime, tolerance in checks:
            with self.subTest(function=function.__name__):
                np.testing.assert_allclose(prime(x), numerical_prime(function, x), rtol=tolerance, atol=tolerance)

    def test_softmax_rows_sum_to_one(self) -> None:
        x = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])

        output = softmax(x)

        np.testing.assert_allclose(np.sum(output, axis=-1), np.ones(2))

    def test_softmax_prime_matches_numeric_vector_jacobian_product(self) -> None:
        x = np.array([[0.2, -0.3, 1.1], [1.7, -0.8, 0.4]])
        upstream_gradient = np.array([[0.5, -1.2, 0.7], [-0.4, 0.9, 0.2]])
        epsilon = 1e-6
        expected = np.zeros_like(x)

        for index in np.ndindex(x.shape):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[index] += epsilon
            x_minus[index] -= epsilon
            loss_plus = np.sum(softmax(x_plus) * upstream_gradient)
            loss_minus = np.sum(softmax(x_minus) * upstream_gradient)
            expected[index] = (loss_plus - loss_minus) / (2 * epsilon)

        np.testing.assert_allclose(softmax_prime(x, upstream_gradient), expected, rtol=1e-6, atol=1e-6)

    def test_softmax_prime_approx_returns_diagonal_terms(self) -> None:
        x = np.array([[0.2, -0.3, 1.1]])
        output = softmax(x)

        np.testing.assert_allclose(softmax_prime_approx(x), output * (1 - output))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from numpai.lossfunction import (
    categorical_hinge_loss,
    categorical_hinge_loss_prime,
    cross_entropy_loss,
    cross_entropy_loss_prime,
    focal_loss,
    focal_loss_prime,
    gce_loss_prime,
    generalized_cross_entropy,
    hinge_loss,
    hinge_loss_prime,
    js_divergence,
    js_divergence_prime,
    kl_divergence,
    kl_divergence_prime,
    mse,
    mse_prime,
    squared_hinge_loss,
    squared_hinge_loss_prime,
)


LossFunction = Callable[[np.ndarray, np.ndarray], np.float64]
LossPrime = Callable[[np.ndarray, np.ndarray], np.ndarray]


def numerical_gradient(
    loss: LossFunction,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    gradient = np.zeros_like(y_pred, dtype=float)
    for index in np.ndindex(y_pred.shape):
        y_pred_plus = y_pred.copy()
        y_pred_minus = y_pred.copy()
        y_pred_plus[index] += epsilon
        y_pred_minus[index] -= epsilon
        gradient[index] = (loss(y_true, y_pred_plus) - loss(y_true, y_pred_minus)) / (2 * epsilon)
    return gradient


class LossFunctionTests(unittest.TestCase):
    def test_smooth_loss_primes_match_numeric_gradients(self) -> None:
        y_true = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        y_pred = np.array([[0.2, 0.7, 0.1], [0.6, 0.3, 0.1]])
        checks: list[tuple[str, LossFunction, LossPrime, float]] = [
            ("mse", mse, mse_prime, 1e-6),
            ("cross_entropy_loss", cross_entropy_loss, cross_entropy_loss_prime, 1e-6),
            ("kl_divergence", kl_divergence, kl_divergence_prime, 1e-6),
            ("generalized_cross_entropy", generalized_cross_entropy, gce_loss_prime, 1e-6),
            ("focal_loss", focal_loss, focal_loss_prime, 1e-6),
            ("js_divergence", js_divergence, js_divergence_prime, 1e-6),
        ]

        for name, loss, prime, tolerance in checks:
            with self.subTest(loss=name):
                np.testing.assert_allclose(
                    prime(y_true, y_pred),
                    numerical_gradient(loss, y_true, y_pred),
                    rtol=tolerance,
                    atol=tolerance,
                )

    def test_hinge_primes_match_numeric_gradients_away_from_kinks(self) -> None:
        y_true = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        y_pred = np.array([[0.2, 0.7, 0.1], [0.8, 0.4, 0.3]])
        checks: list[tuple[str, LossFunction, LossPrime]] = [
            ("hinge_loss", hinge_loss, hinge_loss_prime),
            ("categorical_hinge_loss", categorical_hinge_loss, categorical_hinge_loss_prime),
            ("squared_hinge_loss", squared_hinge_loss, squared_hinge_loss_prime),
        ]

        for name, loss, prime in checks:
            with self.subTest(loss=name):
                np.testing.assert_allclose(
                    prime(y_true, y_pred),
                    numerical_gradient(loss, y_true, y_pred),
                    rtol=1e-6,
                    atol=1e-6,
                )


if __name__ == "__main__":
    unittest.main()

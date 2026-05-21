from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    return np.sum((y_true - y_pred) ** 2)


def mse_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return 2 * (y_pred - y_true)


def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-12) -> np.float64:
    clipped_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return -np.sum(y_true * np.log(clipped_pred))


def cross_entropy_loss_prime(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    clipped_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return -(y_true / clipped_pred)


def kl_divergence(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-12) -> np.float64:
    clipped_true = np.clip(y_true, epsilon, 1.0)
    clipped_pred = np.clip(y_pred, epsilon, 1.0)
    return np.sum(clipped_true * np.log(clipped_true / clipped_pred))


def kl_divergence_prime(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    clipped_true = np.clip(y_true, epsilon, 1.0)
    clipped_pred = np.clip(y_pred, epsilon, 1.0)
    return -(clipped_true / clipped_pred)


def hinge_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> np.float64:
    correct_class_scores = np.sum(y_true * y_pred, axis=-1, keepdims=True)
    margins = np.maximum(0.0, y_pred - correct_class_scores + delta)
    margins *= 1 - y_true
    return np.sum(margins)


def hinge_loss_prime(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> np.ndarray:
    correct_class_scores = np.sum(y_true * y_pred, axis=-1, keepdims=True)
    active_margins = ((y_pred - correct_class_scores + delta) > 0) * (1 - y_true)
    grad = active_margins.astype(float)
    grad -= y_true * np.sum(active_margins, axis=-1, keepdims=True)
    return grad


def categorical_hinge_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> np.float64:
    pos = np.sum(y_true * y_pred, axis=-1)
    neg = np.max(np.where(y_true == 1, -np.inf, y_pred), axis=-1)
    return np.sum(np.maximum(0.0, neg - pos + delta))


def categorical_hinge_loss_prime(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> np.ndarray:
    pos = np.sum(y_true * y_pred, axis=-1, keepdims=True)
    neg_scores = np.where(y_true == 1, -np.inf, y_pred)
    neg = np.max(neg_scores, axis=-1, keepdims=True)
    active = neg - pos + delta > 0
    neg_mask = neg_scores == neg
    grad = active * neg_mask.astype(float)
    grad -= active * y_true
    return grad


def generalized_cross_entropy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gamma: float = 0.7,
    epsilon: float = 1e-12,
) -> np.float64:
    clipped_pred = np.clip(y_pred, epsilon, 1.0)
    return np.sum(((1 - clipped_pred**gamma) / gamma) * y_true)


def gce_loss_prime(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gamma: float = 0.7,
    epsilon: float = 1e-12,
) -> np.ndarray:
    clipped_pred = np.clip(y_pred, epsilon, 1.0)
    return -y_true * clipped_pred ** (gamma - 1)


def focal_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gamma: float = 2.0,
    epsilon: float = 1e-12,
) -> np.float64:
    clipped_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return -np.sum(((1 - clipped_pred) ** gamma) * y_true * np.log(clipped_pred))


def focal_loss_prime(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gamma: float = 2.0,
    epsilon: float = 1e-12,
) -> np.ndarray:
    clipped_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    modulation = (1 - clipped_pred) ** gamma
    modulation_prime = gamma * (1 - clipped_pred) ** (gamma - 1) * np.log(clipped_pred)
    return y_true * (modulation_prime - modulation / clipped_pred)


def squared_hinge_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> np.float64:
    correct_class_scores = np.sum(y_true * y_pred, axis=-1, keepdims=True)
    margins = np.maximum(0.0, y_pred - correct_class_scores + delta)
    margins *= 1 - y_true
    return np.sum(margins**2)


def squared_hinge_loss_prime(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> np.ndarray:
    correct_class_scores = np.sum(y_true * y_pred, axis=-1, keepdims=True)
    margins = np.maximum(0.0, y_pred - correct_class_scores + delta)
    margins *= 1 - y_true
    grad = 2 * margins
    grad -= y_true * np.sum(2 * margins, axis=-1, keepdims=True)
    return grad


def js_divergence(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-12) -> np.float64:
    clipped_true = np.clip(y_true, epsilon, 1.0)
    clipped_pred = np.clip(y_pred, epsilon, 1.0)
    midpoint = 0.5 * (clipped_true + clipped_pred)
    return 0.5 * np.sum(clipped_true * np.log(clipped_true / midpoint)) + 0.5 * np.sum(
        clipped_pred * np.log(clipped_pred / midpoint)
    )


def js_divergence_prime(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    clipped_true = np.clip(y_true, epsilon, 1.0)
    clipped_pred = np.clip(y_pred, epsilon, 1.0)
    midpoint = 0.5 * (clipped_true + clipped_pred)
    return 0.5 * np.log(clipped_pred / midpoint)


__all__ = [
    "categorical_hinge_loss",
    "categorical_hinge_loss_prime",
    "cross_entropy_loss",
    "cross_entropy_loss_prime",
    "focal_loss",
    "focal_loss_prime",
    "gce_loss_prime",
    "generalized_cross_entropy",
    "hinge_loss",
    "hinge_loss_prime",
    "js_divergence",
    "js_divergence_prime",
    "kl_divergence",
    "kl_divergence_prime",
    "mse",
    "mse_prime",
    "squared_hinge_loss",
    "squared_hinge_loss_prime",
]

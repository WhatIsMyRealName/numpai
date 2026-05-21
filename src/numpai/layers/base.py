from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

class Layer(ABC):
    """
    Classe de base de toutes les couches. Pas grand intérêt.
    """
    def __init__(self) -> None:
        self.input: np.ndarray | None = None
        self.output: np.ndarray | None = None

    @abstractmethod
    def forward_propagation(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, output_error: np.ndarray, learning_rate: float, epoch: int) -> np.ndarray | None:
        raise NotImplementedError

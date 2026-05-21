from __future__ import annotations

import numpy as np

from .base import Layer


class Embedding(Layer):
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embeddings = np.random.randn(vocab_size, embed_dim)

    def forward_propagation(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = input_data
        self.output = self.embeddings[input_data]
        return self.output

    def backward_propagation(self, output_error: np.ndarray, learning_rate: float, epoch: int) -> None:
        np.add.at(self.embeddings, self.input, -learning_rate * output_error)
        return None

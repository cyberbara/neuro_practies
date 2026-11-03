"""Hebbian learning (rule of Hebb, 1949) — one output neuron per letter.

The idea in one line: "cells that fire together, wire together".

For a picture `x` and the *clamped* output neuron `c` (the class the
picture belongs to), the update is

    ΔW[c] = η · x

i.e. the weight vector of the winning class grows in the direction of the
picture it just saw. After training, each row of `W` is (up to a scale
factor) the average picture of its class — a prototype the network learned
by itself, without any gradient and without a learning-rate schedule.

There is no Oja rule here on purpose. Plain Hebb grows the weights without
bound, so the *decision* step normalises the row to unit length (cosine
similarity) instead of shrinking the weights during training. That keeps
the update rule textbook-Hebb and still gives a usable classifier.

Reading out a class:

    s_c(x) = cos(W[c], x)          similarity with the learned prototype
    P(class | x) = softmax_c(s_c(x) / T)

The same network answers the "find the letter X" question: P(X) is the
softmax weight of the X neuron.
"""
from __future__ import annotations

import numpy as np


def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


class HebbianClassifier:
    """Associative-memory-style classifier trained with the pure Hebb rule.

    Parameters
    ----------
    n_classes : how many output neurons (one per letter).
    n_input   : dimensionality of the input picture (784 for 28x28).
    eta       : learning rate of the Hebbian update.
    temperature : softmax temperature used when turning similarities into
                  probabilities. Lower = sharper decisions.
    """

    def __init__(self, n_classes: int, n_input: int, eta: float = 0.05,
                 temperature: float = 0.05):
        self.n_classes = int(n_classes)
        self.n_input = int(n_input)
        self.eta = float(eta)
        self.temperature = float(temperature)
        self.W = np.zeros((self.n_classes, self.n_input), dtype=np.float64)
        self.classes_seen: list[int] = []

    # ------------------------------------------------------------------ train
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1) -> None:
        """X: (n, n_input) pictures, y: (n,) integer class labels."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        for _ in range(epochs):
            for x, t in zip(X, y):
                # Hebb with the output clamped to the correct class:
                # the winner's weight vector moves towards this picture.
                self.W[t] += self.eta * x
        self.classes_seen = sorted(set(int(v) for v in np.unique(y)))

    # -------------------------------------------------------------- inference
    def similarities(self, X: np.ndarray) -> np.ndarray:
        """Cosine similarity between every picture and every prototype."""
        X = np.asarray(X, dtype=np.float64)
        norms = np.linalg.norm(self.W, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        xn = np.linalg.norm(X, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        return (X @ self.W.T) / (xn * norms.T)

    def probabilities(self, X: np.ndarray) -> np.ndarray:
        """Softmax over the cosine similarities."""
        return _softmax(self.similarities(X) / self.temperature, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.similarities(X), axis=1)

    def p_target(self, X: np.ndarray, target_index: int) -> np.ndarray:
        """Probability that each picture belongs to `target_index`."""
        return self.probabilities(X)[:, target_index]

    # ----------------------------------------------------------- diagnostics
    def prototype(self, class_index: int) -> np.ndarray:
        """The learned prototype of a class as a picture in [0, 1]."""
        w = self.W[class_index]
        w = w - w.min()
        mx = w.max()
        return w / mx if mx > 0 else w

    def weight_norm(self) -> float:
        """Frobenius norm of W — grows without bound under plain Hebb."""
        return float(np.linalg.norm(self.W))

    def pca_axes(self, k: int = 2) -> np.ndarray:
        """First k principal directions of the learned prototypes."""
        W = self.W - self.W.mean(axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(W, full_matrices=False)
        return vt[:k]

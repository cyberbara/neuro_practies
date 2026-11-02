"""Hebb's rule and Oja's rule for associative memory.

Pure NumPy. Trains a weight matrix W from a set of bipolar (+1/-1) or
real-valued patterns. After training, recall y = sign(W @ x) recovers the
stored pattern closest to x (with some noise).

Without normalisation Hebbian weights grow without bound and saturate.
Oja's rule is the classic fix: keep the spectral norm of W bounded.
"""
from __future__ import annotations

import numpy as np


class HebbianNetwork:
    """Tiny associative-memory network.

    Stores n_input-dimensional patterns as columns of a (n_input x n_input)
    weight matrix W. Recall rule is sign(W @ x).

    Two update rules are supported:
      - "hebb":  ΔW = η · x · xᵀ             (weights blow up)
      - "oja":   ΔW = η · (x · xᵀ − W · x²)  (Oja, stable)
    """

    def __init__(self, n_input: int, rule: str = "oja", eta: float = 0.01):
        if rule not in {"hebb", "oja"}:
            raise ValueError(f"unknown rule: {rule!r}")
        self.n_input = n_input
        self.rule = rule
        self.eta = eta
        self.W = np.zeros((n_input, n_input), dtype=np.float64)

    # ------------------------------------------------------------------ train
    def train(self, patterns: np.ndarray, epochs: int = 1) -> None:
        """patterns: shape (n_patterns, n_input). One Hebbian sweep = one epoch."""
        patterns = np.asarray(patterns, dtype=np.float64)
        eta = self.eta
        W = self.W
        rule = self.rule
        for _ in range(epochs):
            for x in patterns:
                # outer product x · xᵀ
                outer = np.outer(x, x)
                if rule == "hebb":
                    W += eta * outer
                else:  # oja
                    y = W @ x                  # recall vector (length n_input)
                    W += eta * (np.outer(x, x) - np.outer(y, x))

    # ------------------------------------------------------------------ recall
    def recall(self, x: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """One synchronous recall step. Returns sign(W @ x)."""
        x = np.asarray(x, dtype=np.float64)
        y = self.W @ x
        return np.where(y >= threshold, 1.0, -1.0)

    def recall_iterative(self, x: np.ndarray, steps: int = 5) -> np.ndarray:
        """Asynchronous recall: each step updates one component of x by sign(W @ x)."""
        x = np.asarray(x, dtype=np.float64).copy()
        for _ in range(steps):
            y = self.W @ x
            # Update the most "undecided" component each step.
            i = int(np.argmin(np.abs(y)))
            x[i] = 1.0 if y[i] >= 0 else -1.0
        return x

    # ----------------------------------------------------------- diagnostics
    def pattern_overlap(self, patterns: np.ndarray) -> np.ndarray:
        """Return mean |cosine| between every stored pattern and its recall.

        1.0 = perfect recall; 0.0 = orthogonal (forgot).
        """
        patterns = np.asarray(patterns, dtype=np.float64)
        recalls = np.array([self.recall(p) for p in patterns])
        # Normalised patterns so cosine = dot product
        p_norm = patterns / np.linalg.norm(patterns, axis=1, keepdims=True)
        r_norm = recalls / np.linalg.norm(recalls, axis=1, keepdims=True)
        return np.abs(np.einsum("ij,ij->i", p_norm, r_norm))

    def noisy_recovery(self, patterns: np.ndarray, noise: float = 0.2,
                       n_trials: int = 200) -> float:
        """Flip `noise` fraction of bits in a random stored pattern and ask
        the network to recover it. Return fraction of bits correctly recovered."""
        patterns = np.asarray(patterns, dtype=np.float64)
        n_pat, n_in = patterns.shape
        rng = np.random.default_rng(0)
        correct = 0
        total = 0
        for _ in range(n_trials):
            pat = patterns[rng.integers(n_pat)]
            mask = rng.random(n_in) < noise
            noisy = pat.copy()
            noisy[mask] *= -1
            recalled = self.recall_iterative(noisy)
            correct += np.sum(recalled == pat)
            total += n_in
        return correct / total

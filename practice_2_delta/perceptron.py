"""Single-layer perceptron trained with the delta rule (Widrow-Hoff / LMS).

For an input x and target t, with output y = σ(W · x + b):
    error   e = t − y
    ΔW     = η · e · σ'(net) · x
    Δb     = η · e · σ'(net)

Implemented with two activation choices:
  - "step"   : pure perceptron, no gradient (decision is sign(net))
  - "linear" : LMS — closed-form for linear regression
  - "sigmoid": differentiable, classic logistic delta rule

Works on:
  - AND / OR / NAND toy problems (linearly separable)
  - XOR (will fail — the headline result of the practice)
  - Iris (subset, two classes) — shows it can do real classification.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(z):
    # Numerically stable sigmoid.
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


@dataclass
class TrainResult:
    weights: np.ndarray
    bias: float
    mse_history: list[float]
    accuracy_history: list[float]
    epochs_run: int


class Perceptron:
    """One-layer perceptron with sigmoid or step activation."""

    def __init__(self, n_input: int, activation: str = "sigmoid",
                 eta: float = 0.1, seed: int = 0):
        if activation not in {"sigmoid", "step", "linear"}:
            raise ValueError(activation)
        self.n_input = n_input
        self.activation = activation
        self.eta = eta
        rng = np.random.default_rng(seed)
        # Small random init — the practice asks us to study its effect.
        self.W = rng.normal(0, 0.1, size=n_input)
        self.b = 0.0

    # ----------------------------------------------------------- forward pass
    def net_input(self, X):
        return X @ self.W + self.b

    def predict_raw(self, X):
        z = self.net_input(X)
        if self.activation == "sigmoid":
            return sigmoid(z)
        if self.activation == "step":
            return np.where(z >= 0, 1.0, 0.0)
        return z  # linear

    def predict_class(self, X):
        raw = self.predict_raw(X)
        return (raw >= 0.5).astype(int)

    # ----------------------------------------------------------- one epoch
    def _step(self, X, y):
        eta = self.eta
        W = self.W
        b = self.b
        for x, t in zip(X, y):
            z = float(x @ W + b)
            if self.activation == "sigmoid":
                yhat = 1.0 / (1.0 + np.exp(-z))
                grad = (t - yhat) * yhat * (1.0 - yhat)
            elif self.activation == "step":
                yhat = 1.0 if z >= 0 else 0.0
                grad = (t - yhat)
            else:
                yhat = z
                grad = (t - yhat)
            W += eta * grad * x
            b += eta * grad

    # --------------------------------------------------------------- training
    def fit(self, X, y, epochs: int = 200, log_every: int = 1):
        mse_hist = []
        acc_hist = []
        for ep in range(1, epochs + 1):
            self._step(X, y)
            if ep % log_every == 0 or ep == epochs:
                preds = self.predict_raw(X)
                mse = float(((preds - y) ** 2).mean())
                mse_hist.append(mse)
                acc_hist.append(float((self.predict_class(X) == y).mean()))
        return TrainResult(self.W.copy(), self.b, mse_hist, acc_hist, epochs)


# ---------------------------------------------------------------- datasets
def toy(name: str):
    """Return (X, y) for AND / OR / NAND / XOR."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    if name == "AND":
        return X, np.array([0, 0, 0, 1])
    if name == "OR":
        return X, np.array([0, 1, 1, 1])
    if name == "NAND":
        return X, np.array([1, 1, 1, 0])
    if name == "XOR":
        return X, np.array([0, 1, 1, 0])
    raise ValueError(name)

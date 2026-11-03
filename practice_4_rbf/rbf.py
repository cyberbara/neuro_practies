"""Radial-basis-function network.

Architecture:
  - Hidden layer: n_centres Gaussian RBF units, φ_i(x) = exp(−‖x − c_i‖² / (2σ_i²))
  - Output layer: linear combination, y = W · φ(x) + b (or pseudoinverse)

Two training paths are supported for the output weights:
  - "lsmr"     : solve with linear least-squares (scipy.sparse.linalg.lsmr
                 or numpy.linalg.lstsq). Closed-form, no learning rate needed.
  - "gradient" : plain gradient descent on MSE.

Centres are picked by K-means; per-centre widths σ_i are set to the median
distance to the nearest neighbour (a common heuristic).

The output layer supports several classes at once: pass `y` as an (n, c)
matrix of targets and the network learns c linear read-outs sharing the same
hidden layer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class RBFConfig:
    n_centres: int = 16
    output_solver: str = "lstsq"   # "lstsq" | "gradient"
    output_lr: float = 0.05
    output_epochs: int = 200
    kmeans_n_init: int = 3


def _gaussian(x, centres, widths):
    # x: (n, d), centres: (k, d), widths: (k,)
    # returns (n, k)
    diff = x[:, None, :] - centres[None, :, :]
    sq = (diff ** 2).sum(axis=2)
    return np.exp(-sq / (2.0 * widths[None, :] ** 2))


def _fit_centres(X, n_centres, random_state=0, n_init=3):
    km = KMeans(n_clusters=n_centres, n_init=n_init, random_state=random_state)
    km.fit(X)
    centres = km.cluster_centers_
    # Per-centre width = median distance to the k nearest neighbour points.
    diff = X[:, None, :] - centres[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))   # (n, k)
    widths = np.median(dists, axis=0) + 1e-9
    return centres, widths


class RBFNet:
    def __init__(self, cfg: RBFConfig):
        self.cfg = cfg
        self.centres = None
        self.widths = None
        self.W = None
        self.b = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.centres, self.widths = _fit_centres(
            X, self.cfg.n_centres, n_init=self.cfg.kmeans_n_init)
        Phi = _gaussian(X, self.centres, self.widths)
        if self.cfg.output_solver == "lstsq":
            # y can be (n,) or (n, c_out); use lstsq on the augmented matrix.
            ones = np.ones((Phi.shape[0], 1))
            A = np.hstack([Phi, ones])
            sol, *_ = np.linalg.lstsq(A, y if y.ndim > 1 else y[:, None], rcond=None)
            self.W, self.b = sol[:-1], sol[-1]
        else:
            self.W = np.zeros((self.centres.shape[0],) if y.ndim == 1
                              else (self.centres.shape[0], y.shape[1]))
            self.b = 0.0 if y.ndim == 1 else np.zeros(y.shape[1])
            for _ in range(self.cfg.output_epochs):
                pred = Phi @ self.W + self.b
                grad = Phi.T @ (pred - y) / X.shape[0]
                self.W -= self.cfg.output_lr * grad
                self.b -= self.cfg.output_lr * (pred - y).mean(axis=0)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        Phi = _gaussian(X, self.centres, self.widths)
        y = Phi @ self.W
        if np.ndim(self.b) == 0:
            return y + self.b
        return y + self.b

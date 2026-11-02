"""Self-organising map (Kohonen, 1982).

A 2-D grid of n_rows x n_cols neurons, each holding a d-dimensional weight
vector. For every input x:
  1. Find the best-matching unit (BMU): argmin_i ‖w_i − x‖.
  2. Update the BMU and its neighbours: w_i ← w_i + η(t) · h(i, BMU, t) · (x − w_i).

Both η and the neighbourhood radius σ shrink over training epochs following
simple exponentials; this is the classic "conscience" learning schedule.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SOMConfig:
    n_rows: int = 8
    n_cols: int = 8
    n_features: int | None = None
    n_epochs: int = 50
    sigma_init: float = 3.0
    sigma_end: float = 0.5
    lr_init: float = 0.5
    lr_end: float = 0.01
    seed: int = 0


def _gaussian_neighbourhood(d2, sigma):
    return np.exp(-d2 / (2.0 * sigma ** 2))


class SOM:
    def __init__(self, cfg: SOMConfig):
        self.cfg = cfg
        self.W = None            # (n_rows * n_cols, n_features)
        self.grid_coords = None  # (n_rows * n_cols, 2)

    def _build_grid(self, n_features):
        rng = np.random.default_rng(self.cfg.seed)
        self.W = rng.normal(0, 0.5, size=(self.cfg.n_rows * self.cfg.n_cols, n_features))
        rs, cs = np.meshgrid(np.arange(self.cfg.n_rows),
                              np.arange(self.cfg.n_cols), indexing="ij")
        self.grid_coords = np.stack([rs.ravel(), cs.ravel()], axis=1).astype(float)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        if self.W is None:
            self._build_grid(d)
        cfg = self.cfg
        n_steps = cfg.n_epochs * n
        rng = np.random.default_rng(cfg.seed)
        for step in range(n_steps):
            t = step / n_steps          # in [0, 1)
            sigma = cfg.sigma_init * (cfg.sigma_end / cfg.sigma_init) ** t
            lr = cfg.lr_init * (cfg.lr_end / cfg.lr_init) ** t
            x = X[rng.integers(n)]
            # BMU
            d2 = ((self.W - x) ** 2).sum(axis=1)
            bmu = int(np.argmin(d2))
            # Neighbourhood
            grid_d2 = ((self.grid_coords - self.grid_coords[bmu]) ** 2).sum(axis=1)
            h = _gaussian_neighbourhood(grid_d2, sigma)[:, None]
            self.W += lr * h * (x - self.W)
        return self

    # ---------------------------------------------------------- diagnostics
    def u_matrix(self):
        """Average distance between each neuron and its 4-neighbours."""
        rows, cols = self.cfg.n_rows, self.cfg.n_cols
        u = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                d = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        d.append(np.linalg.norm(self.W[r * cols + c] - self.W[nr * cols + nc]))
                u[r, c] = np.mean(d) if d else 0.0
        return u

    def map_labels(self, X, y):
        """Assign each neuron the most common class of the points it BMU-matches."""
        X = np.asarray(X)
        y = np.asarray(y)
        n_neurons = self.cfg.n_rows * self.cfg.n_cols
        votes = [[] for _ in range(n_neurons)]
        for xi, yi in zip(X, y):
            d2 = ((self.W - xi) ** 2).sum(axis=1)
            votes[int(np.argmin(d2))].append(int(yi))
        labels = np.array([np.bincount(v).argmax() if v else -1 for v in votes])
        return labels.reshape(self.cfg.n_rows, self.cfg.n_cols)

"""Self-organising map (Kohonen, 1982).

A 2-D grid of n_rows x n_cols neurons, each holding a d-dimensional weight
vector. For every input x:
  1. Find the best-matching unit (BMU): argmin_i ‖w_i − x‖.
  2. Update the BMU and its neighbours: w_i ← w_i + η(t) · h(i, BMU, t) · (x − w_i).

Both η and the neighbourhood radius σ shrink over training epochs following
simple exponentials; this is the classic "conscience" learning schedule.

Training comes in two flavours:

* ``update_mode="online"`` — the textbook loop: one random picture per step.
* ``update_mode="batch"``  — one epoch at a time. Every picture of the
  epoch contributes at once, and each neuron receives the summed pull of
  all pictures that won it (or one of its neighbours). Mathematically the
  same rule, but it runs on BLAS instead of a Python loop, which matters a
  lot when the pictures are 784-dimensional.
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
    update_mode: str = "batch"      # "batch" | "online"


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

    # ------------------------------------------------------------- training
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        if self.W is None:
            self._build_grid(d)
        cfg = self.cfg
        if cfg.update_mode == "online":
            self._fit_online(X)
        else:
            self._fit_batch(X)
        return self

    def _fit_online(self, X):
        """Textbook one-picture-per-step SOM."""
        n = X.shape[0]
        cfg = self.cfg
        n_steps = cfg.n_epochs * n
        rng = np.random.default_rng(cfg.seed)
        for step in range(n_steps):
            t = step / max(n_steps - 1, 1)
            sigma = cfg.sigma_init * (cfg.sigma_end / cfg.sigma_init) ** t
            lr = cfg.lr_init * (cfg.lr_end / cfg.lr_init) ** t
            x = X[rng.integers(n)]
            # ||w_i - x||² = ||w_i||² - 2 w_i·x + ||x||²
            d2 = (self.W * self.W).sum(axis=1) - 2.0 * (self.W @ x)
            bmu = int(np.argmin(d2))
            grid_d2 = ((self.grid_coords - self.grid_coords[bmu]) ** 2).sum(axis=1)
            h = _gaussian_neighbourhood(grid_d2, sigma)[:, None]
            self.W += lr * h * (x - self.W)

    def _fit_batch(self, X):
        """One epoch at a time — same rule, vectorised over the whole epoch.

        For every neuron j we accumulate
            num_j = Σ_i h(j, c_i) · x_i        (sum of pulls)
            den_j = Σ_i h(j, c_i)              (sum of weights)
        over the pictures i whose BMU c_i is j or a neighbour of j. The
        neuron then moves towards the *weighted average* of those pictures:

            w_j ← w_j + η · (num_j / den_j − w_j)

        which is the batch form of the usual incremental update
        w_j ← w_j + η·h·(x − w_j) — normalised by the total pull, so the
        step size does not grow with the number of pictures in the epoch.
        """
        n = X.shape[0]
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        order = rng.permutation(n)
        for ep in range(cfg.n_epochs):
            t = ep / max(cfg.n_epochs - 1, 1)
            sigma = cfg.sigma_init * (cfg.sigma_end / cfg.sigma_init) ** t
            lr = cfg.lr_init * (cfg.lr_end / cfg.lr_init) ** t
            Xs = X[order]

            # BMU of every picture: (n, k) distance matrix via BLAS.
            d2 = ((Xs * Xs).sum(axis=1)[:, None]
                  - 2.0 * Xs @ self.W.T
                  + (self.W * self.W).sum(axis=1)[None, :])
            bmus = np.argmin(d2, axis=1)

            # Neighbourhood weights between all neuron pairs.
            gd2 = ((self.grid_coords[:, None, :]
                    - self.grid_coords[None, :, :]) ** 2).sum(axis=2)
            H = _gaussian_neighbourhood(gd2, sigma)      # H[j, c]

            # Wsum[j, i] = h(j, c_i)  — the pull every picture exerts on j.
            onehot = np.zeros((Xs.shape[0], self.W.shape[0]))
            onehot[np.arange(Xs.shape[0]), bmus] = 1.0
            Wsum = (onehot @ H.T).T                        # (k, n)
            num = Wsum @ Xs                               # (k, d)
            den = Wsum.sum(axis=1)                        # (k,)

            live = den > 1e-12
            target = np.zeros_like(self.W)
            target[live] = num[live] / den[live, None]
            self.W[live] += lr * (target[live] - self.W[live])

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

    # ----------------------------------------------------------- inference
    def _distances(self, X):
        X = np.asarray(X, dtype=np.float64)
        return ((X * X).sum(axis=1)[:, None]
                - 2.0 * X @ self.W.T
                + (self.W * self.W).sum(axis=1)[None, :])

    def bmu(self, X):
        """Index of the best-matching unit for every sample."""
        return np.argmin(self._distances(X), axis=1)

    def classify(self, X, neuron_labels):
        """Predict a class for each sample using a pre-labelled map.

        `neuron_labels` is the flat (n_neurons,) array produced by
        `map_labels(...).ravel()`. Each sample takes the label of its BMU.
        """
        labels = np.asarray(neuron_labels).ravel()
        return labels[self.bmu(X)]

    def target_score(self, X, neuron_labels, target_index, k: int = 5):
        """Soft "is this the target class?" score in [0, 1].

        Takes the k nearest neurons of every sample, weights them by a
        Gaussian of their distance and returns the weighted share of
        neurons labelled as the target class.
        """
        labels = np.asarray(neuron_labels).ravel()
        d = np.sqrt(np.maximum(self._distances(X), 0.0))
        order = np.argsort(d, axis=1)[:, :k]
        dk = np.take_along_axis(d, order, axis=1)
        sigma = np.median(dk) + 1e-9
        w = np.exp(-(dk ** 2) / (2 * sigma ** 2))
        lab = labels[order]
        num = (w * (lab == target_index)).sum(axis=1)
        den = w.sum(axis=1) + 1e-12
        return num / den

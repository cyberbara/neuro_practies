"""Common dataset loaders used across practices.

Provides a single place to normalise and cache the small datasets
(Iris, two-moons) so each practice can pull what it needs.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris, make_moons
from sklearn.preprocessing import StandardScaler


def iris():
    """Standardised Iris features + integer labels."""
    d = load_iris()
    X = StandardScaler().fit_transform(d.data)
    y = d.target.astype(np.int64)
    return X, y, d.target_names


def two_moons(n: int = 400, noise: float = 0.18, seed: int = 0):
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return X.astype(np.float64), y.astype(np.int64)


def xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([0, 1, 1, 0], dtype=np.int64)
    return X, y

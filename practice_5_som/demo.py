"""Demo for practice 5: Kohonen SOM on Iris.

Trains an 8x8 SOM on the four Iris features, then:
  1. Plots the U-matrix (inter-neuron distances).
  2. Colours the map by the most common class of points assigned to each
     neuron — visual check that topology is preserved.
  3. Compares two map sizes (5x5 vs 10x10) for how well they cluster.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from practice_5_som.som import SOM, SOMConfig

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)


def train_and_evaluate(name, rows, cols, X, y, n_epochs=50):
    cfg = SOMConfig(n_rows=rows, n_cols=cols, n_epochs=n_epochs, seed=0)
    t0 = time.perf_counter()
    som = SOM(cfg).fit(X)
    t = time.perf_counter() - t0
    labels = som.map_labels(X, y)
    # Quantisation error: mean distance from each training point to its BMU.
    d2 = np.min(np.linalg.norm(som.W[None, :, :] - X[:, None, :], axis=2), axis=1)
    qe = float(d2.mean())
    return som, labels, qe, t


def main():
    iris = load_iris()
    X = StandardScaler().fit_transform(iris.data)
    y = iris.target
    target_names = iris.target_names

    # 8x8 reference SOM
    som, labels, qe, t = train_and_evaluate("8x8", 8, 8, X, y, n_epochs=80)
    print(f"8x8  | n_neurons={som.W.shape[0]} | quantisation_error={qe:.3f} | time={t:.2f}s")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    u = som.u_matrix()
    axes[0].imshow(u, cmap="bone_r")
    axes[0].set_title("U-matrix (lighter = farther from neighbours)")
    axes[0].axis("off")

    im = axes[1].imshow(labels, cmap="viridis", vmin=-0.5, vmax=2.5)
    axes[1].set_title("Most-frequent class per neuron")
    axes[1].axis("off")
    cbar = fig.colorbar(im, ax=axes[1], ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(target_names)
    fig.tight_layout()
    fig.savefig(OUT / "p5_som_8x8.png", dpi=120)
    plt.close(fig)

    # 5x5 vs 10x10 comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (r, c) in zip(axes, [(5, 5), (10, 10)]):
        som, labels, qe, t = train_and_evaluate(f"{r}x{c}", r, c, X, y, n_epochs=80)
        ax.imshow(labels, cmap="viridis", vmin=-0.5, vmax=2.5)
        ax.set_title(f"{r}x{c} SOM | qe={qe:.3f} | time={t:.2f}s")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "p5_som_sizes.png", dpi=120)
    plt.close(fig)

    print("figures written to:", OUT)


if __name__ == "__main__":
    main()

"""Demo for practice 5: Kohonen SOM on handwritten digits.

Trains a SOM on the sklearn digits dataset (8x8 images, 10 classes, 64
features) — a real image dataset rather than a 4-feature toy.

Shows:
  1. U-matrix (inter-neuron distances).
  2. Class map — the most common class of the points assigned to each
     neuron. Clean blobs = topology preserved.
  3. Quantisation error vs map size.
  4. Side-by-side with a PCA projection for reference.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from practice_5_som.som import SOM, SOMConfig

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

DIGIT_NAMES = [str(i) for i in range(10)]


def load_data():
    d = load_digits()
    X = StandardScaler().fit_transform(d.data)
    return X, d.target.astype(np.int64)


def train_and_evaluate(rows, cols, X, y, n_epochs=80):
    cfg = SOMConfig(n_rows=rows, n_cols=cols, n_epochs=n_epochs, seed=0)
    t0 = time.perf_counter()
    som = SOM(cfg).fit(X)
    t = time.perf_counter() - t0
    labels = som.map_labels(X, y)
    d2 = np.min(np.linalg.norm(som.W[None, :, :] - X[:, None, :], axis=2), axis=1)
    qe = float(d2.mean())
    return som, labels, qe, t


def main():
    X, y = load_data()
    print(f"digits: {X.shape[0]} samples x {X.shape[1]} features")

    # --- Reference 12x12 SOM -------------------------------------------------
    som, labels, qe, t = train_and_evaluate(12, 12, X, y, n_epochs=100)
    print(f"12x12 | neurons={som.W.shape[0]} | qe={qe:.3f} | time={t:.1f}s")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(som.u_matrix(), cmap="bone_r")
    axes[0].set_title("U-matrix (lighter = farther from neighbours)")
    axes[0].axis("off")

    im = axes[1].imshow(labels, cmap="tab10", vmin=-0.5, vmax=9.5)
    axes[1].set_title("Most-frequent class per neuron")
    axes[1].axis("off")
    cbar = fig.colorbar(im, ax=axes[1], ticks=list(range(10)))
    cbar.ax.set_yticklabels(DIGIT_NAMES)
    fig.suptitle("Kohonen SOM on handwritten digits (12×12)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "p5_som_digits.png", dpi=120)
    plt.close(fig)

    # --- Map size sweep ------------------------------------------------------
    sizes = [(6, 6), (10, 10), (16, 16), (20, 20)]
    qes, times = [], []
    for r, c in sizes:
        som, labels, qe, t = train_and_evaluate(r, c, X, y, n_epochs=100)
        qes.append(qe)
        times.append(t)
        print(f"{r:2d}x{c:<2d} | neurons={som.W.shape[0]:4d} | qe={qe:.3f} | {t:.1f}s")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(sizes))
    axes[0].bar(x, qes, color="C0")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{r}×{c}" for r, c in sizes])
    axes[0].set_ylabel("quantisation error")
    axes[0].set_title("Quantisation error vs map size")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(x, times, color="C1")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{r}×{c}" for r, c in sizes])
    axes[1].set_ylabel("training time (s)")
    axes[1].set_title("Training time vs map size")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "p5_som_sizes.png", dpi=120)
    plt.close(fig)

    # --- SOM vs PCA reference ------------------------------------------------
    som, labels, qe, t = train_and_evaluate(16, 16, X, y, n_epochs=100)
    pca = PCA(n_components=2).fit(X)
    Z = pca.transform(X)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for c in range(10):
        m = y == c
        axes[0].scatter(Z[m, 0], Z[m, 1], s=6, label=DIGIT_NAMES[c],
                        alpha=0.6, cmap="tab10")
    axes[0].set_title("PCA (2 components)")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].legend(markerscale=2, fontsize=7, ncol=5)

    im = axes[1].imshow(labels, cmap="tab10", vmin=-0.5, vmax=9.5)
    axes[1].set_title("SOM 16×16 class map")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "p5_som_vs_pca.png", dpi=120)
    plt.close(fig)

    print("figures written to:", OUT)


if __name__ == "__main__":
    main()

"""Demo for practice 5: Kohonen SOM on the letters dataset.

The SOM trains **without labels** on the pictures of 26 letters, then the
map is *labelled* afterwards (majority class of the points each neuron
won). After that the map can answer "which letter is this?" and, in
particular, "is this the letter X?".

Shows:
  1. U-matrix (inter-neuron distances).
  2. Class map — the most common class of the points assigned to each
     neuron. Clean blobs = topology preserved.
  3. Map-size sweep: quantisation error and "find X" quality.
  4. Side-by-side with a PCA projection for reference.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from common.letters import build_letters
from common.metrics import binary_scores
from practice_5_som.som import SOM, SOMConfig

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)


def train_and_evaluate(rows, cols, X, y, Xte, yte, split, n_epochs=200):
    cfg = SOMConfig(n_rows=rows, n_cols=cols, n_epochs=n_epochs, seed=0,
                    update_mode="batch")
    t0 = time.perf_counter()
    som = SOM(cfg).fit(X)
    t = time.perf_counter() - t0
    labels = som.map_labels(X, y)
    d2 = np.min(np.linalg.norm(som.W[None, :, :] - X[:, None, :], axis=2), axis=1)
    qe = float(d2.mean())
    # classify test samples with the labelled map
    flat = labels.ravel()
    pred = som.classify(Xte, flat)
    acc = float((pred == yte).mean())
    # soft "is it X" score
    p_x = som.target_score(Xte, flat, split.target_index, k=5)
    dec = (pred == split.target_index).astype(float)
    m = binary_scores(split.binary_targets("test"), p_x, decision=dec)
    return som, labels, qe, t, acc, m


def main():
    print("rendering letters dataset…")
    split = build_letters(seed=0)
    X, y = split.X_train, split.y_train
    Xte, yte = split.X_test, split.y_test
    print(f"letters: {X.shape[0]} samples x {X.shape[1]} features")

    # --- Reference 16x16 SOM -------------------------------------------------
    som, labels, qe, t, acc, m = train_and_evaluate(
        16, 16, X, y, Xte, yte, split, n_epochs=200)
    print(f"16x16 | neurons={som.W.shape[0]} | qe={qe:.3f} | "
          f"test acc={acc:.3f} | bal_acc(X)={m['balanced_accuracy']:.3f} | "
          f"AUC={m['auc']:.3f} | {t:.1f}s")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].imshow(som.u_matrix(), cmap="bone_r")
    axes[0].set_title("U-matrix (lighter = farther from neighbours)")
    axes[0].axis("off")

    im = axes[1].imshow(labels, cmap="tab20", vmin=-0.5,
                        vmax=split.n_classes - 0.5)
    axes[1].set_title("Most-frequent class per neuron")
    axes[1].axis("off")
    cbar = fig.colorbar(im, ax=axes[1], ticks=range(split.n_classes))
    cbar.ax.set_yticklabels(list(split.letters))
    fig.suptitle("Kohonen SOM on the letters dataset (16×16)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "p5_som_letters.png", dpi=120)
    plt.close(fig)

    # --- Map size sweep ------------------------------------------------------
    sizes = [(6, 6), (10, 10), (16, 16), (20, 20)]
    qes, times, bals, accs = [], [], [], []
    for r, c in sizes:
        som, labels, qe, t, acc, m = train_and_evaluate(r, c, X, y, Xte, yte,
                                                        split, n_epochs=200)
        qes.append(qe)
        times.append(t)
        bals.append(m["balanced_accuracy"])
        accs.append(acc)
        print(f"{r:2d}x{c:<2d} | neurons={som.W.shape[0]:4d} | qe={qe:.3f} | "
              f"test acc={acc:.3f} | bal_acc(X)={m['balanced_accuracy']:.3f} | "
              f"{t:.1f}s")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    x = np.arange(len(sizes))
    labels_x = [f"{r}×{c}" for r, c in sizes]
    axes[0].bar(x, qes, color="C0")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels_x)
    axes[0].set_ylabel("quantisation error")
    axes[0].set_title("Quantisation error vs map size")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(x, accs, color="C2")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels_x)
    axes[1].set_ylabel("accuracy (26 classes)")
    axes[1].set_title("Labelled-map accuracy vs map size")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", alpha=0.3)
    axes[2].bar(x, times, color="C1")
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels_x)
    axes[2].set_ylabel("training time (s)")
    axes[2].set_title("Training time vs map size")
    axes[2].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "p5_som_sizes.png", dpi=120)
    plt.close(fig)

    # --- SOM vs PCA reference ------------------------------------------------
    som, labels, qe, t, acc, m = train_and_evaluate(16, 16, X, y, Xte, yte,
                                                    split, n_epochs=200)
    pca = PCA(n_components=2).fit(X)
    Z = pca.transform(X)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for c in range(split.n_classes):
        mask = y == c
        axes[0].scatter(Z[mask, 0], Z[mask, 1], s=5, label=split.letters[c],
                        alpha=0.5)
    axes[0].set_title("PCA (2 components)")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].legend(markerscale=2, fontsize=7, ncol=7)

    im = axes[1].imshow(labels, cmap="tab20", vmin=-0.5,
                        vmax=split.n_classes - 0.5)
    axes[1].set_title("SOM 16×16 class map")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "p5_som_vs_pca.png", dpi=120)
    plt.close(fig)

    # --- Where does X live on the map? --------------------------------------
    flat = labels.ravel()
    x_neurons = np.where(flat == split.target_index)[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(labels == split.target_index, cmap="gray_r", vmin=0, vmax=1)
    axes[0].set_title(f"neurons that ended up on letter {split.target}")
    axes[0].axis("off")
    # pictures of the neurons of the X cluster
    n_show = min(6, len(x_neurons))
    for k in range(n_show):
        ax = axes[1]
        ax.axis("off")
        break
    axes[1].axis("off")
    for k, idx in enumerate(x_neurons[:6]):
        inset = axes[1].inset_axes([k / 6, 0.05, 0.15, 0.55])
        inset.imshow(som.W[idx].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
        inset.set_xticks([]); inset.set_yticks([])
        inset.set_title(split.target, fontsize=8)
    fig.suptitle(f"Cluster of the letter {split.target} on the SOM", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "p5_x_cluster.png", dpi=120)
    plt.close(fig)

    print("figures written to:", OUT)


if __name__ == "__main__":
    main()

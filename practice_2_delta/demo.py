"""Demo for practice 2: delta-rule perceptron.

Two experiments:
  1. AND / OR / NAND / XOR — classical lesson. XOR never converges, because
     one neuron draws one straight line.
  2. The real task: **find the letter X** on the letters dataset. One
     sigmoid neuron, 784 inputs, trained with Widrow-Hoff.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.letters import SIZE, build_letters
from common.metrics import binary_scores
from practice_2_delta.perceptron import Perceptron, toy

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)


def decision_grid(model, X, padding=0.5, n=200):
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n),
                          np.linspace(y_min, y_max, n))
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, model.predict_class(grid).reshape(xx.shape)


def run_toy(name, eta=0.5, epochs=200):
    X, y = toy(name)
    model = Perceptron(n_input=2, activation="sigmoid", eta=eta, seed=0)
    t0 = time.perf_counter()
    res = model.fit(X, y, epochs=epochs, log_every=1)
    dt = time.perf_counter() - t0
    return X, y, model, res, dt


def toy_panel():
    """AND / OR / NAND / XOR in one figure."""
    problems = ["AND", "OR", "NAND", "XOR"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for col, name in enumerate(problems):
        X, y, model, res, _ = run_toy(name)
        ax = axes[0, col]
        ax.plot(res.mse_history, label="MSE")
        ax.plot(res.accuracy_history, label="accuracy")
        ax.set_title(f"{name} (final acc={res.accuracy_history[-1]:.2f})")
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax = axes[1, col]
        xx, yy, Z = decision_grid(model, X)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
        for cls in [0, 1]:
            mask = y == cls
            ax.scatter(X[mask, 0], X[mask, 1], label=f"class {cls}", s=80,
                       edgecolors="k")
        ax.set_title(f"{name} boundary (η={model.eta})")
        ax.set_xlabel("x1"); ax.set_ylabel("x2")
    fig.suptitle("Toy problems — the XOR barrier", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "p2_toy.png", dpi=120)
    plt.close(fig)


def letters_panel(split):
    """The real task: one neuron that finds the letter X."""
    Xtr, ytr = split.X_train, split.binary_targets("train")
    Xte, yte = split.X_test, split.binary_targets("test")
    etas = [0.05, 0.1, 0.5, 1.0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # Left: learning curves for several eta (train / test MSE)
    for eta in etas:
        net = Perceptron(n_input=Xtr.shape[1], activation="sigmoid",
                         eta=eta, seed=0)
        res = net.fit(Xtr, ytr, epochs=60, log_every=1)
        axes[0].plot(res.mse_history, label=f"η={eta}")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train MSE")
    axes[0].set_title("Delta rule on letters — convergence vs η")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Middle: score distribution for X vs the rest
    net = Perceptron(n_input=Xtr.shape[1], activation="sigmoid", eta=0.5,
                     seed=0)
    net.fit(Xtr, ytr, epochs=60)
    p = net.predict_proba(Xte)
    axes[1].hist(p[yte == 0], bins=40, alpha=0.7, label="not X", color="tab:blue")
    axes[1].hist(p[yte == 1], bins=40, alpha=0.8, label="X", color="tab:red")
    axes[1].axvline(0.5, color="k", ls="--", lw=1)
    axes[1].set_xlabel("neuron output")
    axes[1].set_title("One neuron separates X from the rest")
    axes[1].legend(fontsize=9)

    # Right: final metrics per eta
    names, bal, auc = [], [], []
    for eta in etas:
        n2 = Perceptron(n_input=Xtr.shape[1], activation="sigmoid", eta=eta,
                        seed=0)
        n2.fit(Xtr, ytr, epochs=60)
        m = binary_scores(yte, n2.predict_proba(Xte))
        names.append(f"η={eta}")
        bal.append(m["balanced_accuracy"])
        auc.append(m["auc"])
    x = np.arange(len(etas))
    axes[2].bar(x - 0.2, bal, 0.4, label="balanced accuracy", color="C0")
    axes[2].bar(x + 0.2, auc, 0.4, label="ROC-AUC", color="C2")
    axes[2].set_xticks(x); axes[2].set_xticklabels(names)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("score")
    axes[2].set_title("Test scores vs learning rate")
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", alpha=0.3)
    fig.suptitle("Practice 2 — one neuron hunting the letter X", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "p2_letters.png", dpi=120)
    plt.close(fig)


def weight_map(split):
    """What did the single neuron learn to look at?"""
    Xtr, ytr = split.X_train, split.binary_targets("train")
    net = Perceptron(n_input=Xtr.shape[1], activation="sigmoid", eta=0.5,
                     seed=0)
    net.fit(Xtr, ytr, epochs=60)

    # average the positive and negative weights into pictures
    w_pos = net.W.reshape(-1).copy()
    pos_mask = Xtr.sum(axis=1) > 0        # ink present
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))

    w = net.W.reshape(SIZE, SIZE)
    vmax = np.abs(w).max()
    axes[0].imshow(w, cmap="gray_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("weights W (784)")
    axes[0].axis("off")

    # mean picture of X vs mean picture of the rest
    Xtr2 = Xtr.reshape(-1, SIZE, SIZE)
    is_x = ytr == 1
    axes[1].imshow(Xtr2[is_x].mean(axis=0), cmap="gray_r", vmin=0, vmax=1)
    axes[1].set_title("mean picture of X")
    axes[1].axis("off")
    axes[2].imshow(Xtr2[~is_x].mean(axis=0), cmap="gray_r", vmin=0, vmax=1)
    axes[2].set_title("mean picture of the rest")
    axes[2].axis("off")
    fig.suptitle("What the single neuron weights look like", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "p2_weights.png", dpi=120)
    plt.close(fig)
    del w_pos, pos_mask


def init_sweep():
    """Variability of AND-convergence across 20 seeds."""
    final_accs = []
    for seed in range(20):
        m = Perceptron(n_input=2, activation="sigmoid", eta=0.1, seed=seed)
        r = m.fit(*toy("AND"), epochs=100)
        final_accs.append(r.accuracy_history[-1])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(range(20), final_accs)
    ax.set_xlabel("seed"); ax.set_ylabel("final accuracy")
    ax.set_title("AND — final accuracy across 20 random initialisations")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "p2_init_sweep.png", dpi=120)
    plt.close(fig)
    print(f"AND accuracy: {np.mean(final_accs):.3f} ± {np.std(final_accs):.3f}")


def main():
    print("rendering letters dataset…")
    split = build_letters(seed=0)
    toy_panel()
    letters_panel(split)
    weight_map(split)
    init_sweep()
    print("figures written to:", OUT)


if __name__ == "__main__":
    main()

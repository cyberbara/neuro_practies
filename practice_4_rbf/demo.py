"""Demo for practice 4: RBF network.

Four experiments:
  1. Approximate a 1-D sine wave; sweep the number of centres.
  2. Letters dataset (28×28, 26 classes) — the real task. One RBF network
     with 26 linear read-outs sharing a single hidden layer of Gaussian
     centres. Sweep the number of centres.
  3. Closed-form (least squares) vs gradient-descent output layer —
     the headline "RBF trains fast" result.
  4. Two-moons decision surface with the K-means centres overlaid.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

from common.letters import build_letters
from common.metrics import binary_scores
from practice_4_rbf.rbf import RBFConfig, RBFNet

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)


def sine_panel():
    """1-D function approximation."""
    rng = np.random.default_rng(0)
    X = np.linspace(-np.pi, np.pi, 200)[:, None]
    y = np.sin(2 * X[:, 0]) + 0.05 * rng.standard_normal(X.shape[0])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    Xplot = np.linspace(-np.pi, np.pi, 400)[:, None]
    true = np.sin(2 * Xplot[:, 0])
    for ax, k in zip(axes, [4, 12, 32]):
        net = RBFNet(RBFConfig(n_centres=k, output_solver="lstsq")).fit(X, y)
        pred = net.predict(Xplot)
        ax.scatter(X[:, 0], y, s=10, alpha=0.5, label="data")
        ax.plot(Xplot[:, 0], true, "k--", label="true")
        ax.plot(Xplot[:, 0], pred, "r-", label=f"k={k}")
        ax.set_title(f"RBF approx, centres={k}")
        ax.legend(fontsize=8)
    fig.suptitle("Function approximation — resolution scales with centres",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "p4_approx.png", dpi=120)
    plt.close(fig)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def letters_panel(split):
    """Letters classification — the real task."""
    Xtr, ytr = split.X_train, split.y_train
    Xte, yte = split.X_test, split.y_test
    n_classes = split.n_classes
    Ytr = np.eye(n_classes)[ytr]
    ks = [10, 30, 60, 120]
    train_accs, test_accs, times = [], [], []
    bal_accs, aucs = [], []

    for k in ks:
        t0 = time.perf_counter()
        net = RBFNet(RBFConfig(n_centres=k, output_solver="lstsq")).fit(Xtr, Ytr)
        t = time.perf_counter() - t0
        Pte = net.predict(Xte)
        Ptr = net.predict(Xtr)
        test_accs.append(float((Pte.argmax(1) == yte).mean()))
        train_accs.append(float((Ptr.argmax(1) == ytr).mean()))
        times.append(t)
        # "find the letter X" score: softmax over the 26 read-out values.
        probs = _softmax(Pte / 0.1)
        p_x = probs[:, split.target_index]
        dec = (Pte.argmax(1) == split.target_index).astype(float)
        m = binary_scores(split.binary_targets("test"), p_x, decision=dec)
        bal_accs.append(m["balanced_accuracy"])
        aucs.append(m["auc"])
        print(f"centres={k:4d} | train={train_accs[-1]:.3f} | "
              f"test={test_accs[-1]:.3f} | bal_acc(X)={bal_accs[-1]:.3f} | "
              f"AUC={m['auc']:.3f} | {t*1000:.0f} ms")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    x = np.arange(len(ks))
    w = 0.4
    axes[0].bar(x - w/2, train_accs, w, label="train", color="C0")
    axes[0].bar(x + w/2, test_accs,  w, label="test",  color="C3")
    axes[0].set_xticks(x); axes[0].set_xticklabels([str(k) for k in ks])
    axes[0].set_xlabel("number of RBF centres")
    axes[0].set_ylabel("accuracy (26 classes)")
    axes[0].set_title("Letters — accuracy vs centres")
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0.2, 1.0)

    axes[1].bar(x, bal_accs, color="C2", label="balanced acc (find X)")
    axes[1].plot(x, aucs, "o--", color="C3", label="ROC-AUC")
    axes[1].set_xticks(x); axes[1].set_xticklabels([str(k) for k in ks])
    axes[1].set_xlabel("number of RBF centres")
    axes[1].set_ylabel("score")
    axes[1].set_title("Finding the letter X")
    axes[1].set_ylim(0.2, 1.05)
    axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(x, [t * 1000 for t in times], color="C1")
    axes[2].set_xticks(x); axes[2].set_xticklabels([str(k) for k in ks])
    axes[2].set_xlabel("number of RBF centres")
    axes[2].set_ylabel("training time (ms)")
    axes[2].set_title("Training time vs centres (K-means + lstsq)")
    axes[2].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "p4_letters.png", dpi=120)
    plt.close(fig)


def solver_panel():
    """Closed-form linear output vs gradient descent."""
    X, y = make_moons(n_samples=400, noise=0.18, random_state=0)
    n_centres = 16

    t0 = time.perf_counter()
    net_ls = RBFNet(RBFConfig(n_centres=n_centres, output_solver="lstsq")).fit(X, y)
    t_ls = time.perf_counter() - t0
    acc_ls = float(((net_ls.predict(X).ravel() >= 0.5) == y).mean())

    t0 = time.perf_counter()
    net_gd = RBFNet(RBFConfig(n_centres=n_centres, output_solver="gradient",
                              output_lr=0.1, output_epochs=300)).fit(X, y)
    t_gd = time.perf_counter() - t0
    acc_gd = float(((net_gd.predict(X).ravel() >= 0.5) == y).mean())

    print(f"closed-form: {t_ls*1000:.1f} ms, acc={acc_ls:.3f}")
    print(f"gradient   : {t_gd*1000:.1f} ms, acc={acc_gd:.3f}")

    # Decision surface
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
                          np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = net_ls.predict(grid).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, Z >= 0.5, alpha=0.3, cmap="RdBu")
    ax.scatter(*net_ls.centres.T, c="k", marker="x", s=80, label="centres")
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=15, label="class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=15, label="class 1")
    ax.set_title("RBF decision surface (two moons)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "p4_moons.png", dpi=120)
    plt.close(fig)


def centres_panel(split):
    """What do the K-means centres look like? (the X-like prototypes)"""
    Xtr, ytr = split.X_train, split.y_train
    n_classes = split.n_classes
    net = RBFNet(RBFConfig(n_centres=60, output_solver="lstsq")).fit(
        Xtr, np.eye(n_classes)[ytr])
    w = net.W                      # (n_centres, n_classes)
    order = np.argsort(-w[:, split.target_index])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, k in zip(axes, order[:3]):
        c = net.centres[k].reshape(28, 28)
        ax.imshow(c, cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"centre #{k}\nread-out on X = {w[k, split.target_index]:.2f}",
                     fontsize=9)
        ax.axis("off")
    fig.suptitle("K-means centres that fire on the letter X", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "p4_centres.png", dpi=120)
    plt.close(fig)


def main():
    print("rendering letters dataset…")
    split = build_letters(seed=0)
    sine_panel()
    letters_panel(split)
    centres_panel(split)
    solver_panel()
    print("figures written to:", OUT)


if __name__ == "__main__":
    main()

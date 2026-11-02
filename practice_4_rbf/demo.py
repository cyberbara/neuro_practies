"""Demo for practice 4: RBF network.

Four experiments:
  1. Approximate a 1-D sine wave; sweep the number of centres.
  2. Handwritten-digit classification (sklearn digits, 8x8, 10 classes) —
     the real task. Sweep the number of centres.
  3. Closed-form (least squares) vs gradient-descent output layer —
     the headline "RBF trains fast" result.
  4. Two-moons decision surface with the K-means centres overlaid.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


def digits_data(seed=0):
    d = load_digits()
    X = StandardScaler().fit_transform(d.data)
    y = d.target.astype(np.int64)
    return train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)


def digits_panel():
    """Handwritten digit classification — the real task."""
    Xtr, Xte, ytr, yte = digits_data()
    ks = [10, 30, 60, 120]
    train_accs, test_accs, times = [], [], []

    for k in ks:
        # One-vs-rest RBF: 10 separate binary RBF nets, one per class.
        t0 = time.perf_counter()
        nets = []
        for c in range(10):
            net = RBFNet(RBFConfig(n_centres=k, output_solver="lstsq"))
            net.fit(Xtr, (ytr == c).astype(np.float64))
            nets.append(net)
        t = time.perf_counter() - t0
        # Predict: argmax over per-class RBF outputs.
        Pte = np.column_stack([n.predict(Xte) for n in nets])
        Ptr = np.column_stack([n.predict(Xtr) for n in nets])
        test_accs.append(float((Pte.argmax(1) == yte).mean()))
        train_accs.append(float((Ptr.argmax(1) == ytr).mean()))
        times.append(t)
        print(f"centres={k:4d} | train={train_accs[-1]:.3f} | "
              f"test={test_accs[-1]:.3f} | {t*1000:.0f} ms")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(ks))
    w = 0.4
    axes[0].bar(x - w/2, train_accs, w, label="train", color="C0")
    axes[0].bar(x + w/2, test_accs,  w, label="test",  color="C3")
    axes[0].set_xticks(x); axes[0].set_xticklabels([str(k) for k in ks])
    axes[0].set_xlabel("number of RBF centres")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Digits (8×8, 10 classes) — accuracy vs centres")
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0.5, 1.0)

    axes[1].bar(x, [t * 1000 for t in times], color="C2")
    axes[1].set_xticks(x); axes[1].set_xticklabels([str(k) for k in ks])
    axes[1].set_xlabel("number of RBF centres")
    axes[1].set_ylabel("training time (ms)")
    axes[1].set_title("Training time vs centres (K-means + lstsq)")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "p4_digits.png", dpi=120)
    plt.close(fig)


def solver_panel():
    """Closed-form linear output vs gradient descent."""
    X, y = make_moons(n_samples=400, noise=0.18, random_state=0)
    n_centres = 16

    t0 = time.perf_counter()
    net_ls = RBFNet(RBFConfig(n_centres=n_centres, output_solver="lstsq")).fit(X, y)
    t_ls = time.perf_counter() - t0
    acc_ls = float(((net_ls.predict(X) >= 0.5) == y).mean())

    t0 = time.perf_counter()
    net_gd = RBFNet(RBFConfig(n_centres=n_centres, output_solver="gradient",
                              output_lr=0.1, output_epochs=300)).fit(X, y)
    t_gd = time.perf_counter() - t0
    acc_gd = float(((net_gd.predict(X) >= 0.5) == y).mean())

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


def main():
    sine_panel()
    digits_panel()
    solver_panel()
    print("figures written to:", OUT)


if __name__ == "__main__":
    main()

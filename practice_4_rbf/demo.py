"""Demo for practice 4: RBF network — function approximation and classification.

Three experiments:
  1. Approximate a 1-D sine wave; sweep the number of centres.
  2. Two-moons classification — visualise centres and decision boundary.
  3. Compare closed-form linear output vs gradient-descent output.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

from practice_4_rbf.rbf import RBFConfig, RBFNet

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)


def main():
    # 1. Function approximation: y = sin(2x) on [-pi, pi] ------------------
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
    fig.tight_layout()
    fig.savefig(OUT / "p4_approx.png", dpi=120)
    plt.close(fig)

    # 2. Two-moons classification -----------------------------------------
    X, y = make_moons(n_samples=400, noise=0.18, random_state=0)
    net = RBFNet(RBFConfig(n_centres=30, output_solver="lstsq")).fit(X, y)
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
                          np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = net.predict(grid).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, Z >= 0.5, alpha=0.3, cmap="RdBu")
    ax.scatter(*net.centres.T, c="k", marker="x", s=80, label="centres")
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=15, label="class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=15, label="class 1")
    ax.set_title("RBF decision surface (two moons)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "p4_moons.png", dpi=120)
    plt.close(fig)

    # 3. Closed-form vs gradient output weights ----------------------------
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

    print("figures written to:", OUT)


if __name__ == "__main__":
    main()

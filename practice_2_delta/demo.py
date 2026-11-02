"""Demo for practice 2: delta-rule perceptron.

Three experiments:
  1. AND / OR / NAND / XOR — classical lesson. XOR doesn't converge.
  2. Breast-cancer Wisconsin — real binary classification (30 features).
     Train/test split, η sweep, convergence on a real linear-ish problem.
  3. Effect of weight initialisation on the toy AND task.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


def breast_cancer_data(seed=0):
    d = load_breast_cancer()
    X = StandardScaler().fit_transform(d.data)
    y = d.target.astype(np.float64)
    return train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)


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


def breast_cancer_panel():
    """Real binary classification: breast_cancer, 30 features."""
    Xtr, Xte, ytr, yte = breast_cancer_data()
    etas = [0.01, 0.05, 0.1, 0.5]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: train/test accuracy curves for each eta
    for eta in etas:
        net = Perceptron(n_input=Xtr.shape[1], activation="sigmoid",
                          eta=eta, seed=0)
        res = net.fit(Xtr, ytr, Xte, yte, epochs=300)
        axes[0].plot(res.train_acc_history, label=f"η={eta} train")
        axes[0].plot(res.test_acc_history,  label=f"η={eta} test", linestyle="--")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Breast-cancer Wisconsin — convergence vs η")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=0.3)

    # Right: final accuracy bar chart
    final_train, final_test = [], []
    for eta in etas:
        net = Perceptron(n_input=Xtr.shape[1], activation="sigmoid",
                          eta=eta, seed=0)
        res = net.fit(Xtr, ytr, Xte, yte, epochs=300)
        final_train.append(res.train_acc_history[-1])
        final_test.append(res.test_acc_history[-1])
    x = np.arange(len(etas))
    w = 0.4
    axes[1].bar(x - w/2, final_train, w, label="train", color="C0")
    axes[1].bar(x + w/2, final_test,  w, label="test",  color="C3")
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"η={e}" for e in etas])
    axes[1].set_ylabel("accuracy"); axes[1].set_ylim(0.85, 1.0)
    axes[1].set_title("Final train/test accuracy")
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "p2_breast.png", dpi=120)
    plt.close(fig)


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
    toy_panel()
    breast_cancer_panel()
    init_sweep()
    print("figures written to:", OUT)


if __name__ == "__main__":
    main()

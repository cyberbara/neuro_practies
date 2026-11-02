"""Demo for practice 2: delta-rule perceptron on AND / OR / NAND / XOR / Iris.

Shows the classic result that a single-layer perceptron cannot learn XOR,
plots the MSE-per-epoch curves, and visualises the decision boundary.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


def main():
    # --- AND / OR / NAND / XOR -----------------------------------------------
    problems = ["AND", "OR", "NAND", "XOR"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for col, name in enumerate(problems):
        X, y, model, res, dt = run_toy(name)
        # MSE / accuracy curve
        ax = axes[0, col]
        ax.plot(res.mse_history, label="MSE")
        ax.plot(res.accuracy_history, label="accuracy")
        ax.set_title(f"{name} (final acc={res.accuracy_history[-1]:.2f})")
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        # Decision region
        ax = axes[1, col]
        xx, yy, Z = decision_grid(model, X)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
        for cls in [0, 1]:
            mask = y == cls
            ax.scatter(X[mask, 0], X[mask, 1], label=f"class {cls}", s=80,
                       edgecolors="k")
        ax.set_title(f"{name} boundary (η={model.eta})")
        ax.set_xlabel("x1"); ax.set_ylabel("x2")
    fig.tight_layout()
    fig.savefig(OUT / "p2_toy.png", dpi=120)
    plt.close(fig)

    # --- Iris (subset, two classes) ------------------------------------------
    from sklearn.datasets import load_iris
    iris = load_iris()
    # Use only classes 0 and 1, two features for a 2-D plot
    X = iris.data[iris.target < 2, :2]
    y = iris.target[iris.target < 2]

    # Normalise features — makes the demo converge in a few hundred epochs.
    X = (X - X.mean(0)) / X.std(0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, eta in zip(axes, [0.05, 0.5]):
        model = Perceptron(n_input=2, activation="sigmoid", eta=eta, seed=42)
        res = model.fit(X, y.astype(float), epochs=300)
        ax.plot(res.mse_history, label="MSE")
        ax.plot(res.accuracy_history, label="accuracy")
        ax.set_title(f"Iris, η={eta}, final acc={res.accuracy_history[-1]:.3f}")
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "p2_iris.png", dpi=120)
    plt.close(fig)

    # --- Effect of weight initialisation -----------------------------------
    final_accs = []
    seeds = list(range(20))
    for seed in seeds:
        m = Perceptron(n_input=2, activation="sigmoid", eta=0.1, seed=seed)
        r = m.fit(*toy("AND"), epochs=100)
        final_accs.append(r.accuracy_history[-1])
    print("AND accuracy over 20 random seeds:", np.mean(final_accs),
          "±", np.std(final_accs))

    print("figures written to:", OUT)


if __name__ == "__main__":
    main()

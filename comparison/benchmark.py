"""Head-to-head benchmark of all five practices.

For each practice we measure:
  - parameters       (count of learned weights)
  - neurons          (functional units)
  - training_time_s  (wall clock on this machine)
  - accuracy         (problem-specific, best-effort)
  - notes            (free-text)

The five practices have very different problem setups, so the accuracy
column doesn't compare apples to apples. The wall-clock and parameter
columns do, and they tell the real story.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from common.datasets import iris, two_moons
from practice_1_hebb.hebb import HebbianNetwork
from practice_2_delta.perceptron import Perceptron
from practice_3_backprop.mlp import MLP, count_params as torch_count, fit as torch_fit
from practice_4_rbf.rbf import RBFConfig, RBFNet
from practice_5_som.som import SOM, SOMConfig

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

CSV_PATH = OUT / "benchmark.csv"


def row(name, params, neurons, t, acc, notes):
    return {
        "practice": name,
        "parameters": params,
        "neurons": neurons,
        "training_time_s": round(t, 4),
        "accuracy": round(acc, 4) if acc is not None else "n/a",
        "notes": notes,
    }


def run_practice_1():
    """Hebb: store 5 patterns, measure noisy-pattern recovery rate."""
    names, patterns = [], []
    for letter in ["X", "O", "T", "L", "I"]:
        # Five 5x5 bipolar letters from the practice 1 demo
        pass
    # Use the demo's patterns via the practice module directly.
    from practice_1_hebb.demo import LETTERS, to_patterns
    pnames, patterns = to_patterns()
    n_in = patterns.shape[1]
    net = HebbianNetwork(n_in, rule="oja", eta=0.05)
    t0 = time.perf_counter()
    net.train(patterns, epochs=10)
    t = time.perf_counter() - t0
    acc = net.noisy_recovery(patterns, noise=0.2, n_trials=500)
    return row("1_hebb_oja", net.W.size, n_in, t, acc,
               f"6 letter patterns, 25 neurons, 20% noise")


def run_practice_2():
    """Delta: AND toy problem."""
    from practice_2_delta.perceptron import toy
    X, y = toy("AND")
    net = Perceptron(n_input=2, activation="sigmoid", eta=0.5, seed=0)
    t0 = time.perf_counter()
    res = net.fit(X, y.astype(float), epochs=200)
    t = time.perf_counter() - t0
    return row("2_delta_and", net.W.size + 1, 1, t, res.accuracy_history[-1],
               "AND toy, sigmoid, 200 epochs")


def run_practice_3():
    """Backprop: small MLP on Fashion-MNIST (subsampled for speed)."""
    import torch
    from practice_3_backprop.mlp import load_fashion_mnist
    Xtr, ytr, Xte, yte = load_fashion_mnist(limit_train=10_000, limit_test=2_000)
    n_in = Xtr.shape[1]
    n_out = int(max(ytr.max(), yte.max()).item()) + 1
    torch.manual_seed(0)
    model = MLP(n_in=n_in, n_hidden=64, n_out=n_out, activation="relu")
    t0 = time.perf_counter()
    res = torch_fit(model, Xtr, ytr, Xte, yte, epochs=3, optimiser="adam", lr=1e-3)
    t = time.perf_counter() - t0
    return row("3_backprop_mlp", res.params, res.params // 10,
               t, res.history["test_acc"][-1],
               f"Fashion-MNIST 10k subset ({n_in}-dim), 64-hidden, 3 epochs, Adam")


def run_practice_4():
    """RBF: two-moons classification, 30 centres."""
    X, y = two_moons()
    net = RBFNet(RBFConfig(n_centres=30, output_solver="lstsq"))
    t0 = time.perf_counter()
    net.fit(X, y)
    t = time.perf_counter() - t0
    acc = float(((net.predict(X) >= 0.5) == y).mean())
    return row("4_rbf_moons", net.W.size + (1 if np.isscalar(net.b) else net.b.size),
               30, t, acc, "two-moons, K-means centres, closed-form output")


def run_practice_5():
    """SOM: 8x8 map on Iris."""
    X, y, _ = iris()
    cfg = SOMConfig(n_rows=8, n_cols=8, n_epochs=50)
    som = SOM(cfg)
    t0 = time.perf_counter()
    som.fit(X)
    t = time.perf_counter() - t0
    # Quantisation error is the natural SOM accuracy proxy.
    d2 = np.min(np.linalg.norm(som.W[None, :, :] - X[:, None, :], axis=2), axis=1)
    qe = float(d2.mean())
    return row("5_som_iris", som.W.size, som.W.shape[0], t, qe,
               "Iris 150x4, 8x8 map, 50 epochs (lower=better)")


def main():
    import torch  # local import; only needed for practice 3
    results = []
    print("running benchmark…")
    for fn in (run_practice_1, run_practice_2, run_practice_3,
               run_practice_4, run_practice_5):
        name = fn.__name__
        try:
            r = fn()
            results.append(r)
            print(f"  {r['practice']:18s} | t={r['training_time_s']:.3f}s | "
                  f"params={r['parameters']:>8} | metric={r['accuracy']}")
        except Exception as e:
            print(f"  {name} FAILED: {e}")
            results.append({"practice": name, "parameters": "err",
                            "neurons": "err", "training_time_s": 0,
                            "accuracy": "err", "notes": str(e)})

    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["practice", "parameters",
                                          "neurons", "training_time_s",
                                          "accuracy", "notes"])
        w.writeheader()
        w.writerows(results)
    print(f"\nresults written to {CSV_PATH}")


if __name__ == "__main__":
    main()

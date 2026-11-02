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

import matplotlib.pyplot as plt
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
PNG_BARS = OUT / "comparison_bars.png"
PNG_SCATTER = OUT / "comparison_scatter.png"


def row(name, params, neurons, t, metric, metric_label, notes):
    return {
        "practice": name,
        "parameters": params,
        "neurons": neurons,
        "training_time_s": round(t, 4),
        "metric_value": round(metric, 4) if metric is not None else None,
        "metric_label": metric_label,
        "notes": notes,
    }


def run_practice_1():
    """Hebb: real letter images, measure recovery under noise."""
    from practice_1_hebb.renderer import make_dataset
    _, patterns = make_dataset(letters="ABCDEFGHIJ", size=28)
    n_in = patterns.shape[1]
    net = HebbianNetwork(n_in, rule="oja", eta=1e-4)
    t0 = time.perf_counter()
    net.train(patterns, epochs=5)
    t = time.perf_counter() - t0
    acc = net.noisy_recovery(patterns, noise=0.2, n_trials=300)
    return row("1_hebb_letters", net.W.size, n_in, t, acc,
               "recovery_rate",
               "40 PIL letter renders (10 letters x 4 fonts, 28x28), 20% noise")


def run_practice_2():
    """Delta: breast-cancer Wisconsin — real binary classification."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    d = load_breast_cancer()
    X = StandardScaler().fit_transform(d.data)
    y = d.target.astype(np.float64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                           random_state=0, stratify=y)
    net = Perceptron(n_input=Xtr.shape[1], activation="sigmoid", eta=0.1, seed=0)
    t0 = time.perf_counter()
    res = net.fit(Xtr, ytr, Xte, yte, epochs=300)
    t = time.perf_counter() - t0
    return row("2_delta_breast", net.W.size + 1, 1, t, res.test_acc_history[-1],
               "test_accuracy",
               "breast-cancer Wisconsin, 30 features, 25% test split, 300 epochs")


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
               "test_accuracy",
               f"Fashion-MNIST 10k subset ({n_in}-dim), 64-hidden, 3 epochs, Adam")


def run_practice_4():
    """RBF: handwritten-digit classification, one-vs-rest RBF nets."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    d = load_digits()
    X = StandardScaler().fit_transform(d.data)
    y = d.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                           random_state=0, stratify=y)
    n_centres = 60
    t0 = time.perf_counter()
    nets = []
    for c in range(10):
        net = RBFNet(RBFConfig(n_centres=n_centres, output_solver="lstsq"))
        net.fit(Xtr, (ytr == c).astype(np.float64))
        nets.append(net)
    t = time.perf_counter() - t0
    P = np.column_stack([n.predict(Xte) for n in nets])
    acc = float((P.argmax(1) == yte).mean())
    params = sum(n.W.size + (1 if np.isscalar(n.b) else n.b.size) for n in nets)
    return row("4_rbf_digits", params, n_centres * 10, t, acc,
               "test_accuracy",
               "sklearn digits (8x8, 10 classes), 60 centres/class, one-vs-rest")


def run_practice_5():
    """SOM: 16x16 map on handwritten digits."""
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import StandardScaler
    d = load_digits()
    X = StandardScaler().fit_transform(d.data)
    cfg = SOMConfig(n_rows=16, n_cols=16, n_epochs=60)
    som = SOM(cfg)
    t0 = time.perf_counter()
    som.fit(X)
    t = time.perf_counter() - t0
    d2 = np.min(np.linalg.norm(som.W[None, :, :] - X[:, None, :], axis=2), axis=1)
    qe = float(d2.mean())
    return row("5_som_digits", som.W.size, som.W.shape[0], t, qe,
               "quantisation_error",
               "sklearn digits (8x8), 16x16 map, 60 epochs (lower=better)")


# ----------------------------------------------------------------- plotting
def plot_bars(results):
    """Three-panel bar chart: training time, parameters, metric value."""
    names = [r["practice"] for r in results]
    times = [r["training_time_s"] for r in results]
    params = [r["parameters"] for r in results]
    raw = [r["metric_value"] for r in results]
    # SOM uses quantisation error (lower=better); flip it visually so all
    # three panels show "higher is better".
    is_lower_better = [r["metric_label"] == "quantisation_error" for r in results]
    display = [1.0 / m if f else m for m, f in zip(raw, is_lower_better)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colours = plt.cm.tab10(np.arange(len(names)))
    for ax, vals, title, ylabel in zip(
        axes, [times, params, display],
        ["training time (s)", "trainable parameters", "metric (higher = better)"],
        ["seconds", "params", "score"]):
        ax.bar(names, vals, color=colours)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Comparison of the five practices", fontsize=14)
    fig.tight_layout()
    fig.savefig(PNG_BARS, dpi=120)
    plt.close(fig)


def plot_scatter(results):
    """Time vs metric, bubble size ∝ params."""
    times = np.array([r["training_time_s"] for r in results])
    params = np.array([r["parameters"] for r in results], dtype=float)
    raw = np.array([r["metric_value"] for r in results])
    is_lower_better = np.array([r["metric_label"] == "quantisation_error"
                                for r in results])
    score = np.where(is_lower_better, 1.0 / raw, raw)
    sizes = 60 + 4 * np.log1p(params)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(times, score, s=sizes, alpha=0.6,
               c=plt.cm.tab10(np.arange(len(results))))
    for r, t, s in zip(results, times, score):
        ax.annotate(r["practice"], (t, s), fontsize=8,
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("training time (s, log)")
    ax.set_ylabel("metric score (higher = better)")
    ax.set_title("Time vs score — bubble size ∝ log params")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PNG_SCATTER, dpi=120)
    plt.close(fig)


def main():
    import torch  # noqa: F401  (used transitively by practice 3)
    results = []
    print("running benchmark…")
    for fn in (run_practice_1, run_practice_2, run_practice_3,
               run_practice_4, run_practice_5):
        name = fn.__name__
        try:
            r = fn()
            results.append(r)
            metric_str = (f"{r['metric_value']:.4f}"
                          if r["metric_value"] is not None else "n/a")
            print(f"  {r['practice']:18s} | t={r['training_time_s']:.3f}s | "
                  f"params={r['parameters']:>8} | "
                  f"{r['metric_label']}={metric_str}")
        except Exception as e:
            print(f"  {name} FAILED: {e}")
            results.append({
                "practice": name, "parameters": "err",
                "neurons": "err", "training_time_s": 0,
                "metric_value": None, "metric_label": "err",
                "notes": str(e)})

    # CSV
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["practice", "parameters",
                                          "neurons", "training_time_s",
                                          "metric_value", "metric_label",
                                          "notes"])
        w.writeheader()
        w.writerows(results)

    # Plots
    ok = [r for r in results if r["metric_value"] is not None]
    if len(ok) >= 2:
        plot_bars(ok)
        plot_scatter(ok)
        print(f"plots: {PNG_BARS.name}, {PNG_SCATTER.name}")
    print(f"csv:   {CSV_PATH.name}")


if __name__ == "__main__":
    main()

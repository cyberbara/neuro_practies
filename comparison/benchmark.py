"""Head-to-head benchmark of all five practices — one task, one dataset.

Every practice solves the *same* problem on the *same* data: the letters
dataset (26 uppercase letters, 28×28, train fonts ≠ test fonts), and the
question is **"is this picture the letter X?"**.

For each practice we measure:
  - parameters        — number of learned weights
  - neurons           — functional units
  - training_time_s   — wall clock on this machine
  - balanced_accuracy — headline metric (see below)
  - accuracy          — plain accuracy
  - auc               — ROC-AUC of the X score
  - recall / precision — for the X class itself
  - class_accuracy    — accuracy over all 26 letters (where the practice
                        can classify all of them; the single-neuron delta
                        rule can only answer the binary question)

Why balanced accuracy and not plain accuracy: X is 1 class out of 26, so a
network that always answers "not X" already gets 96% accuracy. Balanced
accuracy weighs "did it find the X" and "did it leave the others alone"
equally, so the numbers are comparable across practices.

Because the dataset and the task are shared, this table *is* a ranking.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.letters import build_letters
from common.metrics import binary_scores

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

CSV_PATH = OUT / "benchmark.csv"
PNG_BARS = OUT / "comparison_bars.png"
PNG_SCATTER = OUT / "comparison_scatter.png"


def row(name, params, neurons, t, m, class_acc=None, notes=""):
    return {
        "practice": name,
        "parameters": int(params),
        "neurons": int(neurons),
        "training_time_s": round(t, 4),
        "balanced_accuracy": round(m["balanced_accuracy"], 4),
        "accuracy": round(m["accuracy"], 4),
        "auc": round(m["auc"], 4),
        "recall_x": round(m["recall"], 4),
        "precision_x": round(m["precision"], 4),
        "f1_x": round(m["f1"], 4),
        "class_accuracy": (round(class_acc, 4)
                           if class_acc is not None else None),
        "notes": notes,
    }


# ----------------------------------------------------------------- practices
def run_practice_1(split):
    """Hebb: one prototype neuron per letter, cosine readout."""
    from practice_1_hebb.hebb import HebbianClassifier

    net = HebbianClassifier(split.n_classes, split.X_train.shape[1],
                            eta=0.05, temperature=0.05)
    t0 = time.perf_counter()
    net.train(split.X_train, split.y_train, epochs=3)
    t = time.perf_counter() - t0
    pred = net.predict(split.X_test)
    p_x = net.probabilities(split.X_test)[:, split.target_index]
    m = binary_scores(split.binary_targets("test"), p_x,
                      decision=(pred == split.target_index).astype(float))
    return row("1_hebb_letters", net.W.size, split.n_classes, t, m,
               class_acc=float((pred == split.y_test).mean()),
               notes="26 prototype neurons, cosine readout, 3 epochs")


def run_practice_2(split):
    """Delta rule: one sigmoid neuron, 784 inputs, target = is-X."""
    from practice_2_delta.perceptron import Perceptron

    net = Perceptron(n_input=split.X_train.shape[1], activation="sigmoid",
                     eta=0.5, seed=0)
    t0 = time.perf_counter()
    net.fit(split.X_train, split.binary_targets("train"), epochs=60)
    t = time.perf_counter() - t0
    m = binary_scores(split.binary_targets("test"), net.predict_proba(split.X_test))
    return row("2_delta_letters", net.W.size + 1, 1, t, m,
               notes="single sigmoid neuron, binary X / not-X, 60 epochs")


def run_practice_3(split):
    """Backprop: 2-hidden-layer MLP over 26 classes."""
    import torch
    from practice_3_backprop.mlp import MLP, fit, load_letters_tensors

    Xtr, ytr, Xte, yte = load_letters_tensors(split, "cpu")
    torch.manual_seed(0)
    model = MLP(n_in=Xtr.shape[1], n_hidden=128, n_out=split.n_classes,
                activation="relu")
    t0 = time.perf_counter()
    res = fit(model, Xtr, ytr, Xte, yte, epochs=25, optimiser="adam", lr=1e-3)
    t = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(Xte), dim=1).numpy()
    pred = probs.argmax(1)
    m = binary_scores(split.binary_targets("test"), probs[:, split.target_index],
                      decision=(pred == split.target_index).astype(float))
    return row("3_backprop_mlp", res.params, split.n_classes, t, m,
               class_acc=float((pred == split.y_test).mean()),
               notes="784-128-128-26 MLP, ReLU, Adam, 25 epochs")


def run_practice_4(split):
    """RBF: 120 Gaussian centres + 26 linear read-outs (closed form)."""
    from practice_4_rbf.rbf import RBFConfig, RBFNet

    Ytr = np.eye(split.n_classes)[split.y_train]
    net = RBFNet(RBFConfig(n_centres=120, output_solver="lstsq"))
    t0 = time.perf_counter()
    net.fit(split.X_train, Ytr)
    t = time.perf_counter() - t0
    P = net.predict(split.X_test)
    pred = P.argmax(1)
    z = P / 0.1
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    probs = e / e.sum(axis=1, keepdims=True)
    m = binary_scores(split.binary_targets("test"), probs[:, split.target_index],
                      decision=(pred == split.target_index).astype(float))
    params = net.W.size + net.b.size
    return row("4_rbf_letters", params, net.centres.shape[0], t, m,
               class_acc=float((pred == split.y_test).mean()),
               notes="120 K-means centres + 26 lstsq read-outs (closed form)")


def run_practice_5(split):
    """SOM: 16x16 map, labelled afterwards, BMU votes on the class."""
    from practice_5_som.som import SOM, SOMConfig

    cfg = SOMConfig(n_rows=16, n_cols=16, n_epochs=200, seed=0,
                    update_mode="batch")
    som = SOM(cfg)
    t0 = time.perf_counter()
    som.fit(split.X_train)
    t = time.perf_counter() - t0
    flat = som.map_labels(split.X_train, split.y_train).ravel()
    pred = som.classify(split.X_test, flat)
    p_x = som.target_score(split.X_test, flat, split.target_index, k=5)
    m = binary_scores(split.binary_targets("test"), p_x,
                      decision=(pred == split.target_index).astype(float))
    return row("5_som_letters", som.W.size, som.W.shape[0], t, m,
               class_acc=float((pred == split.y_test).mean()),
               notes="16x16 SOM, unsupervised, map labelled after training")


# ----------------------------------------------------------------- plotting
def plot_bars(results):
    """Three-panel bar chart: quality, training time, parameters."""
    names = [r["practice"] for r in results]
    times = [r["training_time_s"] for r in results]
    params = [r["parameters"] for r in results]
    bal = [r["balanced_accuracy"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colours = plt.cm.tab10(np.arange(len(names)))
    panels = [
        (bal, "balanced accuracy (find X)", "score"),
        (times, "training time (s)", "seconds"),
        (params, "trainable parameters (log)", "params"),
    ]
    for ax, (vals, title, ylabel) in zip(axes, panels):
        ax.bar(names, vals, color=colours)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            fmt = f"{v:.0f}" if v >= 100 else f"{v:.2f}"
            ax.text(i, v, fmt, ha="center", va="bottom", fontsize=8)
    axes[0].set_ylim(0, 1.1)
    # Parameter counts span 785 → 120 000, so a linear axis hides the small
    # ones entirely. Log scale makes every practice visible.
    axes[2].set_yscale("log")
    axes[2].set_ylim(10, 1e6)
    fig.suptitle("Five learning rules — one task: find the letter X", fontsize=14)
    fig.tight_layout()
    fig.savefig(PNG_BARS, dpi=120)
    plt.close(fig)


def plot_scatter(results):
    """Time vs quality, bubble size ∝ params."""
    times = np.array([r["training_time_s"] for r in results])
    params = np.array([r["parameters"] for r in results], dtype=float)
    bal = np.array([r["balanced_accuracy"] for r in results])
    sizes = 60 + 4 * np.log1p(params)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(times, bal, s=sizes, alpha=0.65,
               c=plt.cm.tab10(np.arange(len(results))))
    for r, t, b in zip(results, times, bal):
        ax.annotate(r["practice"], (t, b), fontsize=8,
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("training time (s, log)")
    ax.set_ylabel("balanced accuracy (find X)")
    ax.set_ylim(0.3, 1.05)
    ax.set_title("Time vs quality — bubble size ∝ log params")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PNG_SCATTER, dpi=120)
    plt.close(fig)


def main():
    print("rendering letters dataset…")
    split = build_letters(seed=0)
    print(f"train {split.X_train.shape}, test {split.X_test.shape}, "
          f"target = {split.target}")

    results = []
    for fn in (run_practice_1, run_practice_2, run_practice_3,
               run_practice_4, run_practice_5):
        try:
            r = fn(split)
            results.append(r)
            print(f"  {r['practice']:16s} | t={r['training_time_s']:7.3f}s | "
                  f"params={r['parameters']:>7} | "
                  f"bal_acc={r['balanced_accuracy']:.3f} | "
                  f"auc={r['auc']:.3f} | "
                  f"26-class={r['class_accuracy']}")
        except Exception as e:                                  # noqa: BLE001
            print(f"  {fn.__name__} FAILED: {e}")
            results.append({
                "practice": fn.__name__, "parameters": 0, "neurons": 0,
                "training_time_s": 0.0, "balanced_accuracy": 0.0,
                "accuracy": 0.0, "auc": 0.0, "recall_x": 0.0,
                "precision_x": 0.0, "f1_x": 0.0, "class_accuracy": None,
                "notes": f"FAILED: {e}"})

    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    ok = [r for r in results if r["balanced_accuracy"] > 0]
    if len(ok) >= 2:
        plot_bars(ok)
        plot_scatter(ok)
        print(f"plots: {PNG_BARS.name}, {PNG_SCATTER.name}")
    print(f"csv:   {CSV_PATH.name}")


if __name__ == "__main__":
    main()

"""Demo for practice 3: MLP backprop on Fashion-MNIST.

Runs four small experiments:
  1. Hidden-size sweep (32 / 64 / 128 / 256) on a 2-layer MLP.
  2. Activation comparison (ReLU vs Tanh).
  3. Optimiser comparison (SGD vs Adam).
  4. Dropout (0.0 / 0.25 / 0.5).

Each experiment logs train/test curves and saves a separate plot.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from practice_3_backprop.mlp import MLP, fit, load_fashion_mnist

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def plot_curves(history, title, fname):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["test_loss"], label="test")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.set_title(f"{title} — loss")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(history["train_acc"], label="train")
    ax2.plot(history["test_acc"], label="test")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy")
    ax2.set_title(f"{title} — accuracy"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def main():
    print("loading Fashion-MNIST…")
    Xtr, ytr, Xte, yte = load_fashion_mnist()
    Xtr, ytr, Xte, yte = Xtr.to(DEVICE), ytr.to(DEVICE), Xte.to(DEVICE), yte.to(DEVICE)
    print(f"train: {Xtr.shape}, test: {Xte.shape}")

    # 1. Hidden-size sweep
    for h in (32, 64, 128, 256):
        torch.manual_seed(0)
        model = MLP(n_in=784, n_hidden=h, n_out=10, activation="relu").to(DEVICE)
        res = fit(model, Xtr, ytr, Xte, yte, epochs=5, optimiser="adam", lr=1e-3)
        print(f"hidden={h:3d} | params={res.params:>7d} | "
              f"train_acc={res.history['train_acc'][-1]:.3f} | "
              f"test_acc={res.history['test_acc'][-1]:.3f} | "
              f"time={res.wall_time:.1f}s")
        plot_curves(res.history, f"hidden={h}", OUT / f"p3_hidden_{h}.png")

    # 2. Activation comparison
    for act in ("relu", "tanh"):
        torch.manual_seed(0)
        model = MLP(n_in=784, n_hidden=128, n_out=10, activation=act).to(DEVICE)
        res = fit(model, Xtr, ytr, Xte, yte, epochs=5, optimiser="adam", lr=1e-3)
        print(f"{act:5s} | final test_acc={res.history['test_acc'][-1]:.3f}")
        plot_curves(res.history, f"activation={act}", OUT / f"p3_act_{act}.png")

    # 3. Optimiser comparison
    for opt in ("sgd", "adam"):
        torch.manual_seed(0)
        model = MLP(n_in=784, n_hidden=128, n_out=10, activation="relu").to(DEVICE)
        res = fit(model, Xtr, ytr, Xte, yte, epochs=5, optimiser=opt,
                  lr=1e-2 if opt == "sgd" else 1e-3)
        print(f"{opt:5s} | final test_acc={res.history['test_acc'][-1]:.3f}")
        plot_curves(res.history, f"optimiser={opt}", OUT / f"p3_opt_{opt}.png")

    # 4. Dropout sweep
    for p in (0.0, 0.25, 0.5):
        torch.manual_seed(0)
        model = MLP(n_in=784, n_hidden=256, n_out=10, activation="relu",
                    dropout=p).to(DEVICE)
        res = fit(model, Xtr, ytr, Xte, yte, epochs=8, optimiser="adam", lr=1e-3)
        print(f"dropout={p:.2f} | final test_acc={res.history['test_acc'][-1]:.3f}")
        plot_curves(res.history, f"dropout={p}", OUT / f"p3_drop_{p:.2f}.png")

    print("figures written to:", OUT)


if __name__ == "__main__":
    main()

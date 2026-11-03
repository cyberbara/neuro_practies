"""Demo for practice 3: MLP backprop on the letters dataset.

Five experiments:
  1. Hidden-size sweep (32 / 64 / 128 / 256).
  2. Activation comparison (ReLU vs Tanh).
  3. Optimiser comparison (SGD vs Adam).
  4. Dropout (0.0 / 0.25 / 0.5).
  5. Data augmentation (random shift + rotation) — shows the
     overfitting-reduction effect.

The network classifies all 26 letters; the headline number for the repo is
the "find the letter X" score: softmax probability of the X neuron, with the
decision "argmax == X".
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common.letters import build_letters
from common.metrics import binary_scores
from practice_3_backprop.mlp import MLP, count_params, fit, load_letters_tensors

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


# ------------------------------------------------------- augmentation helpers
def random_shift(x: torch.Tensor, max_shift: int = 2) -> torch.Tensor:
    """Randomly shift a (N, 1, H, W) tensor by up to max_shift pixels."""
    n, _, h, w = x.shape
    shifts_x = torch.randint(-max_shift, max_shift + 1, (n,), device=x.device)
    shifts_y = torch.randint(-max_shift, max_shift + 1, (n,), device=x.device)
    out = torch.zeros_like(x)
    for i in range(n):
        sx = int(shifts_x[i].item())
        sy = int(shifts_y[i].item())
        if sx == 0 and sy == 0:
            out[i] = x[i]
            continue
        # Source and destination slices — clipped to image bounds.
        if sx >= 0:
            src_x = slice(0, w - sx); dst_x = slice(sx, w)
        else:
            src_x = slice(-sx, w); dst_x = slice(0, w + sx)
        if sy >= 0:
            src_y = slice(0, h - sy); dst_y = slice(sy, h)
        else:
            src_y = slice(-sy, h); dst_y = slice(0, h + sy)
        out[i, :, dst_y, dst_x] = x[i, :, src_y, src_x]
    return out


def augment_batch(x_flat: torch.Tensor) -> torch.Tensor:
    """Reshape (N, D) to (N, 1, S, S), random-shift, rotate, flatten back.

    S is inferred from D, so this works for 8x8 digits (D=64) and 28x28
    letters (D=784) alike.
    """
    n, d = x_flat.shape
    side = int(round(d ** 0.5))
    if side * side != d:
        raise ValueError(f"feature count {d} is not a perfect square")
    x = x_flat.view(n, 1, side, side)
    x = random_shift(x, max_shift=max(1, side // 14))
    # Random small rotation via affine grid (PyTorch built-in).
    angle = (torch.rand(n, device=x.device) - 0.5) * 20.0  # ±10°
    theta = torch.zeros(n, 2, 3, device=x.device)
    cos = torch.cos(torch.deg2rad(angle))
    sin = torch.sin(torch.deg2rad(angle))
    theta[:, 0, 0] = cos; theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin; theta[:, 1, 1] = cos
    grid = nn.functional.affine_grid(theta, x.size(), align_corners=False)
    x = nn.functional.grid_sample(x, grid, mode="bilinear",
                                   padding_mode="zeros", align_corners=False)
    return x.view(n, d)


def fit_with_augment(model, Xtr, ytr, Xte, yte, epochs=8, batch_size=128,
                     lr=1e-3):
    """Same loop as `fit`, but augments each batch on the fly."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size,
                          shuffle=True)
    history = {"train_loss": [], "train_acc": [],
               "test_loss": [], "test_acc": []}
    for _ in range(epochs):
        model.train()
        tot_loss = tot_correct = 0
        for xb, yb in train_dl:
            xb = augment_batch(xb)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward(); opt.step()
            tot_loss += loss.item() * xb.size(0)
            tot_correct += (out.argmax(1) == yb).sum().item()
        history["train_loss"].append(tot_loss / len(Xtr))
        history["train_acc"].append(tot_correct / len(Xtr))
        model.eval()
        with torch.no_grad():
            out = model(Xte)
            history["test_loss"].append(loss_fn(out, yte).item())
            history["test_acc"].append((out.argmax(1) == yte).float().mean().item())
    from practice_3_backprop.mlp import FitResult
    return FitResult(history, count_params(model), 0.0)


def x_scores(model, Xte, target_index):
    """Softmax probability of the X class + the argmax decision."""
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(Xte), dim=1).cpu().numpy()
    pred = probs.argmax(1)
    return probs[:, target_index], (pred == target_index).astype(float)


def main():
    print("rendering letters dataset…")
    split = build_letters(seed=0)
    Xtr, ytr, Xte, yte = load_letters_tensors(split, DEVICE)
    print(f"train: {Xtr.shape}, test: {Xte.shape}")
    n_in = Xtr.shape[1]
    n_out = int(max(ytr.max(), yte.max()).item()) + 1
    target_index = split.target_index
    y_true_bin = split.binary_targets("test")

    # 1. Hidden-size sweep
    for h in (32, 64, 128, 256):
        torch.manual_seed(0)
        model = MLP(n_in=n_in, n_hidden=h, n_out=n_out, activation="relu").to(DEVICE)
        res = fit(model, Xtr, ytr, Xte, yte, epochs=25, optimiser="adam", lr=1e-3)
        p_x, dec = x_scores(model, Xte, target_index)
        m = binary_scores(y_true_bin, p_x, decision=dec)
        print(f"hidden={h:3d} | params={res.params:>7d} | "
              f"test_acc={res.history['test_acc'][-1]:.3f} | "
              f"bal_acc(X)={m['balanced_accuracy']:.3f} | "
              f"time={res.wall_time:.1f}s")
        plot_curves(res.history, f"hidden={h}", OUT / f"p3_hidden_{h}.png")

    # 2. Activation comparison
    for act in ("relu", "tanh"):
        torch.manual_seed(0)
        model = MLP(n_in=n_in, n_hidden=128, n_out=n_out, activation=act).to(DEVICE)
        res = fit(model, Xtr, ytr, Xte, yte, epochs=25, optimiser="adam", lr=1e-3)
        p_x, dec = x_scores(model, Xte, target_index)
        m = binary_scores(y_true_bin, p_x, decision=dec)
        print(f"{act:5s} | final test_acc={res.history['test_acc'][-1]:.3f} | "
              f"bal_acc(X)={m['balanced_accuracy']:.3f}")
        plot_curves(res.history, f"activation={act}", OUT / f"p3_act_{act}.png")

    # 3. Optimiser comparison
    for opt in ("sgd", "adam"):
        torch.manual_seed(0)
        model = MLP(n_in=n_in, n_hidden=128, n_out=n_out, activation="relu").to(DEVICE)
        res = fit(model, Xtr, ytr, Xte, yte, epochs=25, optimiser=opt,
                  lr=1e-2 if opt == "sgd" else 1e-3)
        p_x, dec = x_scores(model, Xte, target_index)
        m = binary_scores(y_true_bin, p_x, decision=dec)
        print(f"{opt:5s} | final test_acc={res.history['test_acc'][-1]:.3f} | "
              f"bal_acc(X)={m['balanced_accuracy']:.3f}")
        plot_curves(res.history, f"optimiser={opt}", OUT / f"p3_opt_{opt}.png")

    # 4. Dropout sweep
    for p in (0.0, 0.25, 0.5):
        torch.manual_seed(0)
        model = MLP(n_in=n_in, n_hidden=256, n_out=n_out, activation="relu",
                    dropout=p).to(DEVICE)
        res = fit(model, Xtr, ytr, Xte, yte, epochs=25, optimiser="adam", lr=1e-3)
        p_x, dec = x_scores(model, Xte, target_index)
        m = binary_scores(y_true_bin, p_x, decision=dec)
        print(f"dropout={p:.2f} | final test_acc={res.history['test_acc'][-1]:.3f} | "
              f"bal_acc(X)={m['balanced_accuracy']:.3f}")
        plot_curves(res.history, f"dropout={p}", OUT / f"p3_drop_{p:.2f}.png")

    # 5. Augmentation comparison
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    histories = {}
    for label, augment in [("no aug", False), ("aug (shift+rot)", True)]:
        torch.manual_seed(0)
        model = MLP(n_in=n_in, n_hidden=256, n_out=n_out, activation="relu",
                    dropout=0.25).to(DEVICE)
        if augment:
            res = fit_with_augment(model, Xtr, ytr, Xte, yte, epochs=40)
        else:
            res = fit(model, Xtr, ytr, Xte, yte, epochs=40,
                      optimiser="adam", lr=1e-3)
        histories[label] = res.history
        p_x, dec = x_scores(model, Xte, target_index)
        m = binary_scores(y_true_bin, p_x, decision=dec)
        print(f"{label:14s} | final test_acc={res.history['test_acc'][-1]:.3f} | "
              f"bal_acc(X)={m['balanced_accuracy']:.3f}")
    for label, h in histories.items():
        axes[0].plot(h["train_acc"], label=label)
        axes[0].plot(h["test_acc"],  label=f"{label} (test)", linestyle="--")
        axes[1].plot(h["train_loss"], label=label)
        axes[1].plot(h["test_loss"],  label=f"{label} (test)", linestyle="--")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("accuracy")
    axes[0].set_title("Augmentation — accuracy"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("loss")
    axes[1].set_title("Augmentation — loss"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "p3_augment.png", dpi=120)
    plt.close(fig)

    # 6. X-detector view: score distribution + example pictures
    torch.manual_seed(0)
    model = MLP(n_in=n_in, n_hidden=128, n_out=n_out, activation="relu").to(DEVICE)
    fit(model, Xtr, ytr, Xte, yte, epochs=25, optimiser="adam", lr=1e-3)
    p_x, dec = x_scores(model, Xte, target_index)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].hist(p_x[y_true_bin == 0], bins=40, alpha=0.7, label="not X",
                 color="tab:blue")
    axes[0].hist(p_x[y_true_bin == 1], bins=40, alpha=0.8, label="X",
                 color="tab:red")
    axes[0].set_xlabel("P(X)")
    axes[0].set_title("score distribution")
    axes[0].legend(fontsize=9)
    Xte_np = split.X_test
    order = np.argsort(-p_x)
    tp = next(i for i in order if y_true_bin[i] == 1)
    fp = next(i for i in order if y_true_bin[i] == 0)
    axes[1].imshow(Xte_np[tp].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
    axes[1].set_title(f"true X\nP={p_x[tp]:.2f}", fontsize=9, color="tab:red")
    axes[2].imshow(Xte_np[fp].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
    axes[2].set_title(f"actually {split.char_test[fp]}\nP(X)={p_x[fp]:.2f}",
                      fontsize=9, color="tab:blue")
    for a in axes[1:]:
        a.axis("off")
    fig.suptitle("Backprop MLP finds the letter X", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "p3_detection.png", dpi=120)
    plt.close(fig)

    print("figures written to:", OUT)


if __name__ == "__main__":
    main()

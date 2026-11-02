"""Demo for practice 1: Hebbian / Oja associative memory on real letter images.

Generates 10 letters × 4 fonts (40 patterns total) using PIL, trains both
Hebb and Oja networks, and measures:
  - pattern overlap (cosine) after training
  - bit-level recovery under random noise
  - bit-level recovery under rectangular occlusion
  - stability: ‖W‖ across learning rates
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from practice_1_hebb.hebb import HebbianNetwork
from practice_1_hebb.renderer import (DEFAULT_FONTS, make_dataset, noisy,
                                       occlude)

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

SIZE = 28
LETTERS = "ABCDEFGHIJ"


def show_letters(labels, patterns, fname, n_per_letter=2):
    """Show a grid of sample letters: n_per_letter of each class."""
    fig, axes = plt.subplots(n_per_letter, len(LETTERS), figsize=(len(LETTERS) * 1.4, n_per_letter * 1.4))
    seen = {ch: 0 for ch in LETTERS}
    pick = []
    for label, p in zip(labels, patterns):
        ch = label.split(":")[0]
        if seen[ch] < n_per_letter:
            pick.append((ch, p))
            seen[ch] += 1
        if all(v == n_per_letter for v in seen.values()):
            break
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")
    for j, ch in enumerate(LETTERS):
        cell = [p for c, p in pick if c == ch]
        for i, p in enumerate(cell):
            axes[i, j].imshow(p.reshape(SIZE, SIZE), cmap="gray_r",
                              vmin=-1, vmax=1)
        axes[0, j].set_title(ch, fontsize=10)
    fig.suptitle("Sample letter patterns (2 per letter, mixed fonts)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def recovery_curve(net, patterns, kind="noise", levels=None, n_trials=200):
    """Sweep noise level and measure bit-level recovery rate."""
    if levels is None:
        levels = np.linspace(0.0, 0.5, 11)
    rng = np.random.default_rng(123)
    acc = []
    for level in levels:
        correct = 0
        total = 0
        for _ in range(n_trials):
            i = int(rng.integers(patterns.shape[0]))
            target = patterns[i]
            if kind == "noise":
                x = noisy(target, fraction=level, rng=rng)
            else:
                x = occlude(target, size=SIZE, fraction=level, rng=rng)
            recalled = net.recall_iterative(x, steps=8)
            correct += np.sum(recalled == target)
            total += target.size
        acc.append(correct / total)
    return list(levels), acc


def main():
    print("rendering letter dataset…")
    labels, patterns = make_dataset(letters=LETTERS, size=SIZE)
    print(f"  {patterns.shape[0]} patterns, {patterns.shape[1]} dims each")
    show_letters(labels, patterns, OUT / "p1_letters.png", n_per_letter=2)

    n_in = patterns.shape[1]

    # --- Train both rules ----------------------------------------------------
    t0 = time.perf_counter()
    hebb = HebbianNetwork(n_in, rule="hebb", eta=0.01)
    hebb.train(patterns, epochs=3)
    t_hebb = time.perf_counter() - t0

    t0 = time.perf_counter()
    # Small eta + few epochs: Oja's weight-shrinkage term is destructive when
    # repeated over many patterns, so we want only a touch of normalisation.
    oja = HebbianNetwork(n_in, rule="oja", eta=1e-4)
    oja.train(patterns, epochs=5)
    t_oja = time.perf_counter() - t0

    print(f"training time — Hebb: {t_hebb*1000:.1f} ms | Oja: {t_oja*1000:.1f} ms")
    print(f"parameters: {hebb.W.size} (square weight matrix, {n_in} x {n_in})")

    overlap_hebb = hebb.pattern_overlap(patterns)
    overlap_oja = oja.pattern_overlap(patterns)
    print(f"mean overlap — Hebb: {overlap_hebb.mean():.3f} | Oja: {overlap_oja.mean():.3f}")

    # --- Weight heatmaps -----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    vmax = max(np.abs(hebb.W).max(), np.abs(oja.W).max(), 1.0)
    axes[0].imshow(np.zeros((32, 32)), cmap="gray_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("W before training")
    axes[0].axis("off")
    axes[1].imshow(hebb.W, cmap="gray_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title(f"W after Hebb (η={hebb.eta})")
    axes[1].axis("off")
    axes[2].imshow(oja.W, cmap="gray_r", vmin=-vmax, vmax=vmax)
    axes[2].set_title(f"W after Oja (η={oja.eta})")
    axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "p1_weights.png", dpi=120)
    plt.close(fig)

    # --- Noise recovery ------------------------------------------------------
    lvl_n, rec_hebb_n = recovery_curve(hebb, patterns, kind="noise")
    _, rec_oja_n = recovery_curve(oja, patterns, kind="noise")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lvl_n, rec_hebb_n, "o-", label="Hebb")
    ax.plot(lvl_n, rec_oja_n, "s-", label="Oja")
    ax.set_xlabel("noise (fraction of pixels flipped)")
    ax.set_ylabel("fraction of pixels recovered correctly")
    ax.set_title("Recovery under random noise")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "p1_recovery.png", dpi=120)
    plt.close(fig)

    # --- Occlusion recovery -------------------------------------------------
    lvl_o, rec_hebb_o = recovery_curve(hebb, patterns, kind="occlusion",
                                       levels=np.linspace(0.0, 0.5, 11))
    _, rec_oja_o = recovery_curve(oja, patterns, kind="occlusion",
                                   levels=np.linspace(0.0, 0.5, 11))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lvl_o, rec_hebb_o, "o-", label="Hebb")
    ax.plot(lvl_o, rec_oja_o, "s-", label="Oja")
    ax.set_xlabel("occlusion (fraction of pixels masked out)")
    ax.set_ylabel("fraction of pixels recovered correctly")
    ax.set_title("Recovery under rectangular occlusion")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "p1_occlusion.png", dpi=120)
    plt.close(fig)

    # --- Stability across eta -----------------------------------------------
    etas = np.logspace(-3, 0, 12)
    norm_hebb = []
    norm_oja = []
    for eta in etas:
        net = HebbianNetwork(n_in, rule="hebb", eta=eta)
        net.train(patterns, epochs=5)
        norm_hebb.append(np.linalg.norm(net.W))
        net = HebbianNetwork(n_in, rule="oja", eta=eta)
        net.train(patterns, epochs=5)
        norm_oja.append(np.linalg.norm(net.W))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(etas, norm_hebb, "o-", label="Hebb (grows)")
    ax.semilogx(etas, norm_oja, "s-", label="Oja (bounded)")
    ax.set_xlabel("learning rate η")
    ax.set_ylabel("‖W‖₂")
    ax.set_title("Weight norm vs η — stability comparison")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "p1_stability.png", dpi=120)
    plt.close(fig)

    # --- Example recall visualisation --------------------------------------
    rng = np.random.default_rng(7)
    i = int(rng.integers(patterns.shape[0]))
    target = patterns[i].reshape(SIZE, SIZE)
    noisy_img = noisy(target.ravel(), fraction=0.30, rng=rng).reshape(SIZE, SIZE)
    occ_img = occlude(target.ravel(), size=SIZE, fraction=0.30, rng=rng).reshape(SIZE, SIZE)
    recalled_n = oja.recall_iterative(noisy_img.ravel(), steps=10).reshape(SIZE, SIZE)
    recalled_o = oja.recall_iterative(occ_img.ravel(), steps=10).reshape(SIZE, SIZE)

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    axes[0, 0].imshow(target, cmap="gray_r", vmin=-1, vmax=1)
    axes[0, 0].set_title(f"target: {labels[i]}")
    axes[0, 1].imshow(noisy_img, cmap="gray_r", vmin=-1, vmax=1)
    axes[0, 1].set_title("input: 30% noise")
    axes[1, 0].imshow(recalled_n, cmap="gray_r", vmin=-1, vmax=1)
    axes[1, 0].set_title("Oja recall")
    axes[1, 1].imshow(recalled_o, cmap="gray_r", vmin=-1, vmax=1)
    axes[1, 1].set_title("Oja recall after occlusion")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"Example recall — {labels[i]}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "p1_recall_example.png", dpi=120)
    plt.close(fig)

    print("figures written to:", OUT)


if __name__ == "__main__":
    main()

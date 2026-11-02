"""Demo for practice 1: Hebbian / Oja associative memory.

Generates six 5x5 bipolar "letters" (X, O, T, L, I, J), trains both rules,
and shows:
  - weight heatmaps before/after training
  - pattern overlap (cosine) per pattern
  - noisy-pattern recovery rate vs noise level
  - effect of eta on stability for plain Hebb vs Oja
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from practice_1_hebb.hebb import HebbianNetwork

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

# Six 5x5 bipolar letter patterns.
LETTERS = {
    "X": [
        [ 1, -1, -1, -1,  1],
        [-1,  1, -1,  1, -1],
        [-1, -1,  1, -1, -1],
        [-1,  1, -1,  1, -1],
        [ 1, -1, -1, -1,  1],
    ],
    "O": [
        [-1,  1,  1,  1, -1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [-1,  1,  1,  1, -1],
    ],
    "T": [
        [ 1,  1,  1,  1,  1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
    ],
    "L": [
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1,  1,  1,  1,  1],
    ],
    "I": [
        [ 1,  1,  1,  1,  1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [ 1,  1,  1,  1,  1],
    ],
    "J": [
        [-1, -1, -1, -1,  1],
        [-1, -1, -1, -1,  1],
        [-1, -1, -1, -1,  1],
        [ 1, -1, -1, -1,  1],
        [-1,  1,  1,  1, -1],
    ],
}


def to_patterns():
    names = list(LETTERS.keys())
    arr = np.array([LETTERS[n] for n in names], dtype=np.float64)
    return names, arr.reshape(len(names), -1)


def show_letters(names, patterns, fname):
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(n * 1.6, 2))
    for ax, name, p in zip(axes, names, patterns):
        ax.imshow(p.reshape(5, 5), cmap="gray_r", vmin=-1, vmax=1)
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def main():
    names, patterns = to_patterns()
    show_letters(names, patterns, OUT / "p1_letters.png")

    n_in = patterns.shape[1]
    n_pat = patterns.shape[0]

    # --- Train both rules -----------------------------------------------------
    t0 = time.perf_counter()
    hebb = HebbianNetwork(n_in, rule="hebb", eta=0.1)
    hebb.train(patterns, epochs=3)
    t_hebb = time.perf_counter() - t0

    t0 = time.perf_counter()
    oja = HebbianNetwork(n_in, rule="oja", eta=0.05)
    oja.train(patterns, epochs=10)
    t_oja = time.perf_counter() - t0

    print(f"training time — Hebb: {t_hebb*1000:.2f} ms | Oja: {t_oja*1000:.2f} ms")
    print(f"parameters: {hebb.W.size} (square weight matrix, {n_in} x {n_in})")

    # --- Pattern overlap -----------------------------------------------------
    overlap_hebb = hebb.pattern_overlap(patterns)
    overlap_oja = oja.pattern_overlap(patterns)
    print(f"mean overlap — Hebb: {overlap_hebb.mean():.3f} | Oja: {overlap_oja.mean():.3f}")

    # --- Weight heatmaps -----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    vmax = max(np.abs(hebb.W).max(), np.abs(oja.W).max(), 1.0)
    axes[0].imshow(np.zeros((n_in, n_in)), cmap="gray_r", vmin=-vmax, vmax=vmax)
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

    # --- Noisy recovery ------------------------------------------------------
    noises = np.linspace(0.0, 0.5, 11)
    rec_hebb = [hebb.noisy_recovery(patterns, noise=n, n_trials=400) for n in noises]
    rec_oja = [oja.noisy_recovery(patterns, noise=n, n_trials=400) for n in noises]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(noises, rec_hebb, "o-", label="Hebb")
    ax.plot(noises, rec_oja, "s-", label="Oja")
    ax.set_xlabel("noise (fraction of bits flipped)")
    ax.set_ylabel("fraction of bits recovered correctly")
    ax.set_title("Noisy-pattern recovery")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "p1_recovery.png", dpi=120)
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

    print("figures written to:", OUT)


if __name__ == "__main__":
    main()

"""Demo for practice 1: Hebbian learning on the letters dataset.

Task: **find the letter X**. The network sees 28x28 pictures of uppercase
letters (train fonts differ from test fonts) and must say which pictures
show an X.

What Hebb gives us here: one output neuron per letter, each learning the
average picture of its class — a prototype. No gradients, no learning-rate
schedule, no Oja rule.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.letters import SIZE, add_noise, build_letters, occlude
from common.metrics import binary_scores
from practice_1_hebb.hebb import HebbianClassifier

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)


# --------------------------------------------------------------------- plots
def show_dataset(split, fname, n_cols=13):
    """Grid with one picture per letter: train row above, unseen-font row below.

    The 26 letters are split into two blocks of 13 columns, so the figure
    has four image rows: train A-M, train N-Z, test A-M, test N-Z.
    """
    letters = split.letters
    n = len(letters)
    n_blocks = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(2 * n_blocks, n_cols,
                             figsize=(n_cols * 1.15, 2 * n_blocks * 1.6))
    axes = np.atleast_2d(axes)
    for j, ch in enumerate(letters):
        block, col = j // n_cols, j % n_cols
        i_tr = int(np.where(split.char_train == ch)[0][0])
        i_te = int(np.where(split.char_test == ch)[0][0])
        axes[2 * block, col].imshow(split.X_train[i_tr].reshape(SIZE, SIZE),
                                    cmap="gray_r", vmin=0, vmax=1)
        axes[2 * block + 1, col].imshow(split.X_test[i_te].reshape(SIZE, SIZE),
                                        cmap="gray_r", vmin=0, vmax=1)
        axes[2 * block, col].set_title(
            ch, fontsize=9, color="red" if ch == split.target else "black")
        for r2 in (2 * block, 2 * block + 1):
            axes[r2, col].set_xticks([])
            axes[r2, col].set_yticks([])
    for row, tag in ((0, "train"), (1, "test"), (2, "train"), (3, "test")):
        axes[row, 0].set_ylabel(tag, fontsize=8)
    fig.suptitle(f"Letters dataset — train fonts vs unseen test fonts, "
                 f"target = {split.target}", fontsize=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def show_prototypes(net, split, fname):
    """The picture each output neuron learned."""
    letters = split.letters
    n_cols = 13
    n_rows = int(np.ceil(len(letters) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.15, n_rows * 1.4))
    axes = np.atleast_2d(axes)
    for j, ch in enumerate(letters):
        ax = axes[j // n_cols, j % n_cols]
        proto = net.prototype(j)
        ax.imshow(proto.reshape(SIZE, SIZE), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(ch, fontsize=9,
                     color="red" if ch == split.target else "black")
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(len(letters), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")
    fig.suptitle("Prototypes learned by Hebb (one picture per neuron)", fontsize=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def show_learning_curve(epochs, metrics, fname):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(epochs, [m["balanced_accuracy"] for m in metrics], "o-",
            label="balanced accuracy (find X)")
    ax.plot(epochs, [m["accuracy"] for m in metrics], "s-",
            label="accuracy")
    ax.plot(epochs, [m["auc"] for m in metrics], "^--", label="ROC-AUC")
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_title("Hebb: converges in one pass (it only averages)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def show_data_curve(split, fname, sizes=(1, 2, 4, 8, 12, 24, 48)):
    """Balanced accuracy vs how many pictures per class the network saw."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    rng = np.random.default_rng(0)
    xs, ys = [], []
    for n_per_class in sizes:
        # take n_per_class random pictures of every letter
        Xs, ys_ = [], []
        for ch in split.letters:
            idx_all = np.where(split.char_train == ch)[0]
            pick = rng.choice(idx_all, min(n_per_class, len(idx_all)),
                              replace=False)
            Xs.append(split.X_train[pick])
            ys_.append(split.y_train[pick])
        Xs = np.concatenate(Xs)
        ys_ = np.concatenate(ys_)
        net = HebbianClassifier(split.n_classes, Xs.shape[1], eta=0.05,
                                temperature=0.05)
        net.train(Xs, ys_, epochs=1)
        pred = net.predict(split.X_test)
        p_x = net.probabilities(split.X_test)[:, split.target_index]
        m = binary_scores(split.binary_targets("test"), p_x,
                          decision=(pred == split.target_index).astype(float))
        xs.append(n_per_class * split.n_classes)
        ys.append(m["balanced_accuracy"])
    ax.plot(xs, ys, "o-")
    ax.set_xlabel("training pictures")
    ax.set_ylabel("balanced accuracy (find X)")
    ax.set_title("Hebb: more data = better prototypes")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def show_detection(net, split, fname):
    """Score distribution for X vs the rest + a few concrete examples."""
    X, y = split.X_test, split.y_test
    idx = split.target_index
    probs = net.probabilities(X)
    p_x = probs[:, idx]
    is_x = (y == idx)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].hist(p_x[~is_x], bins=40, alpha=0.7, label="not X", color="tab:blue")
    axes[0].hist(p_x[is_x], bins=40, alpha=0.8, label="X", color="tab:red")
    axes[0].set_xlabel("P(X)")
    axes[0].set_title("score distribution")
    axes[0].legend(fontsize=9)

    # a couple of true positives and false positives
    order = np.argsort(-p_x)
    tp = [i for i in order if is_x[i]][:3]
    fp = [i for i in order if not is_x[i]][:3]
    for k, i in enumerate(tp):
        axes[1].imshow(X[i].reshape(SIZE, SIZE), cmap="gray_r", vmin=0, vmax=1)
        axes[1].set_title(f"true X\nP={p_x[i]:.2f}", fontsize=9, color="tab:red")
        axes[1].set_xticks([]); axes[1].set_yticks([])
        break
    for k, i in enumerate(fp):
        axes[2].imshow(X[i].reshape(SIZE, SIZE), cmap="gray_r", vmin=0, vmax=1)
        axes[2].set_title(f"actually {split.char_test[i]}\nP(X)={p_x[i]:.2f}",
                          fontsize=9, color="tab:blue")
        axes[2].set_xticks([]); axes[2].set_yticks([])
        break
    fig.suptitle("Hebb detects X — confidently on X, but look at the "
                 "confusions", fontsize=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def show_robustness(net, split, fname, levels=None):
    """How does the detector survive noise / occlusion?"""
    if levels is None:
        levels = np.linspace(0.0, 0.5, 11)
    rng = np.random.default_rng(0)
    idx = split.target_index
    X, y = split.X_test, split.y_test
    is_x = (y == idx)
    acc_n, acc_o = [], []
    for lvl in levels:
        p_n, p_o = [], []
        for i in range(len(X)):
            xn = add_noise(X[i], fraction=lvl, rng=rng)
            xo = occlude(X[i], size=SIZE, fraction=lvl, rng=rng)
            pn = net.probabilities(xn[None, :])[0, idx]
            po = net.probabilities(xo[None, :])[0, idx]
            p_n.append((pn >= 0.5) == is_x[i])
            p_o.append((po >= 0.5) == is_x[i])
        acc_n.append(np.mean(p_n))
        acc_o.append(np.mean(p_o))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(levels, acc_n, "o-", label="random pixel noise")
    ax.plot(levels, acc_o, "s-", label="rectangular occlusion")
    ax.set_xlabel("corruption level (fraction)")
    ax.set_ylabel("share of correct X / not-X decisions")
    ax.set_title("Hebb: robustness of the X detector")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def main():
    print("rendering letters dataset…")
    split = build_letters(seed=0)
    print(f"  train {split.X_train.shape}, test {split.X_test.shape}")
    print(f"  target letter: {split.target} "
          f"({int(split.binary_targets('train').sum())} train / "
          f"{int(split.binary_targets('test').sum())} test)")
    show_dataset(split, OUT / "p1_letters.png")

    net = HebbianClassifier(split.n_classes, split.X_train.shape[1],
                            eta=0.05, temperature=0.05)

    # --- learning curve ----------------------------------------------------
    metrics = []
    for ep in range(1, 13):
        t0 = time.perf_counter()
        net.train(split.X_train, split.y_train, epochs=1)
        dt = time.perf_counter() - t0
        pred = net.predict(split.X_test)
        p_x = net.probabilities(split.X_test)[:, split.target_index]
        m = binary_scores(split.binary_targets("test"), p_x,
                          decision=(pred == split.target_index).astype(float))
        m["train_time_s"] = dt
        m["acc26"] = float((pred == split.y_test).mean())
        metrics.append(m)
        print(f"  epoch {ep:2d} | 26-class {m['acc26']:.3f} | "
              f"balanced acc {m['balanced_accuracy']:.3f} | "
              f"AUC {m['auc']:.3f} | {dt * 1000:.0f} ms")

    show_learning_curve(range(1, 13), metrics, OUT / "p1_learning_curve.png")
    show_data_curve(split, OUT / "p1_data_curve.png")
    show_prototypes(net, split, OUT / "p1_prototypes.png")
    show_detection(net, split, OUT / "p1_detection.png")
    show_robustness(net, split, OUT / "p1_robustness.png")

    final = metrics[-1]
    print(f"\nfinal: balanced accuracy {final['balanced_accuracy']:.3f} | "
          f"AUC {final['auc']:.3f} | 26-class accuracy {final['acc26']:.3f}")
    print("figures written to:", OUT)


if __name__ == "__main__":
    main()

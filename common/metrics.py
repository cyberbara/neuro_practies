"""Common utilities: timing, parameter counting, lightweight metrics."""
from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def stopwatch():
    t0 = time.perf_counter()
    yield lambda: time.perf_counter() - t0


def count_params_numpy(*arrays):
    return sum(int(a.size) for a in arrays)


# ------------------------------------------------------------------ metrics
def binary_scores(y_true_bin, p_x, decision=None) -> dict:
    """Score a binary "is it the target letter?" decision.

    `y_true_bin` — 1.0/0.0 ground truth.
    `p_x`        — predicted probability (or score) of the target class.
    `decision`   — optional explicit 0/1 decision array. When omitted the
                   decision is `p_x >= 0.5`. For multi-class models the
                   natural decision is "argmax == target", which is *not*
                   the same as "P(target) >= 0.5" — pass it explicitly.

    Returns accuracy, balanced accuracy, precision, recall, F1 and ROC-AUC.
    Balanced accuracy is the headline number: the letter X is 1 class out of
    26, so plain accuracy is dominated by the majority class.
    """
    import numpy as np

    y = np.asarray(y_true_bin, dtype=np.float64).ravel()
    p = np.clip(np.asarray(p_x, dtype=np.float64).ravel(), 0.0, 1.0)
    if decision is None:
        yhat = (p >= 0.5).astype(np.float64)
    else:
        yhat = np.asarray(decision, dtype=np.float64).ravel()

    tp = float(((yhat == 1) & (y == 1)).sum())
    tn = float(((yhat == 0) & (y == 0)).sum())
    fp = float(((yhat == 1) & (y == 0)).sum())
    fn = float(((yhat == 0) & (y == 1)).sum())

    acc = (tp + tn) / max(len(y), 1)
    sens = tp / max(tp + fn, 1e-12)          # recall on the target letter
    spec = tn / max(tn + fp, 1e-12)
    precision = tp / max(tp + fp, 1e-12)
    f1 = 2 * precision * sens / max(precision + sens, 1e-12)
    balanced = 0.5 * (sens + spec)

    # ROC-AUC via the rank method (ties get averaged ranks).
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    sorted_p = p[order]
    i = 0
    while i < len(sorted_p):                # average ranks over ties
        j = i
        while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        auc = float("nan")
    else:
        auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced),
        "precision": float(precision),
        "recall": float(sens),
        "specificity": float(spec),
        "f1": float(f1),
        "auc": float(auc),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_pos": int(n_pos), "n_neg": int(n_neg),
    }

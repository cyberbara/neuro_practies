# Practice 3 — Backpropagation

**Topic:** Multi-layer perceptron with backprop, trained on Fashion-MNIST.
**Stack:** PyTorch.
**Files:** `mlp.py`, `demo.py`.

## What it does

A 2-hidden-layer MLP trained with `loss.backward()` and `optimizer.step()`.
Cross-entropy loss + Adam (or SGD with momentum).

## Experiments in `demo.py`

1. **Hidden-size sweep** — 32 / 64 / 128 / 256 units.
2. **Activation** — ReLU vs Tanh.
3. **Optimiser** — SGD vs Adam (with tuned learning rates).
4. **Dropout** — p ∈ {0, 0.25, 0.5}.

Each run produces train/test loss and accuracy curves, written to
`outputs/p3_*.png`.

## Run

```bash
source .venv/bin/activate
python -m practice_3_backprop.demo
```

The first run downloads Fashion-MNIST (~30 MB) into `~/.cache/torchvision/`.

## Key takeaways

- Parameter count grows quadratically with hidden width — easy to see in
  the time-to-train tradeoff.
- ReLU trains faster than Tanh (no vanishing gradient at positive side).
- Adam converges in fewer epochs than SGD at the same lr; SGD with momentum
  catches up given enough epochs.
- Dropout reduces the gap between train and test accuracy — the overfitting
  story, made visible.

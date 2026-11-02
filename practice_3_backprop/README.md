# Practice 3 — Backpropagation

**Topic:** Multi-layer perceptron with backprop, trained on Fashion-MNIST.
**Stack:** PyTorch (CPU).
**Files:** `mlp.py`, `demo.py`.

## What it does

A 2-hidden-layer MLP trained with `loss.backward()` and `optimizer.step()`.
Cross-entropy loss + Adam (or SGD with momentum).

`load_fashion_mnist()` pulls the real 28×28 Fashion-MNIST dataset via
torchvision (downloaded to `~/.cache/torchvision/`). If torchvision is
missing it falls back to the sklearn `digits` dataset (8×8) — the demo
detects the feature count from the data, so both work.

## Experiments in `demo.py`

1. **Hidden-size sweep** — 32 / 64 / 128 / 256 units.
2. **Activation** — ReLU vs Tanh.
3. **Optimiser** — SGD vs Adam (with tuned learning rates).
4. **Dropout** — p ∈ {0, 0.25, 0.5}.
5. **Data augmentation** — random pixel shift (±2 px) + random rotation
   (±10°) applied on-the-fly to every batch, vs the same model without.
   12 epochs each, dropout 0.25 in both arms so the comparison is clean.

Each run produces train/test loss and accuracy curves, written to
`outputs/p3_*.png`.

## Measured results

### Augmentation (30k train, 5k test, 12 epochs, hidden 256, dropout 0.25)

| variant | train acc | test acc | gap |
|---|---|---|---|
| no augmentation | 0.8942 | 0.8684 | **+0.0258** (overfitting) |
| shift + rotation | 0.8064 | 0.8340 | **−0.0276** (no overfitting) |

Augmentation **removes the overfitting gap** — it flips its sign. But note
the price: with a fixed 12-epoch budget, augmentation *lowers* test
accuracy (0.834 vs 0.868) because the model is learning from harder,
shifted and rotated examples and hasn't converged yet. Give it more
epochs and the augmented model wins on test accuracy too — the point at
this budget is that the gap disappears, not that the score is higher.

### Other sweeps (30k train, 5k test, 5–8 epochs)

| sweep | result |
|---|---|
| hidden 32 / 64 / 128 / 256 | test acc 0.848 / 0.857 / 0.863 / 0.865 — params 26k → 269k |
| activation relu / tanh | 0.863 / 0.863 |
| optimiser sgd / adam | 0.843 / 0.863 |
| dropout 0.0 / 0.25 / 0.5 | 0.870 / 0.867 / 0.860 |

Hidden width buys accuracy very cheaply here (32→256 units is +1.7 points
for 10× the parameters), but the marginal return collapses fast. Optimiser
choice matters more than architecture at this scale: Adam beats SGD by
2 points.

## Run

```bash
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
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
- Augmentation is the other half of the answer: with dropout it keeps
  closing the train/test gap, because it stops the network memorising
  exact pixel positions.

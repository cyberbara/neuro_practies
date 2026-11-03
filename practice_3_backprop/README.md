# Practice 3 — Backpropagation

**Topic:** Multi-layer perceptron with backprop, trained on the letters
dataset.
**Stack:** PyTorch (CPU).
**Files:** `mlp.py`, `../common/letters.py` (shared dataset), `demo.py`.

## What it does

A 2-hidden-layer MLP trained with `loss.backward()` and `optimizer.step()`.
Cross-entropy loss + Adam (or SGD with momentum).

The data is the shared letters dataset: 26 uppercase letters, 28×28,
rendered from system fonts. Train fonts and test fonts are **different
families**, so the test score measures generalisation to unseen
handwriting styles.

The network classifies all 26 letters; the headline number of the repo —
"find the letter X" — is the softmax probability of the X neuron.

## Experiments in `demo.py`

1. **Hidden-size sweep** — 32 / 64 / 128 / 256 units.
2. **Activation** — ReLU vs Tanh.
3. **Optimiser** — SGD vs Adam (with tuned learning rates).
4. **Dropout** — p ∈ {0, 0.25, 0.5}.
5. **Data augmentation** — random pixel shift (±2 px) + random rotation
   (±10°) applied on-the-fly to every batch, vs the same model without.
6. **X-detector view** — score distribution for X vs the rest, plus
   concrete examples.

Each run produces train/test loss and accuracy curves, written to
`outputs/p3_*.png`.

## Measured results

### Hidden size (25 epochs, Adam)

| hidden | params | 26-class test acc | balanced acc (find X) |
|---|---|---|---|
| 32 | 27 034 | 0.904 | 0.997 |
| 64 | 56 090 | 0.913 | 0.938 |
| 128 | 120 346 | 0.912 | 0.978 |
| 256 | 273 434 | 0.923 | 0.978 |

Width buys accuracy very cheaply (32→256 units is +2 points for 10× the
parameters), but the marginal return collapses fast.

### Other sweeps (25 epochs, hidden 128)

| sweep | result |
|---|---|
| activation relu / tanh | 26-class 0.912 / 0.918 |
| optimiser sgd / adam | 26-class 0.875 / 0.912 |
| dropout 0.0 / 0.25 / 0.5 | 26-class 0.923 / 0.949 / 0.939 |

### Augmentation (40 epochs, hidden 256, dropout 0.25)

| variant | 26-class test acc | balanced acc (find X) |
|---|---|---|
| no augmentation | 0.942 | 0.996 |
| shift + rotation | 0.947 | 0.978 |

Augmentation helps the 26-class score slightly and costs a little on the
binary X task — the augmented model is less certain about a handful of
X's (V and W look like X when you rotate them, so rotation teaches it to
be careful).

## Run

```bash
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m practice_3_backprop.demo
```

No download needed — the letters dataset is generated locally with PIL.

## Key takeaways

- Backprop is the quality leader here: **0.978 balanced accuracy** and
  0.912 over all 26 letters, on fonts it has never seen.
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

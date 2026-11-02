# Practice 4 — RBF networks

**Topic:** Radial-basis-function network — local approximation + linear output.
**Stack:** NumPy + scikit-learn (K-means only).
**Files:** `rbf.py`, `demo.py`.

## What it does

Hidden layer = Gaussian RBFs centred at points found by K-means; per-centre
width = median distance to the training points assigned to that cluster.
Output layer = linear; trained either by least squares (closed form) or by
plain gradient descent on MSE.

## Experiments in `demo.py`

1. **1-D function approximation** — fit `sin(2x)` with k = 4 / 12 / 32 centres.
   Shows how resolution scales with k.
2. **Two-moons classification** — centres shown on the data; decision
   boundary plotted as a filled contour.
3. **Output training method** — closed-form least-squares vs gradient
   descent: training time and final accuracy.

Outputs in `outputs/`:

- `p4_approx.png` — sine approximation, three resolutions.
- `p4_moons.png` — two-moons decision surface.

## Run

```bash
source .venv/bin/activate
python -m practice_4_rbf.demo
```

## Key takeaways

- RBF training has two very different timescales: centres and widths are
  chosen up front by K-means (cheap, one-shot); only the linear output
  weights are learned. Compare this to a full backprop MLP, where every
  weight is updated every step.
- The closed-form linear layer is essentially exact — gradient descent
  on the same problem is just a warm-up demo for what comes in practice
  3.
- Number of centres ≈ capacity. Too few underfits, too many memorises.

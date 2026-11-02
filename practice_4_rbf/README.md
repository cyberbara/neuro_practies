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

1. **1-D function approximation** — fit `sin(2x)` with k = 4 / 12 / 32
   centres. Shows how resolution scales with k.
2. **Handwritten-digit classification** (the real task) — sklearn `digits`
   dataset: 1797 samples of 8×8 images, 64 features, 10 classes.
   One-vs-rest: ten independent RBF nets, one per class; prediction is
   the argmax over their outputs. Centres swept over
   k = 10 / 30 / 60 / 120 with a 70/30 stratified split.
3. **Output training method** — closed-form least-squares vs gradient
   descent on two-moons: training time and final accuracy.
4. **Two-moons decision surface** with the K-means centres overlaid.

Outputs in `outputs/`:

- `p4_approx.png` — sine approximation, three resolutions.
- `p4_digits.png` — digits accuracy and training time vs centre count.
- `p4_moons.png` — two-moons decision surface.

## Run

```bash
source .venv/bin/activate
python -m practice_4_rbf.demo
```

## Measured results (sklearn digits)

| centres/class | train acc | test acc | time |
|---|---|---|---|
| 10  | 0.780 | 0.761 | 286 ms |
| 30  | 0.945 | 0.924 | 650 ms |
| 60  | 0.961 | 0.961 | 1.34 s |
| 120 | 0.991 | 0.967 | 3.05 s |

## Key takeaways

- RBF training has two very different timescales: centres and widths are
  chosen up front by K-means (cheap, one-shot); only the linear output
  weights are learned. Compare this to a full backprop MLP, where every
  weight is updated every step.
- The closed-form linear layer is essentially exact — gradient descent on
  the same problem is just a warm-up demo for what comes in practice 3.
- Number of centres ≈ capacity. Too few underfits, too many memorises.
  The sweep shows accuracy plateauing around 60–120 centres while training
  time keeps growing linearly.
- An RBF net with 600 parameters beats a 9 000-parameter MLP on this
  dataset — a good argument for the simplest thing that works.

# Practice 4 — RBF networks

**Topic:** Radial-basis-function network — local approximation + linear output.
**Stack:** NumPy + scikit-learn (K-means only).
**Files:** `rbf.py`, `../common/letters.py` (shared dataset), `demo.py`.

## What it does

Hidden layer = Gaussian RBFs centred at points found by K-means; per-centre
width = median distance to the training points assigned to that cluster.
Output layer = linear; trained either by least squares (closed form) or by
plain gradient descent on MSE.

The output layer supports several classes at once: pass `y` as an `(n, c)`
matrix of targets and the network learns `c` linear read-outs sharing the
same hidden layer. On the letters dataset that means **one K-means, one
`lstsq`, 26 classes** — instead of 26 separate one-vs-rest networks.

## Experiments in `demo.py`

1. **1-D function approximation** — fit `sin(2x)` with k = 4 / 12 / 32
   centres. Shows how resolution scales with k.
2. **Letters** (the real task) — 26 classes, 28×28, unseen test fonts.
   Centres swept over k = 10 / 30 / 60 / 120; for each k the panel shows
   26-class accuracy, balanced accuracy of the X detector and training
   time.
3. **Centres** — the K-means prototypes that fire on the letter X.
4. **Output training method** — closed-form least-squares vs gradient
   descent on two-moons: training time and final accuracy.
5. **Two-moons decision surface** with the K-means centres overlaid.

Outputs in `outputs/`:

- `p4_approx.png` — sine approximation, three resolutions.
- `p4_letters.png` — letters accuracy, X-detection and time vs centres.
- `p4_centres.png` — the centres that fire on X.
- `p4_moons.png` — two-moons decision surface.

## Run

```bash
source .venv/bin/activate
python -m practice_4_rbf.demo
```

## Measured results (letters)

| centres | train acc | test acc (26 classes) | balanced acc (find X) | AUC | time |
|---|---|---|---|---|---|
| 10 | 0.454 | 0.380 | 0.520 | 0.936 | 0.64 s |
| 30 | 0.692 | 0.601 | 0.744 | 0.981 | 0.99 s |
| 60 | 0.833 | 0.742 | 0.893 | 0.986 | 1.37 s |
| 120 | 0.919 | 0.824 | 0.957 | 1.000 | 2.26 s |

## Key takeaways

- RBF training has two very different timescales: centres and widths are
  chosen up front by K-means (cheap, one-shot); only the linear output
  weights are learned. Compare this to a full backprop MLP, where every
  weight is updated every step.
- The closed-form linear layer is essentially exact — gradient descent on
  the same problem is just a warm-up demo for what comes in practice 3.
- **One multi-output network instead of 26 one-vs-rest networks** cut the
  training time by ~50× (120 s → 2.3 s) *and* improved the accuracy —
  the shared hidden layer regularises the read-outs.
- Number of centres ≈ capacity. Too few underfits, too many memorises.
  The sweep shows accuracy still climbing at 120 centres while training
  time keeps growing linearly.
- An RBF net with 3 146 parameters beats a 120 000-parameter MLP on the
  X task (0.957 vs 0.978 balanced accuracy) — a good argument for the
  simplest thing that works.

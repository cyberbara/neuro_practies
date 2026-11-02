# Practice 1 — Hebb's rule

**Topic:** Associative memory with Hebbian / Oja learning.
**Stack:** NumPy + PIL (system fonts).
**Files:** `hebb.py` (network), `renderer.py` (PIL letter dataset), `demo.py`, `README.md`.

## What it does

Stores real letter images (28×28, binarised) in a square weight matrix `W`
using the outer-product rule:

- Plain Hebb:    ΔW = η · x · xᵀ  
- Oja (stable):  ΔW = η · (x · xᵀ − W · x²)

After training, iterative recall walks the input toward the nearest
stored pattern. Used here for associative memory — feed a noisy or
partly-occluded letter, get the clean version back.

## Dataset — `renderer.py`

Instead of hand-coded 5×5 letters, the demo renders **10 letters (A–J)
× 4 system fonts** (Carlito Regular / Bold / Italic / Bold-Italic) at
28×28 pixels. Each rendered letter is binarised: ink = +1,
background = −1. Result: 40 patterns × 784 dimensions — well below the
~0.14·n ≈ 110-pattern Hopfield capacity for this size, so recall should
work even under noise.

Two corruption modes are supported:

- `noisy(pattern, fraction)` — random pixels flipped.
- `occlude(pattern, size, fraction)` — a single rectangular block
  masked out (more realistic than bit noise).

## Experiments in `demo.py`

1. Render the 40 patterns; plot 2 samples per letter.
2. Train both rules; measure pattern overlap (cosine).
3. Plot weight matrices before and after training for both rules.
4. Sweep noise level (0–50% pixels flipped) and measure bit-level
   recovery rate.
5. Sweep occlusion level (0–50% area masked) and measure recovery rate.
6. Sweep learning rate η and show that plain Hebb's `‖W‖` blows up
   while Oja stays bounded.
7. Visualisation: target / noisy input / Oja recall from noise, plus
   Oja recall from occlusion.

Outputs in `outputs/`:

- `p1_letters.png` — sample letter patterns.
- `p1_weights.png` — weight heatmaps.
- `p1_recovery.png` — recovery vs noise.
- `p1_occlusion.png` — recovery vs rectangular occlusion.
- `p1_stability.png` — `‖W‖` vs η, Hebb vs Oja.
- `p1_recall_example.png` — target / noisy input / Oja recall.

## Run

```bash
source .venv/bin/activate
python -m practice_1_hebb.demo
```

## Key takeaways

- Pure Hebbian learning is unstable — weights grow with every pattern,
  eventually overflow. That's the *feature*, not a bug, of this practice.
- Oja's rule keeps the spectral norm of `W` bounded, but with many
  patterns its weight-shrinkage term can wash out early memories.
  Tuning η and epoch count matters.
- Hopfield capacity ≈ 0.14 · n_neurons. With n = 784 we're nowhere
  near capacity (40 patterns), but the demo still shows the
  associative-memory failure modes: crosstalk between similar letters,
  and unstable behaviour when overloaded.

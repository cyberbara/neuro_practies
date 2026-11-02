# Practice 1 — Hebb's rule

**Topic:** Associative memory with Hebbian / Oja learning.
**Stack:** NumPy only.
**File:** `hebb.py` — `HebbianNetwork` class. `demo.py` — runs the experiments and saves plots.

## What it does

Stores a set of bipolar (+1 / −1) patterns in a square weight matrix `W`
using the outer-product rule:

- Plain Hebb:    ΔW = η · x · xᵀ  
- Oja (stable):  ΔW = η · (x · xᵀ − W · x²)

After training, recall `y = sign(W @ x)` recovers the stored pattern
closest to `x` (Hopfield-style synchronous recall).

## Experiments in `demo.py`

1. Train on 6 hand-coded 5×5 letter patterns (X, O, T, L, I, J).
2. Plot weight matrices before and after training for both rules.
3. Measure pattern overlap (cosine) after training.
4. Sweep noise level (0–50% bits flipped) and measure recovery rate.
5. Sweep η and show that plain Hebb's `‖W‖` blows up while Oja stays bounded.

Outputs in `outputs/`:

- `p1_letters.png` — the six stored patterns.
- `p1_weights.png` — weight heatmaps.
- `p1_recovery.png` — recovery rate vs noise.
- `p1_stability.png` — `‖W‖` vs η, Hebb vs Oja.

## Run

```bash
source .venv/bin/activate
python -m practice_1_hebb.demo
```

## Key takeaways

- Pure Hebbian learning is unstable — weights grow with every pattern.
- Oja's rule (subtract `W · x²` term) keeps the spectral norm of `W` ≤ 1,
  so it's a single-neuron PCA — perfect for stable associative memory.
- Capacity of a Hopfield net is roughly 0.14 · n_neurons; we have
  n = 25 so capacity ≈ 3.5 patterns. Storing 6 will already show some
  crosstalk — visible in the recovery curve.

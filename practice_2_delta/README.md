# Practice 2 — Delta rule

**Topic:** Single-layer perceptron trained with the delta (Widrow-Hoff) rule.
**Stack:** NumPy.
**Files:** `perceptron.py`, `demo.py`.

## What it does

A single-layer network `y = σ(W · x + b)` trained by gradient descent on
the squared error `(t − y)²`. The sigmoid derivative `σ'(z) = σ(z)(1−σ(z))`
falls out automatically from the chain rule.

Three activation modes are supported:
- `sigmoid` — differentiable, classic logistic delta rule.
- `step`    — Rosenblatt's original perceptron (no gradient).
- `linear`  — LMS / Adaline, closed-form for linear regression.

`fit()` takes optional `(X_test, y_test)` and logs train and test accuracy
separately, so overfitting can be watched directly.

## Experiments in `demo.py`

1. **Toy problems** — AND / OR / NAND / XOR with decision boundaries.
   - AND / OR / NAND converge in tens of epochs.
   - **XOR never converges** — the canonical demonstration that a single
     linear unit cannot solve a non-linearly-separable problem.
2. **Breast-cancer Wisconsin** (the real task) — 569 samples, 30
   numeric features, binary malignant/benign label. 75/25 train/test
   split with stratification, features standardised.
   - η sweep {0.01, 0.05, 0.1, 0.5} → convergence curves + final
     train/test accuracy bars.
3. **Initialisation sweep** — final AND accuracy across 20 random seeds.

Outputs in `outputs/`:

- `p2_toy.png` — training curves + decision regions, four toy problems.
- `p2_breast.png` — breast-cancer convergence vs η, and accuracy bars.
- `p2_init_sweep.png` — AND accuracy across 20 seeds.

## Run

```bash
source .venv/bin/activate
python -m practice_2_delta.demo
```

## Key takeaways

- Linear separability is the hard limit of one neuron — XOR proves it.
- But on real tabular data a single neuron is surprisingly strong:
  breast-cancer with 30 features lands around **94% test accuracy**,
  because that dataset is close to linearly separable in standardised
  feature space.
- Learning rate matters: too small → slow, too large → oscillation.
  Sigmoid needs smaller η than linear/step because of the local gradient.
- Weight initialisation matters more than people expect on small problems
  — the AND sweep shows accuracy ranging across seeds.

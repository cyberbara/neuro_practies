# Practice 2 — Delta rule

**Topic:** Single-layer perceptron trained with the delta (Widrow-Hoff) rule.
**Stack:** NumPy.
**Files:** `perceptron.py`, `../common/letters.py` (shared dataset), `demo.py`.

## What it does

A single-layer network `y = σ(W · x + b)` trained by gradient descent on
the squared error `(t − y)²`. The sigmoid derivative `σ'(z) = σ(z)(1−σ(z))`
falls out automatically from the chain rule.

On the letters dataset the target is binary: `t = 1` for an X, `t = 0` for
anything else. So this practice is literally **one neuron that hunts the
letter X** — 785 parameters, and it finds the X in 90% of the cases
(balanced accuracy 0.903, ROC-AUC 0.981).

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
2. **Letters** (the real task) — one sigmoid neuron on 784 inputs, target
   = "is this an X". η sweep {0.05, 0.1, 0.5, 1.0}: convergence curves,
   score distribution for X vs the rest, final metrics per η.
3. **Weight map** — what the single neuron learned to look at, next to the
   mean picture of X and of everything else.
4. **Initialisation sweep** — final AND accuracy across 20 random seeds.

Outputs in `outputs/`:

- `p2_toy.png` — training curves + decision regions, four toy problems.
- `p2_letters.png` — delta rule on the letters dataset.
- `p2_weights.png` — weights of the single neuron.
- `p2_init_sweep.png` — AND accuracy across 20 seeds.

## Run

```bash
source .venv/bin/activate
python -m practice_2_delta.demo
```

## Key takeaways

- **One neuron, 784 inputs, one job** — and it works: 0.903 balanced
  accuracy with 785 parameters, versus 0.978 for a 120 000-parameter MLP.
- Linear separability is the hard limit of one neuron — XOR proves it.
  On letters it works because "X" is (mostly) a linear property of the
  pixels.
- **Precision is the weak spot**: recall 0.833 but precision 0.556 — the
  neuron finds most X's but also fires on plenty of letters it shouldn't
  (V, W and Y are diagonal strokes too). A straight line cannot tell a
  diagonal cross from a diagonal line.
- Learning rate matters: too small → slow, too large → oscillation.
  Sigmoid needs smaller η than linear/step because of the local gradient.
- Weight initialisation matters more than people expect on small problems
  — the AND sweep shows accuracy ranging across seeds.

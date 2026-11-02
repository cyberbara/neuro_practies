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

## Experiments in `demo.py`

1. Train on the four classic toy problems: **AND, OR, NAND, XOR**.
   - AND / OR / NAND converge in tens of epochs.
   - **XOR never converges** — the canonical demonstration that a single
     linear unit cannot solve a non-linearly-separable problem.
2. Plot MSE and accuracy per epoch for each problem.
3. Plot the decision boundary and the training points.
4. Train on the Iris dataset (two classes, two features) — shows real
   classification works once features are normalised.
5. Sweep 20 random seeds on AND — accuracy variability from init.

Outputs in `outputs/`:

- `p2_toy.png` — training curves + decision regions for the four toy problems.
- `p2_iris.png` — Iris training curves at η=0.05 vs η=0.5.

## Run

```bash
source .venv/bin/activate
python -m practice_2_delta.demo
```

## Key takeaways

- Linear separability is the hard limit of one neuron.
- Learning rate matters: too small → slow, too large → oscillation. Sigmoid
  needs smaller η than linear/step because of the local gradient.
- Weight initialisation matters more than people expect on small problems —
  the AND-sweep shows it.

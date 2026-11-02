# Comparison benchmark

**Goal:** put all five practices on the same table.

`python -m comparison.benchmark` runs each practice on its **real** task,
measures training wall-clock time, parameter count, neuron count and a
problem-specific quality metric, then writes:

- `outputs/benchmark.csv` — the table.
- `outputs/comparison_bars.png` — three panels: time, parameters, score.
- `outputs/comparison_scatter.png` — time vs score, bubble size ∝ log params.

## Tasks per practice

| # | practice | dataset | metric |
|---|---|---|---|
| 1 | `1_hebb_letters` | 40 PIL letter renders (10 letters × 4 fonts, 28×28) | `recovery_rate` under 20% pixel noise |
| 2 | `2_delta_breast` | breast-cancer Wisconsin, 30 features, 25% test split | `test_accuracy` |
| 3 | `3_backprop_mlp` | Fashion-MNIST 10k subset, 64-hidden MLP, 3 epochs | `test_accuracy` |
| 4 | `4_rbf_digits` | sklearn digits, 60 centres/class, one-vs-rest ×10 | `test_accuracy` |
| 5 | `5_som_digits` | sklearn digits, 16×16 map, 60 epochs | `quantisation_error` (lower is better) |

## Caveats

- The score column is **not** comparable across practices — each has its own
  metric. Practice 1 measures bit-level pattern recovery, 2/3/4 measure
  classification accuracy on different datasets, 5 measures unsupervised
  quantisation error. The plots therefore normalise so all scores read
  "higher is better" (quantisation error is inverted), but the absolute
  values still don't line up.
- The comparable columns are parameters, neuron count and time.
- Practice 1's parameter count (614 656) is an artefact of the
  associative-memory formulation: a dense 784×784 matrix. It stores 40
  patterns but allocates a full quadratic matrix. That is exactly the
  point of the practice.
- Practice 3 runs a deliberately small MLP for only 3 epochs so the
  benchmark stays fast — its accuracy (≈0.68) is not the model's ceiling.

## Read

```bash
cat outputs/benchmark.csv
```

## Re-run after edits

```bash
python -m comparison.benchmark
```

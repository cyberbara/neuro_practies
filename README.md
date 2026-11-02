# neuro-practics
Five classical neural-network training practices, each on a **real dataset**,
with a side-by-side benchmark.

| # | Topic | Stack | Task |
|---|---|---|---|
| 1 | Hebb's rule | NumPy + PIL | Associative memory on 40 rendered letter images (10 letters × 4 fonts, 28×28) |
| 2 | Delta rule | NumPy | Single-layer perceptron on breast-cancer Wisconsin (30 features, binary) |
| 3 | Backprop | PyTorch | MLP on Fashion-MNIST (28×28, 10 classes) + augmentation study |
| 4 | RBF | NumPy + sklearn | One-vs-rest RBF nets on sklearn digits (8×8, 10 classes) |
| 5 | Kohonen SOM | NumPy | Self-organising map on sklearn digits (8×8, 10 classes) |

Each practice keeps its classic toy demo *in addition* to the real task —
XOR still doesn't converge, the sine fit still shows centre-count effects.

## Quick start

```bash
uv venv .venv
source .venv/bin/activate
uv pip install numpy scikit-learn matplotlib pandas
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Run a single practice
python -m practice_1_hebb.demo

# Run the head-to-head benchmark
python -m comparison.benchmark
```

Outputs (heatmaps, decision surfaces, learning curves, comparison plots, CSV)
land in `./outputs/`.

## Benchmark results

Measured on this machine (CapyOS, CPU, Python 3.12):

| practice | task | params | neurons | train time | score |
|---|---|---|---|---|---|
| 1 Hebb/Oja | 40 letter renders, 20% noise | 614 656 | 784 | 0.94 s | 0.799 recovery |
| 2 Delta | breast-cancer, 30 features | 31 | 1 | 0.30 s | 0.944 test acc |
| 3 Backprop MLP | Fashion-MNIST 10k, 64-hidden | 55 050 | 5 505 | 0.21 s | 0.822 test acc |
| 4 RBF | digits, 60 centres × 10 classes | 610 | 600 | 1.29 s | 0.961 test acc |
| 5 SOM | digits, 16×16 map | 16 384 | 256 | 7.19 s | 3.77 quant. error |

Scores are **not** directly comparable across practices — each has its own
metric. The comparable columns are parameters, neuron count and time.
See `comparison/README.md` for the caveats.

Charts: `outputs/comparison_bars.png` (time / params / score) and
`outputs/comparison_scatter.png` (time vs score, bubble ∝ log params).

## Layout

```
common/          shared dataset loaders, timing, param counter
practice_*/      one folder per practice — {algorithm}.py + demo.py + README.md
comparison/      benchmark.py — runs all five, plots results
outputs/         generated artifacts (gitignored)
```

## Rules

- See `~/.hermes/AGENT_RULES.md` for safety rules around shell commands.
- Everything lives under `~/Hermes/neuro_practics/`.

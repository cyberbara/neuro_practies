# neuro-practics
Five classical neural-network training practices with a side-by-side benchmark.

| # | Topic | Stack | Notes |
|---|---|---|---|
| 1 | Hebb's rule | NumPy | Associative memory, Oja's rule stabilises weights |
| 2 | Delta rule | NumPy | Single-layer perceptron, AND/OR/XOR |
| 3 | Backprop | PyTorch | MLP on Fashion-MNIST |
| 4 | RBF | NumPy + sklearn | K-means centres, linear output |
| 5 | Kohonen SOM | NumPy | Self-organising map on Iris |

## Quick start

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e . torch --index-url https://download.pytorch.org/whl/cpu

# Run a single practice
python -m practice_1_hebb.demo

# Run the head-to-head benchmark
python -m comparison.benchmark
```

Outputs (heatmaps, decision surfaces, plots, CSV) land in `./outputs/`.

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

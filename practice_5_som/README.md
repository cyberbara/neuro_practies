# Practice 5 — Kohonen SOM

**Topic:** Self-organising map — unsupervised, topology-preserving projection.
**Stack:** NumPy.
**Files:** `som.py`, `demo.py`.

## What it does

A 2-D grid of neurons, each holding a d-dimensional weight vector. For
every input:

1. Find the best-matching unit (BMU) — the closest neuron by Euclidean
   distance.
2. Pull the BMU and its grid-neighbours towards the input by an amount
   that decays with both training step and grid distance.

Both learning rate `η` and neighbourhood radius `σ` shrink geometrically
over training — a standard schedule.

## Experiments in `demo.py`

1. Train an 8×8 SOM on Iris (standardised).
2. Plot the **U-matrix** — average inter-neuron distance, drawn as a
   heatmap. Lighter bands = cluster boundaries.
3. Plot the **most-frequent-class map** — the network's idea of cluster
   membership. Good topology preservation means each cell is dominated
   by one class.
4. Compare 5×5 vs 10×10 maps for quantisation error and training time.

Outputs in `outputs/`:

- `p5_som_8x8.png` — U-matrix + class map.
- `p5_som_sizes.png` — side-by-side 5×5 vs 10×10.

## Run

```bash
source .venv/bin/activate
python -m practice_5_som.demo
```

## Key takeaways

- Unsupervised, but the BMU→class assignment still tells us how well the
  map preserved the natural class structure of the data.
- Bigger maps have lower quantisation error but cost more to train — the
  5×5 vs 10×10 comparison shows the tradeoff.
- Topology preservation is a soft constraint: if neighbouring neurons
  don't end up representing similar inputs, the SOM has failed to
  organise.

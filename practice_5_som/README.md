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

All on the sklearn `digits` dataset — 1797 handwritten 8×8 images,
64 features, 10 classes.

1. Train a 12×12 SOM; plot the **U-matrix** (average inter-neuron
   distance — lighter bands mark cluster boundaries) and the
   **most-frequent-class map** (the network's idea of cluster membership).
2. Map-size sweep: 6×6, 10×10, 16×16, 20×20 — quantisation error and
   training time vs map size.
3. Side-by-side with a **PCA projection** of the same data, for
   reference on what a linear method does with this manifold.

Outputs in `outputs/`:

- `p5_som_digits.png` — U-matrix + class map, 12×12.
- `p5_som_sizes.png` — quantisation error and time vs map size.
- `p5_som_vs_pca.png` — PCA scatter vs SOM class map.

## Run

```bash
source .venv/bin/activate
python -m practice_5_som.demo
```

## Measured results (sklearn digits)

| map | neurons | quantisation error | time |
|---|---|---|---|
| 6×6   | 36  | 4.975 | 3.5 s |
| 10×10 | 100 | 4.294 | 6.1 s |
| 16×16 | 256 | 3.699 | 12.1 s |
| 20×20 | 400 | 3.402 | 16.6 s |

## Key takeaways

- Unsupervised, but the BMU→class assignment still tells us how well the
  map preserved the natural class structure of the data.
- Bigger maps have lower quantisation error but cost more to train — and
  the error reduction flattens out. 20×20 gives only ~8% better
  quantisation than 16×16 for 2.6× the training time.
- Topology preservation is a soft constraint: if neighbouring neurons
  don't end up representing similar inputs, the SOM has failed to
  organise.
- Compare the class map with the PCA scatter — the SOM's advantage is
  that it puts class boundaries on the map's own grid, giving a
  topological map you can navigate, not just a projection.

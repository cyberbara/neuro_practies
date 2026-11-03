# Practice 5 — Kohonen SOM

**Topic:** Self-organising map — unsupervised, topology-preserving projection.
**Stack:** NumPy.
**Files:** `som.py`, `../common/letters.py` (shared dataset), `demo.py`.

## What it does

A 2-D grid of neurons, each holding a d-dimensional weight vector. For
every input:

1. Find the best-matching unit (BMU) — the closest neuron by Euclidean
   distance.
2. Pull the BMU and its grid-neighbours towards the input by an amount
   that decays with both training step and grid distance.

Both learning rate `η` and neighbourhood radius `σ` shrink geometrically
over training — a standard schedule.

**Two update modes:**
- `update_mode="online"` — the textbook loop, one random picture per step.
- `update_mode="batch"` — one epoch at a time. Every picture of the epoch
  contributes at once and each neuron receives the summed pull of all
  pictures that won it or one of its neighbours. Mathematically the same
  rule, but it runs on BLAS instead of a Python loop — with 784-dimensional
  pictures that is the difference between 5 seconds and 4 minutes.

## From unsupervised map to "find X"

The SOM trains **without labels**. After training the map is *labelled*
afterwards: every neuron takes the most common class of the pictures it
won. From then on the map answers two questions:

- `classify(X)` — the class of the BMU.
- `target_score(X, ...)` — a soft score in [0, 1]: the Gaussian-weighted
  share of the 5 nearest neurons that ended up on the target class.

## Experiments in `demo.py`

All on the letters dataset — 2 496 training pictures of 26 letters,
28×28, with unseen test fonts.

1. Train a 16×16 SOM; plot the **U-matrix** (average inter-neuron
   distance — lighter bands mark cluster boundaries) and the
   **most-frequent-class map** (the network's idea of cluster membership).
2. Map-size sweep: 6×6, 10×10, 16×16, 20×20 — quantisation error,
   labelled-map accuracy, X-detection and training time.
3. Side-by-side with a **PCA projection** of the same data, for
   reference on what a linear method does with this manifold.
4. **Where X lives** — the neurons that ended up on the letter X and what
   their prototypes look like.

Outputs in `outputs/`:

- `p5_som_letters.png` — U-matrix + class map, 16×16.
- `p5_som_sizes.png` — quantisation error, accuracy and time vs map size.
- `p5_som_vs_pca.png` — PCA scatter vs SOM class map.
- `p5_x_cluster.png` — the X cluster on the map.

## Run

```bash
source .venv/bin/activate
python -m practice_5_som.demo
```

## Measured results (letters)

| map | neurons | quantisation error | 26-class acc | balanced acc (find X) | time |
|---|---|---|---|---|---|
| 6×6 | 36 | 8.848 | 0.333 | 0.682 | 2.6 s |
| 10×10 | 100 | 8.093 | 0.489 | 0.890 | 3.3 s |
| 16×16 | 256 | 7.302 | 0.646 | 0.853 | 5.6 s |
| 20×20 | 400 | 6.906 | 0.673 | 0.833 | 8.8 s |

## Key takeaways

- Unsupervised, but the BMU→class assignment still tells us how well the
  map preserved the natural class structure of the data.
- **The map must be big enough for the number of classes.** 26 letters in
  a 6×6 map (36 neurons) means roughly one neuron per letter — the map
  cannot place similar letters next to each other, and accuracy collapses
  to 0.333. By 16×16 there is room to spare.
- Bigger maps have lower quantisation error but cost more to train — and
  the error reduction flattens out. 20×20 gives only ~5% better
  quantisation than 16×16 for 1.6× the training time.
- Topology preservation is a soft constraint: if neighbouring neurons
  don't end up representing similar inputs, the SOM has failed to
  organise.
- Compare the class map with the PCA scatter — the SOM's advantage is
  that it puts class boundaries on the map's own grid, giving a
  topological map you can navigate, not just a projection.
- The batch update mode is not a different algorithm — it is the same rule
  with the epoch's pulls averaged instead of applied one by one, and it
  makes high-dimensional pictures tractable.

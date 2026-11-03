# Comparison benchmark

**Goal:** put all five practices on the same table — same data, same task,
same metric.

`python -m comparison.benchmark` runs every practice on the **letters
dataset** and asks each one the same question: **is this picture the
letter X?**

It then writes:

- `outputs/benchmark.csv` — the table.
- `outputs/comparison_bars.png` — three panels: quality, time, parameters.
- `outputs/comparison_scatter.png` — time vs quality, bubble size ∝ log params.

## The shared setup

| what | value |
|---|---|
| dataset | 26 uppercase letters, 28×28 binary, rendered with PIL from system fonts |
| train fonts | Carlito (4 styles) + Liberation Sans (4 styles) |
| test fonts | Liberation Serif (3 styles) + FreeSans (3 styles) — **never seen in training** |
| train size | 2 496 pictures (96 of them an X) |
| test size | 624 pictures (24 of them an X) |
| task | binary: X vs not-X |
| headline metric | **balanced accuracy** |

Balanced accuracy, not plain accuracy: X is 1 class out of 26, so a network
that always says "not X" already scores 96% plain accuracy. Balanced
accuracy weighs "found the X" against "left the others alone" equally —
which is exactly the question the practice asks.

## Results

| practice | params | neurons | train time | balanced acc | AUC | recall(X) | precision(X) | 26-class acc |
|---|---|---|---|---|---|---|---|---|
| `1_hebb_letters` | 20 384 | 26 | 0.015 s | 0.892 | 0.992 | 0.792 | 0.826 | 0.713 |
| `2_delta_letters` | 785 | 1 | 1.17 s | 0.903 | 0.981 | 0.833 | 0.556 | — |
| `3_backprop_mlp` | 120 346 | 26 | 1.17 s | 0.978 | 1.000 | 0.958 | 0.958 | 0.912 |
| `4_rbf_letters` | 3 146 | 120 | 2.47 s | 0.957 | 1.000 | 0.917 | 0.917 | 0.824 |
| `5_som_letters` | 200 704 | 256 | 5.53 s | 0.853 | 0.953 | 0.708 | 0.944 | 0.646 |

The `26-class acc` column is the accuracy over all 26 letters — the delta
rule gets a dash because a single neuron can only answer the binary
question, it cannot name all 26.

## How to read it

- **Balanced accuracy is comparable across rows.** Same data, same split,
  same metric — this table *is* a ranking.
- **Time is not a fair fight either.** Practice 1 trains in 15 ms because
  it only averages pictures; practice 3 runs 25 epochs of gradient descent
  over a 120k-parameter network. The methods are not doing comparable
  amounts of work.
- **Parameters are not comparable either.** The SOM's 200 704 numbers are
  256 neurons × 784 inputs — one prototype per neuron, and the map has to
  be big enough to cover 26 classes. The delta rule's 785 numbers do the
  same job with one neuron, because a straight line is enough to separate
  X from the rest (mostly).
- **The winner depends on what you optimise for.** By quality:
  backprop (0.978) > RBF (0.957) > delta (0.903) > Hebb (0.892) > SOM
  (0.853). By speed: Hebb wins by two orders of magnitude. By size: delta
  wins by 250× over the MLP.

## Caveats

- Practice 3's numbers are not its ceiling — it trains for 25 epochs on a
  2 496-picture dataset to keep the benchmark fast.
- Practice 5's SOM is trained *without labels*; the map is labelled
  afterwards, which is the standard way to turn an SOM into a classifier.
- Test fonts are deliberately unseen families, so every number here
  measures generalisation to new handwriting styles, not memorisation.

## Read

```bash
cat outputs/benchmark.csv
```

## Re-run after edits

```bash
python -m comparison.benchmark
```

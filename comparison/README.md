# Comparison benchmark

**Goal:** put all five practices on the same table.

`python -m comparison.benchmark` runs each practice, measures training
wall-clock time, parameter count, and a problem-specific quality metric
(classification accuracy for 2/3/4, pattern-recovery rate for 1,
quantisation error for 5), and writes the result to
`outputs/benchmark.csv`.

## Caveats

- The accuracy numbers are NOT comparable across practices. Each practice
  has its own natural problem and metric. The point of the table is the
  *shape* of the column: which methods need more neurons, more compute,
  more data.
- The neural-network practice (3) dwarfs everything else in parameters
  and time because Fashion-MNIST is a real ML benchmark; the others are
  toy problems.

## Read

```bash
cat outputs/benchmark.csv
```

## Re-run after edits

```bash
python -m comparison.benchmark
```

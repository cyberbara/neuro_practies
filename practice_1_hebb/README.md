# Practice 1 — Hebb's rule

**Topic:** Hebbian learning — prototypes without gradients.
**Stack:** NumPy + PIL (system fonts).
**Files:** `hebb.py` (network), `../common/letters.py` (shared dataset),
`demo.py`, `README.md`.

## What it does

One output neuron per letter. Hebb's rule with the output clamped to the
correct class:

```
ΔW[c] = η · x
```

The winner's weight vector moves towards the picture it just saw. After
training, each row of `W` is (up to scale) the average picture of its
class — a **prototype** the network built by itself, with no gradients, no
learning-rate schedule and no teacher beyond the class label.

Readout:

```
s_c(x)   = cos(W[c], x)          similarity with the prototype
P(class) = softmax_c(s_c(x) / T)
```

The answer to "is this the letter X?" is the softmax weight of the X
neuron.

**There is no Oja rule here.** Plain Hebb grows the weights without bound,
so instead of shrinking them during training we normalise at the *decision*
step (cosine similarity). The update rule stays textbook-Hebb, and the
network still works.

## Dataset — `common/letters.py`

The dataset is shared by all five practices:

- **26 letters** A–Z, 28×28, binary (ink = 1, background = 0)
- **train fonts** — Carlito (Regular / Bold / Italic / Bold-Italic) +
  Liberation Sans (4 styles)
- **test fonts** — Liberation Serif (3 styles) + FreeSans (3 styles),
  **families never seen during training**
- 12 renderings per (letter, train font) → 2 496 train / 624 test
- augmentation per picture: ±2 px offset, random rotation ±8°, glyph
  scaling

Test fonts are disjoint on purpose: the score then measures
generalisation to new handwriting styles, not memorisation.

## Experiments in `demo.py`

1. Show the dataset — 26 letters, train row vs unseen-font row.
2. Learning curve — 12 epochs of Hebb.
3. Data curve — balanced accuracy vs how many pictures per class the
   network saw.
4. Prototypes — what each neuron learned (the pictures look like the
   letters).
5. Detection — score distribution for X vs the rest, plus examples.
6. Robustness — noise and rectangular occlusion.

Outputs in `outputs/`:

- `p1_letters.png` — the dataset.
- `p1_learning_curve.png` — accuracy vs epoch.
- `p1_data_curve.png` — accuracy vs training-set size.
- `p1_prototypes.png` — the learned prototypes.
- `p1_detection.png` — score distribution + examples.
- `p1_robustness.png` — accuracy under noise / occlusion.

## Run

```bash
source .venv/bin/activate
python -m practice_1_hebb.demo
```

## Key takeaways

- **Hebb learns prototypes, not features.** After one pass the weight
  matrix is a gallery of average letters — the picture below shows it
  directly.
- **One pass is enough.** The learning curve is flat: Hebb only averages,
  so the second epoch adds nothing but a scale factor. That is a property
  of the rule, not a bug.
- **Data matters more than epochs.** The data curve keeps climbing long
  after the learning curve has flattened — more pictures per class means
  a sharper prototype.
- **Plain Hebb without normalisation is unusable for classification**
  (weights grow without bound), but normalising at the readout step
  solves it without touching the learning rule — no Oja needed.
- **Hebb is not enough on its own.** It gets 0.71 on 26 classes versus
  0.91 for backprop: a prototype per class cannot express "X is a
  diagonal cross" as well as a trained network can.

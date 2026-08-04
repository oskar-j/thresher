# Thresher

**Your model gives you probabilities. Where you cut them is a decision — stop leaving it at 0.5.**

A classifier that outputs `predict_proba` hands you a number between 0 and 1. Turning that
into an actual yes-or-no answer needs a cut-off, and almost every pipeline uses 0.5 —
because it is the default, not because anyone measured it.

The cut-off that actually maximizes accuracy depends on your data: how far the two classes
overlap, how imbalanced they are, and how well your model is calibrated. It is rarely 0.5.
When one class is rare it can be nowhere near it.

Thresher measures it:

```python
from thresher import Thresher

Thresher().optimize_threshold(scores, actual_classes)
```

Or from a terminal, without writing any Python:

```console
$ thresher scores.csv
0.35
```

## What you get

<div class="grid cards" markdown>

-   **Exact, not approximate**

    The default algorithm returns the best threshold that exists — optimal over every
    split a threshold can induce, verified against brute force.

-   **And fast**

    `O(n log n)`, which is roughly 1,400× quicker than the exhaustive search it replaced
    at 16,000 rows, and the gap widens with every row.

-   **Seven algorithms**

    The exact sweep plus six approximations, all selectable by name, with a
    [measured comparison](algorithms.md#how-they-compare) of accuracy, speed and memory.

-   **Scales out**

    Optional [parallel backends](backends.md) spread the counting over your CPU cores or a cluster, and
    [Spark](spark.md) reads the data where it already lives — neither changes the answer.

</div>

## Where to go next

- [Getting started](getting-started.md) — install it and find your first threshold
- [Algorithms](algorithms.md) — what each one does, and which to choose
- [Command line](cli.md) — the `thresher` command
- [API reference](api/thresher.md) — every public class and function

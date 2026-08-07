# Algorithms

Seven are available. The first is exact and the default; the rest approximate.

```python
Thresher(algorithm="grid").optimize_threshold(scores, actual_classes)
```

`'auto'`, `'default'` and `'default_heuristics'` are accepted as synonyms for the default.
`Thresher.get_supported_algorithms()` lists the ids at runtime, and each has synonyms —
`thresher --list-algorithms` shows them all.

## How they compare

On 2,000 rows, averaged over 5 seeds. Accuracy is relative to the **exact** optimum, computed
independently by sweeping the sorted scores rather than by asking one of the algorithms
under test, so 100% means "found a cut-off as good as the best one that exists".

| Algorithm | Separable | Overlapping | Imbalanced | Time | Complexity | Memory |
|---|---|---|---|---|---|---|
| `exact` | **100.00%** | **100.00%** | **100.00%** | **1 ms** | **O(n log n)** | **O(d)** |
| `hist` | 99.99% | 99.99% | 99.98% | 1 ms | O(n + k) | **O(k)** |
| `ls` | 100.00% | 100.00% | 100.00% | 266 ms | O(n²) | O(n) |
| `grid` | 99.83% | 99.99% | 99.92% | 11 ms | O(c·n) | O(c) |
| `sgrid` | 99.57% | 97.81% | 99.62% | 1 ms | O(c·r·n) | O(r·n) |
| `gen` | 99.62% | 98.23% | 88.64% | 117 ms | O(e·r·n) | O(r·n) |
| `sgd` | 99.54% | 97.01% | 88.37% | 12 ms | O(i·r·n) | O(r·n) |

Where _n_ is the number of scores, _d_ the number of **distinct** scores, _c_ the grid
candidates, _k_ the histogram bins, _r_ the `stoch_ratio` sample fraction, _e_ the genetic evaluations and _i_ the
sgd steps.

Reproduce it with `uv run python examples/benchmark.py`.

!!! tip "Use `exact` unless you have a reason not to"

    It is the only exact algorithm here *and* the cheapest. The others are strictly
    dominated on this evidence — they exist because exactness used to cost `O(n²)`.

## Size guidance

Each algorithm declares the input size beyond which it becomes slow, and logs a warning
rather than making you wait to find out:

```
WARNING  Linear search is likely to be slow on 12,000 rows - it is usually comfortable up
to about 10,000. The 'exact' algorithm is exact and O(n log n)...
```

| Algorithm | Comfortable up to | Why |
|---|---|---|
| `hist` | 50,000,000 | one pass, no sort, and memory that does not follow the input |
| `exact` | 10,000,000 | `O(n log n)` — the input runs out before the algorithm does |
| `sgrid` | 10,000,000 | reads only `stoch_ratio` of the data per candidate |
| `sgd` | 2,000,000 | linear in the sampled fraction |
| `grid` | 1,000,000 | linear, with a fixed candidate count |
| `gen` | 100,000 | linear, but thousands of evaluations per run |
| `ls` | 10,000 | `O(n²)` — 0.9 s at 4,000 rows becomes 18 s at 16,000 |

These are guidance rather than limits, measured on one laptop at roughly where a run passes
ten seconds. Crossing one warns and continues. Silence it by asking for less:

```python
import thresher
thresher.set_verbosity("error")
```

See [reporting and progress](logging.md) for the rest of what a run can be asked to say.

## Exact sweep

`exact` — **the default.** Returns the best threshold that exists, in `O(n log n)`.

Linear search is quadratic because it recomputes the whole confusion matrix for each
candidate. That work is almost all redundant: moving the threshold past one sample changes
the number of correct predictions by exactly one, in a direction fixed by that sample's
class. Sort once, then sweep while carrying running counts:

```
correct(k) = (negatives among the first k) + (positives among the remaining n - k)
```

Both terms are running totals, so each candidate costs constant time and the sort is the
only real expense.

This is the standard exact splitter for a decision-stump threshold, and the same linear
scan that generates an ROC curve — see Fawcett, *An introduction to ROC analysis* (Pattern
Recognition Letters, 2006), and Google's
[decision forests documentation](https://developers.google.com/machine-learning/decision-forests/binary-classification),
which gives the same `O(n log n)` bound "because of the sorting of the feature values".

It has no parameters. There is no accuracy left to trade for speed.

## Histogram sweep

`hist` — added in `0.5.3`. The approximation to reach for when the data is too large to
hold, or when you want a bounded, predictable error rather than a statistical one.

The exact sweep sorts the scores and walks them. Sorting is what costs it `O(n log n)` and
what forces it to keep the data. This gives up a little precision for neither: divide the
score range into a fixed number of bins, count the classes falling into each in one pass,
then sweep the *bins* with the same running-total argument:

```
correct(j) = (negatives in bins below j) + (positives in bins from j upwards)
```

Nothing is sorted, no row is read twice, and the only thing held is the counters — so
memory is set by the resolution, not the input:

| | `hist` | `exact` |
|---|---|---|
| 100,000 rows | 19 KB | 12 MB |
| 1,000,000 rows | **49 KB** | 107 MB |

Those figures hold whatever you hand it — a list, an array or a Series. Until 0.7.2 they
only held for a list, because `optimize_threshold` copied the other two on the way in and
so allocated the input size before this algorithm ever ran. See
[what you can pass in](getting-started.md#what-you-can-pass-in).

The cost is resolution. A threshold can only sit on a bin edge, so the answer is off by at
most one bin width — an error you control directly, and the same every run, unlike the
sampling-based approximations. At the default 1,024 bins it captures 99.98% of the
achievable accuracy.

| Parameter | Default | Meaning |
|---|---|---|
| `no_of_bins` | 1024 | resolution. Doubling it halves the worst-case error and costs one more counter per bin — and nothing per row |

Where `grid` also tests a fixed set of candidates, it rescans every row for each one
(`O(c·n)`). This reads each row once, whatever the resolution.

## Linear search

`ls` — exhaustive and exact, but `O(n²)`. Superseded by the exact sweep, which returns the
same answer, or a marginally better one, far faster. Kept for comparison and for its
multiprocessing option.

| Parameter | Default | Meaning |
|---|---|---|
| `n_jobs` | 1 | `-1` uses every processor bar one; 2 or more enables multiprocessing |

## Grid search

`grid` — evaluates every point on an evenly spaced grid across the data. Cost depends on
the grid resolution rather than the input size, and its memory does not grow with the input.

Since `0.6.4` the grid spans `[min(scores), max(scores)]`, plus one point below the minimum
so "classify everything as positive" remains expressible. It previously spanned `[0, 1]`
whatever the scores held, so any non-probability score — a logit, a margin — put every
candidate outside the data and returned an edge at chance accuracy.

| Parameter | Default | Meaning |
|---|---|---|
| `no_of_decimal_places` | 2 | grid resolution — `10**places + 1` candidates, spread across the range the data occupies |

## Stochastic grid search

`sgrid` — the same grid, with each candidate scored against a random subsample.

| Parameter | Default | Meaning |
|---|---|---|
| `no_of_decimal_places` | 2 | grid resolution |
| `stoch_ratio` | 0.05 | fraction of the data each candidate reads |
| `reshuffle` | False | draw a fresh subsample per candidate instead of reusing one |

## Evolutionary algorithm

`gen` — evolves a population of candidate thresholds, scoring each against random
subsamples, discarding the least fit and breeding replacements. What comes back is the
fittest agent that was measured, across every generation.

| Parameter | Default | Meaning |
|---|---|---|
| `population_size` | 30 | agents per generation |
| `number_of_generations` | 20 | rounds of selection |
| `number_of_iterations` | 10 | fitness samples per agent per generation |
| `sus_factor` | 2 | how many of the least fit are left child-less. Must be below `population_size` |
| `stoch_ratio` | 0.02 | fraction of the data each sample reads |
| `mutation_chance` | 0.05 | probability one agent is nudged per generation |
| `mutation_factor` | 0.10 | how far that nudge can move it, either way |

Until 0.7.3 it returned the mean of the population bred after the last round of scoring,
which nothing had ever evaluated — so a crossover and a mutation reached the answer with
no selection in front of them. With the nudge firing every generation at
`mutation_factor=50`, that returned `1.3188` on data spanning `[0, 1]`, at 53% accuracy
where 90% was available. The nudge was also drawn from `[0, mutation_factor)` and so could
only ever push a threshold up.

## Stochastic gradient descent

`sgd` — walks down the error curve from the mean of the scores, scoring each step against a
random subsample. The least precise algorithm here, and least reliable when one class is
rare, because then a small subsample says little about where the boundary lies.

| Parameter | Default | Meaning |
|---|---|---|
| `num_of_iters` | 200 | maximum steps |
| `stop_thresh` | 0.001 | improvement below which a step counts as making no progress |
| `stop_patience` | 3 | how many such steps in a row end the walk |
| `alpha` | 0.01 | how quickly the step size decays |
| `step_ratio` | 0.05 | the first step, as a fraction of the score range |
| `stoch_ratio` | 0.05 | fraction of the data each step reads |

Raising `stoch_ratio` is the lever against the imbalance weakness. On 2,000 rows with 5%
positives, going from `0.05` to `0.5` took mean error from `0.0394` to `0.0035` and the
worst case from `0.302` to `0.013`, at the cost of reading ten times as much data per step.

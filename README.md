# Thresher - THRESHold EvaluatoR for Python

[![PyPI version](https://img.shields.io/pypi/v/thresher-py.svg)](https://pypi.org/project/thresher-py/)
[![Python versions](https://img.shields.io/pypi/pyversions/thresher-py.svg)](https://pypi.org/project/thresher-py/)
[![Build](https://img.shields.io/github/actions/workflow/status/oskar-j/thresher/ci.yml?branch=main&label=build)](https://github.com/oskar-j/thresher/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-%E2%89%A5%2090%25-brightgreen)](https://github.com/oskar-j/thresher/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/readthedocs/thresher)](https://thresher.readthedocs.io/en/stable/)
[![Downloads](https://img.shields.io/pepy/dt/thresher-py)](https://pepy.tech/project/thresher-py)
[![Stars](https://img.shields.io/github/stars/oskar-j/thresher)](https://github.com/oskar-j/thresher/stargazers)
[![License](https://img.shields.io/pypi/l/thresher-py.svg)](https://github.com/oskar-j/thresher/blob/main/LICENSE)

> ### Your model gives you probabilities. Where you cut them is a decision — stop leaving it at 0.5.

A classifier that outputs `predict_proba` hands you a number between 0 and 1. Turning that
into an actual yes-or-no answer needs a cut-off, and almost every pipeline uses 0.5 —
because it is the default, not because anyone measured it.

The cut-off that actually maximizes accuracy depends on your data: how far the two classes
overlap, how imbalanced they are, and how well your model is calibrated. It is rarely 0.5.
When one class is rare it can be nowhere near it, and the gap between the default and the
right answer is the gap between a model that looks fine on paper and one that is useful.

**Thresher measures it.** Give it your scores and the ground truth, and it returns the
threshold that classifies the highest fraction of your samples correctly:

```python
from thresher import Thresher

Thresher().optimize_threshold(scores, actual_classes)
```

Or without writing any Python at all, straight from a terminal:

```
thresher scores.csv
```

That is the whole interface. Everything else in this README is about tuning what happens
underneath — which of the seven search algorithms runs, and how hard it looks.

📖 **Full documentation: [thresher.readthedocs.io](https://thresher.readthedocs.io/en/stable/)** — a build for every release, with a version switcher.

## Table of contents

- [Project description](#project-description)
  - [Choosing an algorithm](#choosing-an-algorithm)
- [Implemented algorithms](#implemented-algorithms)
  - [Exact sweep](#exact-sweep)
  - [Histogram sweep](#histogram-sweep)
  - [Linear search](#linear-search)
  - [2-dim Stochastic Gradient Descent](#2-dim-stochastic-gradient-descent)
  - [Evolutionary algorithm](#evolutionary-algorithm)
  - [Grid search](#grid-search)
  - [Stochastic Grid search](#stochastic-grid-search)
- [Algorithm scores](#algorithm-scores)
- [How to setup?](#how-to-setup)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Development setup](#development-setup)
  - [Project layout](#project-layout)
- [Custom parameters](#custom-parameters)
  - [Control parameters for the algorithms](#control-parameters-for-the-algorithms)
- [Sample usage](#sample-usage)
- [Handling errors](#handling-errors)
- [Command line](#command-line)
- [Running on Ray](#running-on-ray)
- [Performance tests](#performance-tests)
- [Future work](#future-work)

## Project description

A bare pandas implementation of a tool for finding the threshold which maximizes accuracy
of `predict_proba` like-outputs (from e.g. `scikit-learn`), in regard to the provided ground truth (labels).

_Note: you can jump directly to the sample usage [here](https://github.com/oskar-j/thresher#sample-usage)._

Method interesting for the user is `optimize_threshold(scores, actual_classes)`, which is available
from the `Thresher` class. This method, for given _scores_ and _actual classes_,
returns a threshold that yields the _**highest fraction** of correctly classified_ samples.

```
optimize_threshold parameters:
  scores​:list
    The list of scores.
  actual_classes​:list
    The list of ground truth (correct) classes.
    Classes are represented as -1 and 1.
returns:
  threshold:​float
    The threshold value that yields ​the highest fraction of correctly classified
    samples​. If multiple thresholds give the optimal fraction, return any threshold.
```

### Choosing an algorithm

[Exact sweep](#exact-sweep) is the default, and is what you want unless you have a specific
reason otherwise: it returns the best threshold that exists, and does so faster than any of
the approximations. Pick a different one by name when you want to:

```python
Thresher(algorithm='grid').optimize_threshold(scores, actual_classes)
```

`'auto'`, `'default'` and `'default_heuristics'` are all accepted as synonyms for the
default, so you can spell it either way.

Each algorithm declares the input size beyond which it becomes slow, and warns rather than
making you wait to find out:

```
WARNING thresher.dispatch: Linear search is likely to be slow on 12,000 rows - it is
usually comfortable up to about 10,000. The 'exact' algorithm is exact and O(n log n)...
```

| Algorithm | Comfortable up to | Why |
|---|---|---|
| `exact` | 10,000,000 | `O(n log n)` — the input size runs out before the algorithm does |
| `sgrid` | 10,000,000 | reads only `stoch_ratio` of the data per candidate |
| `sgd` | 2,000,000 | linear in the sampled fraction |
| `grid` | 1,000,000 | linear, with a fixed candidate count |
| `gen` | 100,000 | linear, but thousands of evaluations per run |
| `ls` | 10,000 | `O(n²)` — 0.9 s at 4,000 rows becomes 18 s at 16,000 |

These come from the timings in `examples/benchmark.py` on one laptop, at roughly where a
run passes ten seconds. They are guidance, not limits — a faster machine moves them all up,
which is why crossing one warns rather than refuses. Silence it with ordinary logging
configuration:

```python
logging.getLogger("thresher").setLevel(logging.ERROR)
```

## Implemented algorithms

### Exact sweep

**The default, and the one you want.** Added in version `0.4.0`. It returns the best
threshold that exists - not an approximation - in `O(n log n)`.

The insight is that linear search does the same work over and over. Moving a threshold
past one sample changes the number of correct predictions by exactly one, in a direction
fixed by that sample's class, so there is no need to rescore the whole dataset for every
candidate. Sort the samples once, then sweep through them keeping running counts:

```
correct(k) = (negatives among the first k) + (positives among the remaining n - k)
```

Both terms are running totals, so each candidate costs constant time and the sort is the
only real expense.

This is the standard exact splitter used to pick a decision-stump threshold, and the same
linear scan that generates an ROC curve - see Fawcett, _An introduction to ROC analysis_
(Pattern Recognition Letters, 2006), and Google's
[decision forests documentation](https://developers.google.com/machine-learning/decision-forests/binary-classification),
which gives the same `O(n log n)` bound "because of the sorting of the feature values".

It has no parameters. There is no accuracy left to trade for speed.

Since `0.4.1` it considers both edge splits as well as every interior one, so the answer is
optimal over *every* split a threshold can induce. One of those edges — "classify
everything as positive" — needs a threshold below your smallest score, so that is the one
case where the result can fall just outside the range of your input. It is returned only
when it genuinely beats every threshold inside the data, which takes scores and classes
running contrary to each other.

Compared with linear search, which it supersedes, it is exact in the same sense and
strictly faster - by 104× at 1,000 rows and 1,358× at 16,000, a gap that widens with every
row. It is also *slightly more accurate*: linear search only ever considers the midpoints
between adjacent scores, so it cannot express the "classify everything as negative" split,
which the sweep reaches for free at `max(scores)`. On randomised inputs that mattered in
about 10% of cases.

### Histogram sweep

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

The cost is resolution. A threshold can only sit on a bin edge, so the answer is off by at
most one bin width — an error you control directly, and the same every run, unlike the
sampling-based approximations. At the default 1,024 bins it captures 99.98% of the
achievable accuracy.

| Parameter | Default | Meaning |
|---|---|---|
| `no_of_bins` | 1024 | resolution. Doubling it halves the worst-case error and costs one more counter per bin — and nothing per row |

Where `grid` also tests a fixed set of candidates, it rescans every row for each one
(`O(c·n)`). This reads each row once, whatever the resolution.

### Linear search

Superseded by [Exact sweep](#exact-sweep), which returns the same answer - or a
marginally better one - in a fraction of the time. Kept for comparison and for its
multiprocessing option.

This is the most basic, iterative approach. For every _threshold_ present in the input (in the _scores_ list), we evaluate it by calculating the exact accuracy of _split_ produced by such threshold. Then, return the threshold which produce the most accurate split.

List of parameters to customize:
* `n_jobs` (default: 1) - set to `-1` for using all available processors except one; any value of `2` or more
enables multiprocessing, while the default value of `1` disables multiprocessing

### 2-dim Stochastic Gradient Descent

This algorithm uses a naive implementation of the popular algorithm 'Stochastic Gradient Descent', which tries to converge over a function - in our case, it
is an error curve representing ratio of miss-classifies for a threshold. Using a gradient, algorithm follows the curve to find the optimal value, that is,
a threshold producing the smaller number of miss-classifies.

Each step scores only a random subsample, which is what makes it cheap enough for the
largest inputs - and also what makes it the least precise algorithm here. It walks from the
mean of your scores and keeps the best point it visits. It is least reliable when one class
is rare, because then a subsample carries little information about where the boundary lies;
prefer grid search where you can afford it.

List of parameters to customize:
* `num_of_iters` (default: 200) - number of iterations during which algorithm tries to converge
* `stop_thresh` (default: 0.001) - improvement below which a step counts as making no progress
* `stop_patience` (default: 3) - how many such steps in a row end the walk. Every evaluation
reads a different random subsample, so a single small improvement is as likely to be noise as
real convergence; raise this if the result looks like it stopped short
* `alpha` (default: 0.01) - how quickly the step size decays as the walk proceeds
* `stoch_ratio` (default: 0.05) - fraction of the data each step reads. Raising this is the
lever against this algorithm's weak spot: when one class is rare, a small subsample carries
little information about where the boundary lies. On 2,000 rows with 5% positives, going
from `0.05` to `0.5` took the mean error from `0.0394` to `0.0035` and the worst case from
`0.302` to `0.013`, at the cost of reading ten times as much data per step

### Evolutionary algorithm

This is a simulation approach which uses an evolutionary algorithm. It works by simulating multiple generations of a "population" of candidate solutions. During every iteration of a single generation, algorithm stochasticly evaluates the candidate solution. After the end of a single generation, we remove the from the population least fit agents (solutions), and do the _crossover_ between the left solitions to produce new "offspring" candidate solutions. Moreover, they may mutate to provide additional random chance.

List of parameters to customize:
* `population_size` (default: 30) - number of agents in the simulation
* `number_of_generations` (default: 20) - number of generations
* `number_of_iterations` (default: 10) - number of iterations per a generation
* `sus_factor` (default: 2) - how many least-fit agents should be childless at the end of generation
* `stoch_ratio` (default: 0.02) - percentage of data to evaluate fit of a single agent per iteration
* `optimized_start` (default: True)
* `mutation_chance` (default: 0.05)
* `mutation_factor` (default: 0.10)

### Grid search

Added in version `0.1.2`. This algorithm works by generate a grid of possible solutions, with a granularity set
by parameter named `no_of_decimal_places`. All candidate solutions are evaluated thoroughly
and the best one is chosen at the end.

List of parameters to customize:
* `no_of_decimal_places` (default: 2) - generate the grid by rounding the number to the given number of decimal places

### Stochastic Grid search

Added in version `0.1.2`. This algorithm works similarly like the above-mentioned 'Grid search' method, with the difference, that
every single point generated by the grid is evaluated only partially (which can be controlled by the `stoch_ratio` parameter)

List of parameters to customize:
* `no_of_decimal_places` (default: 2) - generate the grid by rounding the number to the given number of decimal places
* `stoch_ratio` (default: 0.05) - percentage of data to evaluate fit of a candidate number in the grid
* `reshuffle` (default: False) - set whether the random projection should be calculated every step, or not

## Algorithm scores

How the seven compare on 2,000 rows, averaged over 5 seeds. Accuracy is relative to the
**exact** optimum — the best accuracy any single threshold could reach on that dataset,
computed independently by sweeping the sorted scores rather than by asking one of the
algorithms under test. 100% therefore means "found a cut-off as good as the best one that
exists", not merely "did well".

| Algorithm | Separable | Overlapping | Imbalanced | Time | Complexity | Memory |
|---|---|---|---|---|---|---|
| `exact` | **100.00%** | **100.00%** | **100.00%** | **1 ms** | **O(n log n)** | **O(d)** |
| `hist` | 99.99% | 99.99% | 99.98% | 1 ms | O(n + k) | **O(k)** |
| `ls` | 100.00% | 100.00% | 100.00% | 221 ms | O(n²) | O(n) |
| `grid` | 99.82% | 99.98% | 100.00% | 18 ms | O(c·n) | O(c) |
| `sgrid` | 99.57% | 97.84% | 99.70% | 1 ms | O(c·r·n) | O(n) |
| `gen` | 99.62% | 98.23% | 88.64% | 117 ms | O(e·r·n) | O(r·n) |
| `sgd` | 99.54% | 97.01% | 88.37% | 12 ms | O(i·r·n) | O(r·n) |

Where _n_ is the number of scores, _d_ the number of **distinct** scores (never more than
_n_, and far fewer for rounded probabilities), _c_ the grid candidates
(`10**no_of_decimal_places + 1`, so 101 by default), _k_ the histogram bins, _r_ the `stoch_ratio` sample fraction,
_e_ the genetic evaluations (`population_size × number_of_generations × number_of_iterations`)
and _i_ the sgd steps (at most `num_of_iters`).

Memory is the peak extra allocation, read off the implementations. Three of the entries are
worth a second look:

- **`exact` is O(d), not O(n)** — it keeps one count per *distinct* score, not per row. If
  your probabilities are rounded to two decimals, that is at most 101 entries whatever the
  row count, which is why it stays cheap on very large inputs.
- **`grid` and `hist` do not grow with the input at all** — `grid` allocates its candidates
  and streams the data past them; `hist` keeps one pair of counters per bin. At a million
  rows `hist` uses 49 KB against `exact`'s 107 MB.
- **`sgrid` is O(n) despite reading only a fraction of the data.** It builds the full
  paired list before sampling from it, so the subsampling buys time but not memory. That is
  an implementation detail rather than something inherent to the algorithm.

The top row is the short version of the whole table: **exact accuracy, at the lowest cost
of anything here**. Every other algorithm exists because, before `0.4.0`, exactness meant
paying `O(n²)` for it.

The growth rates are why. Linear search rescores all n samples for each of n-1 candidates;
the exact sweep sorts once and never rescores anything:

| Algorithm | n=1,000 | n=4,000 | n=16,000 | growth per 4× |
|---|---|---|---|---|
| `exact` | **0.6 ms** | **2.2 ms** | **12 ms** | **~4× (linear-ish)** |
| `ls` | 57 ms | 944 ms | 17,796 ms | ~16× (quadratic) |
| `grid` | 9 ms | 38 ms | 161 ms | ~4× |
| `sgrid` | <1 ms | 2 ms | 9 ms | ~4× |
| `gen` | 60 ms | 198 ms | 855 ms | ~4× |
| `sgd` | 5 ms | 18 ms | 67 ms | ~4× |

At 16,000 rows that is 12 ms against 18 seconds — roughly 1,400×, and the gap widens with
every row, because one algorithm is linear-ish and the other is quadratic.

What to take from this:

- **Use `exact` unless you have a specific reason not to.** It is the default, it is the
  best answer available, and it is the cheapest way to get it.
- **The approximations are now strictly dominated**: slower *and* less accurate than
  `exact`. They stay selectable, and remain interesting if you want to watch how a
  particular search behaves, but there is no longer an accuracy/speed trade to make.
- **`gen` and `sgd` struggle when one class is rare.** Both read only a small random
  subsample per evaluation, and the rarer the minority class, the less any subsample says
  about where the boundary lies. That weakness was the strongest argument for having an
  exact algorithm which stays cheap at scale.

Timings come from one laptop and are only meaningful relative to one another. Reproduce
the whole table with:

```
uv run python examples/benchmark.py
```

## How to setup?

The process is rather straightforward, you just need to just whether to install
from the sources (latest revision), or from the PyPI repository (stable release).

### Requirements

Requires Python `3.10+`. Tested on Python 3.10, 3.11, 3.12, 3.13 and 3.14.

### Installation

Stable release using the `pip` tool:

```
pip install thresher-py
```

Or with [uv](https://docs.astral.sh/uv/):

```
uv add thresher-py
```

Installation from source (latest revision):

```
pip install git+https://github.com/oskar-j/thresher.git
```

### Development setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management, with
`pyproject.toml` and a locked `uv.lock`:

```
uv sync --group dev
```

Run the test suite ([pytest](https://docs.pytest.org/), from anywhere in the repo):

```
uv run pytest
```

Or use the shortcuts, which also install the git pre-commit hook:

```
make install     # dependencies + git hook
make test        # the test suite
make check       # every lint, format and type check CI runs
make help        # the rest
```

Prefer `make check` to running `pre-commit run --all-files` directly: that command only
sees files git already tracks, so a newly created file is skipped in silence and the run
reports success without having looked at it. `make check` makes new files visible to the
hooks first.

#### Coverage

```
make cov
```

Test coverage is measured with branch coverage on, and **CI enforces a floor of 90% on
every supported Python version**. A pull request that drops below it fails the `test`
jobs, which are required checks on `main` — so the badge above states what is actually
guaranteed rather than a number that could drift.

One caveat when running it locally: Ray cannot be installed on macOS x86_64, so
`backends/ray_backend.py` is uncovered there and the local figure understates what CI
measures.

### Project layout

```
src/thresher/     the package  (src layout, so tests run against the installed copy)
  algs/           one sub-package per algorithm, plus shared helpers in algs/common
tests/            pytest suite; fixtures in conftest.py, data in tests/data
docs/             documentation, with images in docs/assets
examples/         runnable usage samples
```

## Custom parameters

It's possible to provide additional parameters in the `Thresher` constructor.

```python
Thresher(algorithm='exact',
         allow_parallel=True,
         verbose=False,
         progress_bar=False,
         labels=(0,1))
```

Here is a description of what does every particular parameter do:

* **algorithm** (default value: `'exact'`) - choose the algorithm from the list of available ones.
The same effect can be achieved by calling `set_algorithm(algorithm_name)` on the `Thresher` instance.
`'auto'`, `'default'` and `'default_heuristics'` are accepted as synonyms for the default
* **allow_parallel** (default value: `True`) - enables/disabled multiprocessing for algorithms
* **verbose** (default value: `False`) - enables verbosity
* **progress_bar** (default value: `False`) - shows a progress bar in the terminal (if supported by the algorithm)
* **labels** - necessary if your labels are different from `(-1, 1)` - first item from the tuple/list is a negative label,
and the second item is a positive label

### Control parameters for the algorithms

Some of the above-mentioned algorithms allow to change their parameters.
They should be provided in a dictionary, inside the `algorithm_params` parameter.
If no such customs parameters are provided, default values apply.

Examples:

```python
t = thresher.Thresher(algorithm_params={'n_jobs': 3})
```

```python
t = thresher.Thresher(algorithm_params={'no_of_decimal_places': 3,
                                        'stoch_ratio': 0.10})
```

## Sample usage

```python
from thresher import Thresher

t = Thresher()

print('Currently supported algorithms:')
print(t.get_supported_algorithms())

cases = [0.1, 0.3, 0.4, 0.7]
actual_labels = [-1, -1, 1, 1]

print(f'Optimization result: {t.optimize_threshold(cases, actual_labels)}')
```

See the [examples](https://github.com/oskar-j/thresher/tree/main/examples) directory for more sample code.

## Handling errors

Everything thresher rejects raises a subclass of `ThresherError`, so you can catch this
package's failures without also catching unrelated ones from numpy, pandas or your own
code:

```python
from thresher import Thresher
from thresher.exceptions import InvalidInputError, ThresherError

try:
    threshold = Thresher().optimize_threshold(scores, actual_classes)
except InvalidInputError as exc:
    print(f"the data cannot be optimized over: {exc}")
```

The hierarchy is:

```
ThresherError
├── ConfigurationError        a name that does not exist            (ValueError)
│   ├── UnknownAlgorithmError
│   └── UnknownBackendError
├── InvalidInputError         the data cannot be optimized over     (ValueError)
│   ├── EmptyInputError
│   ├── LengthMismatchError
│   ├── MissingLabelsError
│   ├── UnexpectedLabelsError
│   ├── SingleClassError
│   └── InsufficientDataError
├── LabelMappingError         the `labels` option cannot map        (TypeError)
├── NotIterableError          scores or classes are not iterable    (AttributeError)
├── BackendDependencyError    an optional dependency is missing     (ImportError)
├── AlgorithmNotWiredError    a bug in this package                 (NotImplementedError)
└── ShardMergeError           a bug in this package                 (ValueError)
```

Each class **also** inherits the builtin shown on the right — the one it was raised as
before `0.4.5` — so existing `except ValueError` code keeps working unchanged.

Errors carry their detail as attributes, so you do not have to parse the message:

```python
except LengthMismatchError as exc:
    print(f"{exc.score_count} scores against {exc.class_count} classes")
except UnknownAlgorithmError as exc:
    print(f"{exc.name!r} is not one of {exc.available}")
```

## Command line

Installing the package also installs a `thresher` command, so you can find a threshold
without writing any Python. Point it at a file with one row per sample — a score and a
ground-truth class:

```
$ thresher scores.csv
0.35
```

It prints the bare number to stdout and nothing else, so it pipes cleanly:

```
$ THRESHOLD=$(thresher scores.csv)
$ cat scores.csv | thresher -          # '-' reads stdin
```

Your data rarely arrives in exactly the expected shape, so the common adjustments are all
flags:

```
thresher scores.csv --labels 0,1                          # your classes are 0 and 1
thresher scores.csv -a grid                               # choose the algorithm yourself
thresher data.tsv --sep '\t' --no-header                  # tab-separated, no header row
thresher wide.csv --score-column pred --label-column y    # pick columns by name
thresher scores.csv -a ls -p n_jobs=4                     # pass algorithm parameters
```

`thresher --list-algorithms` shows the algorithms and their aliases, and `thresher --help`
documents every flag. Errors are reported in command-line terms rather than as Python
tracebacks, and exit codes follow the usual convention: `2` for a usage mistake, `1` when
the data itself cannot be optimized.

## Running on Ray

Thresher runs in two modes. By default everything happens in your process, exactly as it
always has. Pass `backend='ray'` and the counting is spread over a
[Ray](https://github.com/ray-project/ray) cluster instead:

```
pip install 'thresher-py[ray]'
```

```python
from thresher import Thresher

Thresher(backend="ray").optimize_threshold(scores, actual_classes)
```

or from the terminal:

```
thresher big.csv --backend ray
```

If you have already called `ray.init(address=...)`, Thresher joins that cluster. If you
have not, Ray starts a local one.

**A backend changes where the work happens, never the answer.** The map and reduce steps
are the same functions in both modes — the Ray backend ships them to workers rather than
reimplementing them — and there are tests asserting the two backends return *identical*
results, not merely close ones, including on data full of duplicates and ties.

### What gets distributed

The work is a map-reduce: shard the data once, count on each shard, add the partial counts
together.

| Algorithm | On Ray | Why |
|---|---|---|
| `exact` | **yes** | Needs only the class counts at each distinct score, so the per-row work shards cleanly and the driver is left with one pass over the distinct scores |
| `ls` | **yes** | "Score these candidates, keep the best" — each shard tallies every candidate, and the tallies add up |
| `grid` | **yes** | Same shape as `ls`, with candidates from the grid rather than the data |
| `sgrid`, `gen` | no | Each evaluation reads its own random subsample. Sharding would change which samples are drawn, and so the result |
| `sgd` | no | A sequential walk — each step depends on the one before it, so there is nothing to run in parallel |

The last three still work under `backend='ray'`; they simply run in-process. Nothing
silently changes its answer to become distributable.

### When it is worth it

Not for small data. Scheduling a shard costs far more than counting a few thousand rows,
so the backend keeps shards at 5,000 rows or more and will happily use a single shard for
a small input. Ray earns its keep when the data is large, when it already lives in a Ray
cluster, or when you are calling Thresher from inside a Ray application and want it to use
the resources you have.

For finer control, pass a configured backend instead of a name:

```python
from thresher.backends.ray_backend import RayBackend

Thresher(backend=RayBackend(num_shards=16)).optimize_threshold(scores, actual_classes)
```

> [!NOTE]
> Ray publishes wheels for Linux and Apple Silicon, but **not for macOS x86_64** — the
> `[ray]` extra cannot be installed on an Intel Mac. Use the default backend there.

## Performance tests

A very basic performance test (with 10 repeats, on a real-world [anonymized data](https://github.com/oskar-j/thresher/blob/main/examples/performance_test/milion_samples.7z) consisting of `10^6` rows) can be found in the Notebook [located here](https://github.com/oskar-j/thresher/blob/main/examples/performance_test/TresherPerformanceTest.ipynb).
Similar experiment, but with more iterations, was conducted in the file [TresherPerformanceTestExtended.ipynb](https://github.com/oskar-j/thresher/blob/main/examples/performance_test/TresherPerformanceTestExtended.ipynb). Both notebooks are older than the current algorithm set and are kept as a record of the measurements the size guidance above was drawn from.

For a head-to-head comparison of accuracy and runtime across all five algorithms, see
[Algorithm scores](#algorithm-scores) above — that one is reproducible from
`examples/benchmark.py` rather than recorded in a notebook.

## Future work

* adding more algorithms,
* publishing on conda,
* more heavy test loads,
* python docs.

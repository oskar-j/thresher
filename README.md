# Thresher - THRESHold EvaluatoR for Python

[![PyPI version](https://img.shields.io/pypi/v/thresher-py.svg)](https://pypi.org/project/thresher-py/)
[![Build](https://img.shields.io/github/actions/workflow/status/oskar-j/thresher/ci.yml?branch=main&label=build)](https://github.com/oskar-j/thresher/actions/workflows/ci.yml)
[![Downloads](https://img.shields.io/pepy/dt/thresher-py)](https://pepy.tech/project/thresher-py)
[![Stars](https://img.shields.io/github/stars/oskar-j/thresher)](https://github.com/oskar-j/thresher/stargazers)

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
import thresher

thresher.Thresher().optimize_threshold(scores, actual_classes)
```

That is the whole interface. Everything else in this README is about tuning what happens
underneath — which of the five search algorithms runs, and how hard it looks.

## Table of contents

- [Project description](#project-description)
  - [An oracle mechanism](#an-oracle-mechanism)
- [Implemented algorithms](#implemented-algorithms)
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

### An oracle mechanism

We implemented a meta-optimizer - an 'oracle' mechanism, which chooses a proper algorithm in regard to the provided data. This is the default behaviour, and can be controlled by changing the `algorithm` param of the `Thresher` constructor. See the source code of [oracle.py](https://github.com/oskar-j/thresher/blob/main/thresher/oracle.py) and [interface.py](https://github.com/oskar-j/thresher/blob/main/thresher/interface.py) for more details.

## Implemented algorithms

### Linear search

This is the most basic, iterative approach. Recommended for smaller datasets. For every _threshold_ present in the input (in the _scores_ list), we evaluate it by calculating the exact accuracy of _split_ produced by such threshold. Then, return the threshold which produce the most accurate split.

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

How the five compare on 2,000 rows, averaged over 5 seeds. Accuracy is relative to the
**exact** optimum — the best accuracy any single threshold could reach on that dataset,
computed independently by sweeping the sorted scores rather than by asking one of the
algorithms under test. 100% therefore means "found a cut-off as good as the best one that
exists", not merely "did well".

| Algorithm | Separable | Overlapping | Imbalanced | Time | Complexity |
|---|---|---|---|---|---|
| `ls` | 100.00% | 100.00% | 100.00% | 252 ms | O(n²) |
| `grid` | 99.82% | 99.98% | 100.00% | 18 ms | O(c·n) |
| `sgrid` | 99.57% | 97.84% | 99.70% | 1 ms | O(c·r·n) |
| `gen` | 99.62% | 98.23% | 88.64% | 110 ms | O(e·r·n) |
| `sgd` | 99.54% | 97.01% | 88.37% | 8 ms | O(i·r·n) |

Where _n_ is the number of scores, _c_ the grid candidates (`10**no_of_decimal_places + 1`,
so 101 by default), _r_ the `stoch_ratio` sample fraction, _e_ the genetic evaluations
(`population_size × number_of_generations × number_of_iterations`) and _i_ the sgd steps
(at most `num_of_iters`).

Only linear search grows with the *square* of the input, because it scores each of the
n-1 candidate midpoints against all n samples. Every other algorithm takes its candidate
count from its own parameters rather than from n, which is what keeps it linear — and what
the measured timings show:

| Algorithm | n=1,000 | n=4,000 | n=16,000 | growth per 4× |
|---|---|---|---|---|
| `ls` | 57 ms | 944 ms | 17,796 ms | ~16× |
| `grid` | 9 ms | 38 ms | 161 ms | ~4× |
| `sgrid` | <1 ms | 2 ms | 9 ms | ~4× |
| `gen` | 60 ms | 198 ms | 855 ms | ~4× |
| `sgd` | 5 ms | 18 ms | 67 ms | ~4× |

Three things worth reading out of this:

- **Linear search is exact**, and on small data it is cheap enough that nothing else is
  worth using. By 16,000 rows it already costs 18 seconds, and it is quadratic, so it only
  gets worse from there. That is why the oracle stops choosing it above 1,000 rows.
- **Grid search is the sweet spot** — within 0.2% of exact on every dataset here, at a
  fraction of the cost. `sgrid` gives up a little accuracy for another ~20× speedup.
- **`gen` and `sgd` struggle when one class is rare.** Both read only a small random
  subsample per evaluation, and the rarer the minority class, the less any subsample says
  about where the boundary lies. Prefer grid search on imbalanced data if you can afford
  it, or raise `stoch_ratio` for `gen`.

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

Lint, format and type-check with the same hooks CI runs:

```
uv run pre-commit run --all-files
```

Optionally install the git hook so those run on every commit:

```
uv run pre-commit install
```

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
Thresher(algorithm='auto',
         allow_parallel=True,
         verbose=False,
         progress_bar=False,
         labels=(0,1))
```

Here is a description of what does every particular parameter do:

* **algorithm** (default value: `'auto'`) - allows to manually choose the algorithm from the list of available algorithms.
Same effect can be achieved with running the method called `set_algorithm(algorithm_name)` on the `Thresher` instance.
The default value is 'auto', which means that the tool uses an oracle mechanism to manually choose a proper algorithm.
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
import thresher

t = thresher.Thresher()

print('Currently supported algorithms:')
print(t.get_supported_algorithms())

cases = [0.1, 0.3, 0.4, 0.7]
actual_labels = [-1, -1, 1, 1]

print(f'Optimization result: {t.optimize_threshold(cases, actual_labels)}')
```

See the [examples](https://github.com/oskar-j/thresher/tree/main/examples) directory for more sample code.

## Performance tests

A very basic performance test (with 10 repeats, on a real-world [anonymized data](https://github.com/oskar-j/thresher/blob/main/examples/performance_test/milion_samples.7z) consisting of `10^6` rows) can be found in the Notebook [located here](https://github.com/oskar-j/thresher/blob/main/examples/performance_test/TresherPerformanceTest.ipynb).
Similar experiment, but with more iterations, was conducted in the file [TresherPerformanceTestExtended.ipynb](https://github.com/oskar-j/thresher/blob/main/examples/performance_test/TresherPerformanceTestExtended.ipynb) to test the oracle.

For a head-to-head comparison of accuracy and runtime across all five algorithms, see
[Algorithm scores](#algorithm-scores) above — that one is reproducible from
`examples/benchmark.py` rather than recorded in a notebook.

## Future work

* adding more algorithms,
* publishing on conda,
* more heavy test loads,
* python docs.

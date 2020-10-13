# Thresher - THRESHold EvaluatoR for Python

A bare pandas implementation of a tool for finding the threshold which maximizes accuracy 
of `predict_proba` like-outputs (from e.g. `scikit-learn`), in regard to the provided ground truth (labels).

## Project description

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

### Implemented algorithms

### Linear search

### 2-dim Stochastic Gradient Descent

### Evolutionary algorithm

This is a simulation approach which uses an evolutionary algorithm. It works by simulating multiple generations of a "population" of candidate solutions. During every iteration of a single generation, algorithm stochasticly evaluates the candidate solution.

List of parameters to customize:
* `population_size` (default: 30)
* `number_of_generations` (default: 20)
* `number_of_iterations` (default: 10)
* `sus_factor` (default: 2)
* `stoch_ratio` (default: 0.02)
* `optimized_start` (default: True)

### Grid search

Added in version `0.1.2`.

### Stochastic Grid search

Added in version `0.1.2`.

## How to setup?

The process is rather straightforward, you just need to just whether to install 
from the sources (latest revision), or from the PyPI repository (stable release).

### Requirements

Tested with Python `3.7+`, on a standard Unix environment

### Installation

Installation from source:

```
pip install git+https://github.com/oskar-j/thresher.git
```

Stable release using the `pip` tool:

```
pip install thresher-py
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

## Future work

* adding more algorithms
* publishing on conda
* more heavy test loads
* python docs
* CI pipeline for automated tests

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

This is the most basic, iterative approach. Recommended for smaller datasets. For every _threshold_ present in the input (in the _scores_ list), we evaluate it by calculating the exact accuracy of _split_ produced by such threshold. Then, return the threshold which produce the most accurate split. 

### 2-dim Stochastic Gradient Descent

tbd

### Evolutionary algorithm

This is a simulation approach which uses an evolutionary algorithm. It works by simulating multiple generations of a "population" of candidate solutions. During every iteration of a single generation, algorithm stochasticly evaluates the candidate solution. After the end of a single generation, we remove the from the population least fit agents (solutions), and do the _crossover_ between the left solitions to produce new "offspring" candidate solutions. Moreover, they may mutate to provide additional random chance. 

List of parameters to customize:
* `population_size` (default: 30)
* `number_of_generations` (default: 20)
* `number_of_iterations` (default: 10)
* `sus_factor` (default: 2)
* `stoch_ratio` (default: 0.02)
* `optimized_start` (default: True)
* `mutation_chance` (default: 0.05)
* `mutation_factor` (default: 0.10)

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

## Custom parameters

It's possible to provide additional parameters in the `Thresher` constructor. 

```python
Thresher(algorithm='auto',
         verbose=False, 
         progress_bar=False,
         labels=(0,1))
```

Here is a description of what does every particular parameter do:

* **algorithm** (default value: `'auto'`) - allows to manually choose the algorithm from the list of available algorithms.
Same effect can be achieved with running the method called `set_algorithm(algorithm_name)` on the `Thresher` instance. 
The default value is 'auto', which means that the tool uses an oracle mechanism to manually choose a proper algorithm.
* **verbose** (default value: `False`) - enables verbosity
* **progress_bar** (default value: `False`) - shows a progress bar in the terminal (if supported by the algorithm)
* **labels** - necessary if your labels are different from `(-1, 1)` - first item from the tuple/list is a negative label, 
and the second item is a positive label

### Control parameters for the algorithms

tbd

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

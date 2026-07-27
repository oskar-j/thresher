# Getting started

## Install

```console
$ pip install thresher-py
```

Or with [uv](https://docs.astral.sh/uv/):

```console
$ uv add thresher-py
```

Python 3.10 or newer. The only runtime dependencies are numpy, pandas and click.

To spread the work over a cluster, ask for the extra as well:

```console
$ pip install 'thresher-py[ray]'
```

!!! note

    Ray publishes wheels for Linux and Apple Silicon, but not for macOS x86_64, so the
    `[ray]` extra cannot be installed on an Intel Mac.

## Find a threshold

Give it the scores your model produced and the classes they should have had:

```python
from thresher import Thresher

scores = [0.1, 0.3, 0.4, 0.7]
actual_classes = [-1, -1, 1, 1]

Thresher().optimize_threshold(scores, actual_classes)
# 0.35
```

The result is the cut-off that classifies the highest fraction of your samples correctly.
Where several thresholds tie, you get one of them.

## If your labels are not -1 and 1

Most datasets use `0` and `1`. Say so, and they are translated for you:

```python
Thresher(labels=(0, 1)).optimize_threshold(scores, [0, 0, 1, 1])
```

The negative class goes first.

## With scikit-learn

`optimize_threshold` takes the second column of `predict_proba` — the probability of the
positive class:

```python
from sklearn.linear_model import LogisticRegression
from thresher import Thresher

model = LogisticRegression().fit(X_train, y_train)
scores = model.predict_proba(X_validation)[:, 1]

threshold = Thresher(labels=(0, 1)).optimize_threshold(scores, y_validation)
predictions = (model.predict_proba(X_test)[:, 1] > threshold).astype(int)
```

Fit the threshold on data the model was not trained on, for the same reason you would not
measure accuracy on the training set.

## Choosing an algorithm

The default is the [exact sweep](algorithms.md#exact-sweep), and it is what you want unless
you have a specific reason otherwise. To pick another:

```python
Thresher(algorithm="grid").optimize_threshold(scores, actual_classes)
```

See [Algorithms](algorithms.md) for what each one does and how they compare.

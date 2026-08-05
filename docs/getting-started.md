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

## What you can pass in

Lists, `numpy` arrays and `pandas` Series are all read as they are — the array coming out
of `predict_proba` above is used where it lies rather than copied into a list first. Only
input that can be walked once, such as a generator, is collected up, because every
algorithm needs more than one pass over the data.

The difference shows up on [`hist`](algorithms.md#histogram-sweep), whose memory is meant
not to follow the input size. Optimizing 200,000 rows peaked at 12.2 MiB for an array
before 0.7.2 and at 17 KiB now — the same figure a list has always cost.

A Series is read through the array beneath it, so its index never enters into the answer:
a frame you filtered before slicing the column gives what the equivalent list would.

!!! note

    The result is always a plain `float`. Before 0.7.2 numpy input came back as an
    `np.float64` from some algorithms and a `float` from others.

## Choosing an algorithm

The default is the [exact sweep](algorithms.md#exact-sweep), and it is what you want unless
you have a specific reason otherwise. To pick another:

```python
Thresher(algorithm="grid").optimize_threshold(scores, actual_classes)
```

See [Algorithms](algorithms.md) for what each one does and how they compare.

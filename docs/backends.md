# Running on Ray

Thresher runs in two modes. By default everything happens in your process. Pass
`backend="ray"` and the counting is spread over a [Ray](https://github.com/ray-project/ray)
cluster instead.

```console
$ pip install 'thresher-py[ray]'
```

```python
from thresher import Thresher

Thresher(backend="ray").optimize_threshold(scores, actual_classes)
```

Or from a terminal:

```console
$ thresher big.csv --backend ray
```

If you have already called `ray.init(address=...)`, Thresher joins that cluster. If you have
not, Ray starts a local one.

!!! note

    Ray publishes wheels for Linux and Apple Silicon, but not for macOS x86_64, so the
    `[ray]` extra cannot be installed on an Intel Mac.

## A backend never changes the answer

Only where the work happens. The map and reduce steps are the same functions in both modes —
the Ray backend ships them to workers rather than reimplementing them — and the reduce step
is addition, which is order-independent, so shard boundaries cannot influence the result.
There are tests asserting the two backends return **identical** results, not merely close
ones, including on data full of duplicates and ties.

## What gets distributed

The work is a map-reduce: shard the data once, count on each shard, add the partial counts
together.

| Algorithm | On Ray | Why |
|---|---|---|
| `exact` | **yes** | needs only the class counts at each distinct score, so the per-row work shards cleanly |
| `ls` | **yes** | "score these candidates, keep the best" — each shard tallies every candidate, and tallies add up |
| `grid` | **yes** | same shape, with candidates from the grid rather than the data |
| `sgrid`, `gen` | no | each evaluation draws its own random subsample; sharding would change which samples are read, and so the result |
| `sgd` | no | a sequential walk — each step depends on the one before it |

The last three still work under `backend="ray"`; they simply run in-process. Nothing
silently changes its answer in order to become distributable.

## When it is worth it

Not for small data. Scheduling a shard costs far more than counting a few thousand rows, so
shards are kept at 5,000 rows or more and a small input happily uses a single shard. Ray
earns its keep when the data is large, when it already lives in a cluster, or when you are
calling Thresher from inside a Ray application.

For finer control, pass a configured backend rather than a name:

```python
from thresher.backends.ray_backend import RayBackend

Thresher(backend=RayBackend(num_shards=16)).optimize_threshold(scores, actual_classes)
```

# Running in parallel

Thresher runs in three modes. By default everything happens in your process. A backend
spreads the same counting wider: `mp` over the CPU cores of the machine you are on, `ray`
over a cluster.

| Backend | Needs | Use it when |
|---|---|---|
| `local` | nothing, and it is the default | anything that finishes fast enough as it is |
| `mp` | nothing — `multiprocessing` is in the standard library | one machine, several cores, data large enough to be worth dividing |
| `ray` | `pip install 'thresher-py[ray]'` | the data is large, or already lives in a Ray cluster |

## The multiprocessing backend

Added in `0.7.0`.

```python
from thresher import Thresher

def main():
    Thresher(backend="mp").optimize_threshold(scores, actual_classes)

if __name__ == "__main__":
    main()
```

Or from a terminal, where no guard is needed:

```console
$ thresher big.csv --backend mp
```

!!! warning "A script using `mp` must guard its entry point"

    On macOS and Windows — and on newer Pythons elsewhere — worker processes re-import the
    script that started them. Without an `if __name__ == "__main__":` guard, every worker
    re-runs your module-level code and starts workers of its own.

    Before `0.7.0` that hung the process with no error at all, and the run had to be
    killed. It now raises `ParallelBootstrapError` within a second or two and explains the
    fix. The default `local` backend needs no guard, and neither does the `thresher`
    command.

To choose the number of processes, pass a configured backend rather than a name:

```python
from thresher.backends import MultiprocessingBackend

Thresher(backend=MultiprocessingBackend(num_workers=4)).optimize_threshold(scores, actual_classes)
```

`num_workers=-1` means every processor bar one. Asking for more than the machine has is
clamped to what it has; `0` or anything below `-1` names no sensible process count and is
refused.

### `n_jobs` is the same thing

Linear search's `n_jobs` parameter predates backends and now runs on this backend, so the
two cannot disagree about what `-1` means or drift apart in behaviour:

```python
Thresher(algorithm="ls", algorithm_params={"n_jobs": 4})   # same as backend="mp"
```

Two things changed with it in `0.7.0`. Parallelising no longer changes the answer — the
old parallel path scored the raw scores as thresholds where the sequential path scores the
midpoints between them, so the two disagreed on the same data. And a backend chosen
explicitly wins over `n_jobs`, rather than the two competing for the same cores.

## Running on Ray

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
    `[ray]` extra cannot be installed on an Intel Mac. The `mp` backend works there.

## A backend never changes the answer

Only where the work happens. The map and reduce steps are the same functions in every mode —
the `mp` and `ray` backends ship them to workers rather than reimplementing them — and the
reduce step is addition, which is order-independent, so shard boundaries cannot influence
the result. There are tests asserting the backends return **identical** results, not merely
close ones, including on data full of duplicates and ties.

## What gets distributed

The work is a map-reduce: shard the data once, count on each shard, add the partial counts
together.

| Algorithm | Parallel | Why |
|---|---|---|
| `exact` | **yes** | needs only the class counts at each distinct score, so the per-row work shards cleanly |
| `ls` | **yes** | "score these candidates, keep the best" — each shard tallies every candidate, and tallies add up |
| `grid` | **yes** | same shape, with candidates from the grid rather than the data |
| `sgrid`, `gen` | no | each evaluation draws its own random subsample; sharding would change which samples are read, and so the result |
| `sgd` | no | a sequential walk — each step depends on the one before it |

The last three still work under `backend="mp"` or `backend="ray"`; they simply run
in-process. Nothing silently changes its answer in order to become distributable.

## When it is worth it

Not for small data. Handing work to another process costs far more than counting a few
thousand rows, so shards are kept above a floor — 2,000 rows for `mp`, 5,000 for Ray, whose
scheduling costs more — and a smaller input is simply counted in this process.

Between the two: `mp` needs no setup and uses the cores you already have. Ray earns its keep
when the data is large, when it already lives in a cluster, or when you are calling Thresher
from inside a Ray application.

# Reporting and progress

Added in `0.8.0`. Everything this package has to say goes through
[loguru](https://loguru.readthedocs.io/), at a level per message, and one setting decides
how much of it you see.

Before that, most of it went to `print()` behind an `if verbose:` — on stdout, unformatted
and unroutable — and the rest went to the standard library's `logging`, from two modules
only. The same run reported half of itself one way and half the other.

## Verbosity

```python
from thresher import Thresher

Thresher(verbosity="info").optimize_threshold(scores, actual_classes)
```

| Level | What you get |
|---|---|
| `debug` | every step: each sgd iteration, each generation of the evolutionary search |
| `info` | the shape of the run: the algorithm, the data, the answer and how good it is |
| `warning` | **the default** — only what you should know about a run going ahead anyway |
| `error` | nothing at all, in practice: this package raises rather than logs its failures |
| `critical` | the same |

Nothing is emitted at the default level unless something is worth saying. In practice that
is one message: the warning that the chosen algorithm is slow for this much data.

## Three ways to set it, in increasing precedence

```python
import thresher

thresher.set_verbosity("info")                  # until it is set again
thresher.Thresher(verbosity="debug")            # this instance's runs
with thresher.verbosity("debug"):               # this block
    ...
```

The instance setting applies for the duration of each `optimize_threshold` call and to
nothing else, so **two `Thresher` objects in one process can differ** — one verbose, one
silent, at the same time. Neither logging system offers that on its own: both hold their
level globally. The level here is held in a `contextvars.ContextVar`, so threads and async
tasks do not read each other's setting either.

`verbose=True` is the option `verbosity` replaced and still works, meaning
`verbosity="debug"`.

An unknown level is a `ConfigurationError` where you give it — when the `Thresher` is
built, not several seconds into a long run — for the same reason a mistyped option name
is: it leaves you believing you configured a run you did not.

## Sending the records somewhere

This package installs no loguru handler. Records go wherever your application has pointed
loguru, which for an application that has configured nothing is loguru's own stderr
handler.

```python
from loguru import logger

logger.remove()                                    # drop the default handler
logger.add("run.log", level="INFO")                # and keep the records
logger.disable("thresher")                         # or silence this package entirely
```

`logger.disable("thresher")` works because nothing here writes to a console on its own —
every message goes through the global logger, which is what `disable` gates.

### If your application uses the standard library

`logging.getLogger("thresher").setLevel(...)` reached these records until `0.8.0`. One
call brings that back:

```python
import logging
import thresher

thresher.propagate_to_logging()
logging.getLogger("thresher").setLevel(logging.WARNING)
```

Every record is then handed to the `logging` logger named after the module that emitted it
— `thresher.dispatch`, `thresher.algs.exact.compute` — so the existing hierarchy under
`thresher` selects them as it always did.

It is off by default, and deliberately: an application with loguru *and* `logging` both
writing to a console would print every record twice, and a duplicate is the harder problem
to work out from the outside.

## Progress bars

```python
Thresher(progress_bar=True).optimize_threshold(scores, actual_classes)
```

```console
$ thresher big.csv --progress
```

Drawn by [tqdm](https://tqdm.github.io/) where it is installed and by a built-in bar where
it is not:

```console
pip install 'thresher-py[progress]'
```

The bar is formatted the same way either way — the percentage to one decimal place, then
the counts — so installing the extra changes how a run looks and nothing else about it.
The one difference is timing: tqdm coalesces redraws arriving within a tenth of a second,
and the built-in bar redraws on every step.

### Where a bar is drawn, and where it is not

A bar is a thing you watch, so it is drawn where somebody is watching:

- **On stderr, never stdout.** The command line prints the threshold on stdout so it can
  be piped onward. The built-in bar wrote there until `0.8.0`, which went unnoticed only
  because the command line had no way to ask for a bar.
- **Not from a worker.** The `mp` and `ray` backends count in other processes, where
  several bars would interleave into nonsense on one terminal.
- **Not from Spark.** `SparkThresher` has no `progress_bar` option: the counting happens
  in the cluster, and the driver's share of it is a sweep over a few thousand bins.
- **Not at `debug`.** The log and the bar write to the same stream, so together they
  produce a bar interrupted by log lines. The level wins.
- **Not by `sgd`.** It stops when it stops improving rather than after a known number of
  steps, so there is no proportion of the job done for a bar to show.

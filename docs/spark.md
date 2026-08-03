# Running on Spark

[Running on Ray](backends.md) distributes the library's ordinary API: you still hand it two
sequences that are already in memory. That is the wrong shape for data sitting in HDFS, S3
or a Delta table — collecting a billion rows onto one machine to sort them is exactly what
having a cluster is meant to avoid.

`SparkThresher` takes a DataFrame and two column names instead, and never collects the rows.

```console
$ pip install 'thresher-py[spark]'
```

```python
from thresher.spark import SparkThresher

df = spark.read.parquet("s3://predictions/2026-08/")

threshold = SparkThresher().optimize_threshold(df, score_col="probability", label_col="label")
```

Everything else means what it means elsewhere in the library:

```python
SparkThresher(
    "hist",
    {"no_of_bins": 4096},
    labels=(0, 1),
    verbose=True,
).optimize_threshold(df)
```

The column names default to `"score"` and `"label"`, and nothing is assumed about the rest
of the DataFrame — extra columns are simply not read.

!!! note

    PySpark ships as a 450 MB sdist carrying the Spark distribution itself, so installing
    the `[spark]` extra is slow. It also needs a JVM — Java 17 or newer for PySpark 4.x —
    which pip cannot install for you.

## How the work is split

The problem reduces to a map-reduce, and this is that reduction taken literally. Spark
performs one `groupBy` and a pair of sums, so what crosses the network is a shuffle of
counts rather than of rows. What comes back to the driver is a summary whose size is set by
the *resolution* you asked for, not by the number of rows you have:

```
a billion rows  ->  [ Spark: group and count ]  ->  ~1,024 rows  ->  [ driver: sweep ]
```

The driver then runs the sweep — `sweep_bins` or `sweep_class_counts`, the same functions
the in-memory algorithms call, not a second implementation of them — over that summary.

## Spark never changes the answer

Only where the counting happens. This is the same rule the Ray backend rests on, and it
holds for the same reason: the reduce step is addition, which does not care how the data
was partitioned, and the deciding step runs on a summary that is identical whichever way
the rows were split.

`tests/test_spark.py` asserts a Spark run returns the same `float` as the in-memory run —
not close, the same — including on data full of ties, and that repartitioning the DataFrame
to 1, 3 or 8 partitions does not move the result.

## What runs on Spark

| Algorithm | On Spark | Summary sent to the driver |
|---|---|---|
| [`hist`](algorithms.md#histogram-sweep) (default here) | **yes** | One row per bin — `no_of_bins` rows however large the input |
| [`exact`](algorithms.md#exact-sweep) | **yes** | One row per *distinct* score |
| `ls`, `grid`, `sgrid`, `gen`, `sgd` | no | — |

`hist` is the default for Spark, unlike everywhere else in the library, and it is usually
the one to want: its summary is bounded by the bin count, so a billion rows cost the driver
exactly what a million do. Its error is bounded by one bin width rather than being
statistical, and it is deterministic.

`exact` is available and is exactly what its name promises, but it groups by distinct
score, so the summary is one row per distinct value. That is cheap for rounded
probabilities and expensive for 64-bit floats that are all different; it logs a warning
once that count passes a million.

The other five are refused rather than quietly run on a sample or on the driver:

```pycon
>>> SparkThresher("sgd")
Traceback (most recent call last):
    ...
thresher.exceptions.ConfigurationError: 'sgd' cannot run on Spark. Available here: hist, exact. ...
```

`ls` is `O(n²)` in candidates, and `sgrid`, `gen` and `sgd` each draw their own random
subsamples — distributing those would change *which* samples are read, and so the answer,
which is the one thing a change of venue is not allowed to do. Run them through
[`Thresher`](api/thresher.md) on data that fits in memory.

## Reading a Spark DataFrame yourself

If your data does fit in memory and you only want the threshold, there is nothing wrong
with collecting two columns and using the ordinary API:

```python
rows = df.select("probability", "label").collect()
Thresher().optimize_threshold([r.probability for r in rows], [r.label for r in rows])
```

`SparkThresher` exists for when that `collect()` is the problem.

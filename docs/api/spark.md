# Spark

Finding a threshold over a Spark DataFrame. See [Running on Spark](../spark.md) for the
guide.

Importing this module does **not** import PySpark, so it is safe to reference from code
that may run without the `[spark]` extra installed. PySpark is imported when a
`SparkThresher` is constructed, and its absence is reported as a
[`BackendDependencyError`](exceptions.md).

## The entry point

::: thresher.spark.SparkThresher

## The module

::: thresher.spark
    options:
      members: false

## The sweeps it reuses

The deciding half of each supported algorithm is a plain function over counts, which is
what makes running it on a summary from the cluster identical to running it in memory.

::: thresher.algs.histogram.compute.sweep_bins

::: thresher.algs.exact.compute.sweep_class_counts

"""Find a threshold over a Spark DataFrame, without bringing the data to the driver.

The other entry points take a sequence in memory. That is the wrong shape for data living
in HDFS or S3: collecting a billion rows to one machine to sort them defeats the point of
having a cluster.

This does the counting where the data already is. Spark performs one aggregation - a
`groupBy` and a pair of sums - and returns a summary bounded by the *resolution* rather
than by the row count. The driver then runs exactly the same sweep the in-memory
algorithms use, over that summary.

    a billion rows  ->  [Spark: group and count]  ->  ~1,024 rows  ->  [driver: sweep]

Two algorithms fit that shape:

`hist`
    Groups by bin index, so the summary is `no_of_bins` rows however large the input.
    Nothing else here is bounded that way, which makes it the one to reach for.

`exact`
    Groups by distinct score, so the summary is one row per distinct score. Exact, and
    fine when scores are rounded probabilities; a warning is logged when the distinct
    count is large enough for the collect to be the expensive part.

The rest are not offered, for the same reason the Ray backend does not distribute them: a
backend may change where the work happens, never the answer. `ls` is `O(n²)` in candidates,
and `sgrid`, `gen` and `sgd` each draw their own random subsamples, so sharding would
change which samples are read and therefore the result.

PySpark is an optional dependency: `pip install 'thresher-py[spark]'`. It also needs a JVM,
which pip cannot install for you.
"""

import logging
from typing import TYPE_CHECKING, Any

from thresher import algorithm
from thresher.algs.exact.compute import sweep_class_counts
from thresher.algs.histogram.compute import no_of_bins_default, sweep_bins
from thresher.exceptions import (
    BackendDependencyError,
    ConfigurationError,
    EmptyInputError,
    InsufficientDataError,
    SingleClassError,
)
from thresher.utils import NEGATIVE_LABEL, POSITIVE_LABEL, get_or_default

if TYPE_CHECKING:  # pragma: no cover - imported for annotations only
    from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)

#: The algorithms whose work is an aggregation, and so can run in the cluster unchanged.
SUPPORTED_ALGORITHMS = ("hist", "exact")

#: Collecting more distinct scores than this is worth mentioning: at that point `exact` is
#: shipping a large result to the driver, and `hist` would do the same job in bounded space.
DISTINCT_SCORE_WARNING = 1_000_000

PYSPARK_MISSING = (
    "The Spark interface needs PySpark installed: pip install 'thresher-py[spark]'. "
    "It also needs a JVM on the machine, which pip cannot provide."
)

NOT_DISTRIBUTABLE = (
    "{name!r} cannot run on Spark. Available here: {available}. The others either cost "
    "O(n^2) in candidates or draw their own random subsamples, so distributing them would "
    "change the answer rather than only where it is computed - run those through "
    "Thresher() on data that fits in memory."
)


def _require_pyspark() -> Any:
    """Import PySpark's function namespace, or explain how to get it.

    Returns:
        The `pyspark.sql.functions` module.

    Raises:
        BackendDependencyError: if PySpark is not installed. It is an `ImportError`.
    """
    try:
        from pyspark.sql import functions
    except ImportError as exc:  # pragma: no cover - only without PySpark installed
        raise BackendDependencyError(PYSPARK_MISSING) from exc
    return functions


class SparkThresher:
    """Find the optimal threshold over a Spark DataFrame.

    Mirrors `Thresher`, but takes a DataFrame and the names of two columns instead of two
    sequences, and never collects the rows.

    Example:
        >>> from thresher.spark import SparkThresher
        >>> SparkThresher().optimize_threshold(df, "probability", "label")  # doctest: +SKIP
        0.4306640625
    """

    def __init__(
        self,
        algorithm_name: str = "hist",
        algorithm_params: dict[str, Any] | None = None,
        labels: tuple[Any, Any] | None = None,
        verbose: bool = False,
    ) -> None:
        """Configure the search.

        Args:
            algorithm_name: `'hist'` (the default) or `'exact'`, or any of their synonyms.
                `hist` returns a summary bounded by its bin count, so it is the one that
                stays cheap however large the data is.
            algorithm_params: passed through to the algorithm. `hist` reads `no_of_bins`
                (default 1024).
            labels: your two class labels, negative first, if they are not -1 and 1 -
                for example `(0, 1)`.
            verbose: log what is being aggregated.

        Raises:
            UnknownAlgorithmError: if the name matches no algorithm at all.
            ConfigurationError: if it names an algorithm that cannot run as an
                aggregation. Both are `ValueError`.
            BackendDependencyError: if PySpark is not installed.
        """
        _require_pyspark()

        resolved = algorithm.retrieve_by_alias(algorithm_name)
        if resolved.id not in SUPPORTED_ALGORITHMS:
            raise ConfigurationError(
                NOT_DISTRIBUTABLE.format(name=algorithm_name, available=", ".join(SUPPORTED_ALGORITHMS))
            )

        self.algorithm = resolved
        self.algorithm_params = algorithm_params or {}
        self.labels = labels
        self.verbose = verbose

    def optimize_threshold(
        self, df: "DataFrame", score_col: str = "score", label_col: str = "label"
    ) -> float:
        """Find the threshold that classifies the most rows correctly.

        Args:
            df: the DataFrame holding the scores and their true classes. It is read once
                or twice depending on the algorithm, and never collected.
            score_col: name of the column holding the scores.
            label_col: name of the column holding the ground-truth classes.

        Returns:
            The threshold, computed identically to what the in-memory algorithm would
            return for the same data.

        Raises:
            EmptyInputError: if the DataFrame has no rows.
            SingleClassError: if only one class is present, leaving nothing to separate.
            InsufficientDataError: if the aggregation came back empty.
        """
        functions = _require_pyspark()

        negative_label, positive_label = self.labels or (NEGATIVE_LABEL, POSITIVE_LABEL)
        is_positive = functions.col(label_col) == functions.lit(positive_label)
        is_negative = functions.col(label_col) == functions.lit(negative_label)

        # One pass for the shape of the data: how many rows, of which classes, over what
        # range. Everything after this is decided from counts.
        summary = df.agg(
            functions.count(functions.lit(1)).alias("rows"),
            functions.sum(is_positive.cast("long")).alias("positives"),
            functions.sum(is_negative.cast("long")).alias("negatives"),
            functions.min(functions.col(score_col)).alias("lowest"),
            functions.max(functions.col(score_col)).alias("highest"),
        ).first()

        if summary is None or not summary["rows"]:
            raise EmptyInputError

        positives = summary["positives"] or 0
        negatives = summary["negatives"] or 0
        if positives == 0 or negatives == 0:
            raise SingleClassError(positive_label if positives else negative_label)

        if self.verbose:
            logger.info(
                "Aggregating %s rows over [%s, %s] with %s.",
                f"{summary['rows']:,}",
                summary["lowest"],
                summary["highest"],
                self.algorithm.full_name,
            )

        if self.algorithm.id == "hist":
            return self._binned(df, score_col, is_positive, summary)
        return self._by_distinct_score(df, score_col, is_positive)

    def _binned(self, df: "DataFrame", score_col: str, is_positive: Any, summary: Any) -> float:
        """Group by bin index and sweep the bins.

        The summary Spark returns has one row per bin, so its size is set by `no_of_bins`
        and not by the input at all.

        Args:
            df: the source DataFrame.
            score_col: name of the score column.
            is_positive: a column expression that is true for the positive class.
            summary: the first-pass aggregate, carrying the score range.

        Returns:
            The best threshold expressible on a bin edge.
        """
        functions = _require_pyspark()

        bins: int = get_or_default(self.algorithm_params, "no_of_bins", no_of_bins_default)
        if bins < 1:
            raise InsufficientDataError(f"no_of_bins must be at least 1, got {bins}.")

        lowest = float(summary["lowest"])
        span = float(summary["highest"]) - lowest

        if span == 0:
            # Every score is identical, so there is one bin and nothing to place inside it.
            index_expr = functions.lit(0)
        else:
            scaled = (functions.col(score_col) - functions.lit(lowest)) / functions.lit(span)
            # The largest score would land one past the end, so it shares the last bin.
            index_expr = functions.least(
                functions.floor(scaled * functions.lit(bins)), functions.lit(bins - 1)
            )

        rows = (
            df.select(index_expr.alias("bin"), is_positive.cast("long").alias("positive"))
            .groupBy("bin")
            .agg(
                functions.sum("positive").alias("positives"),
                functions.count(functions.lit(1)).alias("total"),
            )
            .collect()
        )

        negatives_per_bin = [0] * bins
        positives_per_bin = [0] * bins
        for row in rows:
            index = int(row["bin"])
            positives_here = int(row["positives"] or 0)
            positives_per_bin[index] = positives_here
            negatives_per_bin[index] = int(row["total"]) - positives_here

        threshold, _ = sweep_bins(negatives_per_bin, positives_per_bin, lowest=lowest, span=span)
        return threshold

    def _by_distinct_score(self, df: "DataFrame", score_col: str, is_positive: Any) -> float:
        """Group by distinct score and sweep those counts.

        Exact, because no score is merged with any other - but the summary is one row per
        distinct score, so it is only cheap while that number is.

        Args:
            df: the source DataFrame.
            score_col: name of the score column.
            is_positive: a column expression that is true for the positive class.

        Returns:
            The best threshold, matching what `exact` returns in memory.
        """
        functions = _require_pyspark()

        rows = (
            df.select(
                functions.col(score_col).alias("score"),
                is_positive.cast("long").alias("positive"),
            )
            .groupBy("score")
            .agg(
                functions.sum("positive").alias("positives"),
                functions.count(functions.lit(1)).alias("total"),
            )
            .collect()
        )

        if len(rows) > DISTINCT_SCORE_WARNING:
            logger.warning(
                "'exact' collected %s distinct scores to the driver. The 'hist' algorithm "
                "answers the same question from a fixed number of bins, whatever the input.",
                f"{len(rows):,}",
            )

        counts: dict[float, tuple[int, int]] = {}
        for row in rows:
            positives_here = int(row["positives"] or 0)
            counts[float(row["score"])] = (int(row["total"]) - positives_here, positives_here)

        threshold, _ = sweep_class_counts(counts)
        return threshold

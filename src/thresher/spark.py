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

from typing import TYPE_CHECKING, Any

from thresher import algorithm, log
from thresher.algs.exact.compute import sweep_class_counts
from thresher.algs.histogram.compute import no_of_bins_default, sweep_bins
from thresher.dispatch import validate_algorithm_params
from thresher.exceptions import (
    BackendDependencyError,
    ConfigurationError,
    EmptyInputError,
    InsufficientDataError,
    MissingLabelsError,
    SingleClassError,
    UndefinedScoresError,
    UnexpectedLabelsError,
)
from thresher.utils import NEGATIVE_LABEL, POSITIVE_LABEL, get_or_default

if TYPE_CHECKING:  # pragma: no cover - imported for annotations only
    from pyspark.sql import DataFrame

#: The algorithms whose work is an aggregation, and so can run in the cluster unchanged.
SUPPORTED_ALGORITHMS = ("hist", "exact")

#: How many offending label values to name when refusing a frame. Enough to show the
#: shape of the problem without collecting an unbounded set to the driver.
UNEXPECTED_LABEL_SAMPLE = 5

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


def _is_undefined(functions: Any, df: "DataFrame", score_col: str) -> Any:
    """Build the expression that is true for a score no threshold can be placed against.

    Null and NaN both qualify, and both used to pass through unnoticed. Spark's `least`
    skips nulls, so `least(floor(null), bins - 1)` returned the last bin and a row with no
    score at all was counted as though it held the highest one. NaN was worse: Spark
    orders it above everything, so it became the maximum, the span became NaN, and every
    candidate evaluated to NaN - collapsing the answer to "classify everything positive".

    Args:
        functions: the `pyspark.sql.functions` module.
        df: the DataFrame, read for the column's declared type.
        score_col: name of the score column.

    Returns:
        A column expression, true where the score is unusable.
    """
    undefined = functions.col(score_col).isNull()
    # isnan() is only defined for the floating-point types; asking it of an integer or
    # decimal column is an analysis error rather than a false, and those types have no NaN
    # to find in the first place.
    if df.schema[score_col].dataType.simpleString() in ("double", "float"):
        undefined = undefined | functions.isnan(functions.col(score_col))
    return undefined


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
        verbosity: str | None = None,
    ) -> None:
        """Configure the search.

        Args:
            algorithm_name: `'hist'` (the default) or `'exact'`, or any of their synonyms.
                `hist` returns a summary bounded by its bin count, so it is the one that
                stays cheap however large the data is.
            algorithm_params: passed through to the algorithm. `hist` reads `no_of_bins`
                (default 1024); `exact` reads none. A key the chosen algorithm does not
                read raises `ConfigurationError` rather than being ignored.
            labels: your two class labels, negative first, if they are not -1 and 1 -
                for example `(0, 1)`.
            verbose: log what is being aggregated. `verbose=True` means
                `verbosity='debug'`, the same as it does on `Thresher`.
            verbosity: how much this instance reports - `'debug'`, `'info'`, `'warning'`
                (the default), `'error'` or `'critical'`. Applies for the duration of each
                `optimize_threshold` call.

        Note:
            No progress bar is offered, deliberately. The counting happens on executors,
            where there is nobody watching a terminal, and the driver's share of the work
            is a sweep over a few thousand bins - see `thresher.progress`.

        Raises:
            UnknownAlgorithmError: if the name matches no algorithm at all.
            ConfigurationError: if it names an algorithm that cannot run as an
                aggregation, or if `verbosity` names no known level. Both are `ValueError`.
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
        # The same check the in-memory interface runs: a key this algorithm does not read
        # would otherwise leave the default in place across an entire cluster run.
        validate_algorithm_params(resolved, self.algorithm_params)
        self.labels = labels
        self.verbosity = log.resolve_verbosity(verbosity, verbose)

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
            UndefinedScoresError: if any score is null or NaN.
            MissingLabelsError: if any label is null.
            UnexpectedLabelsError: if any label is neither of the two declared classes.
            SingleClassError: if only one class is present, leaving nothing to separate.
            InsufficientDataError: if the aggregation came back empty.

        Note:
            Those refusals are the same ones the in-memory path makes, deliberately. Until
            0.7.1 none of them existed here: a row whose label matched neither class - a
            null, a third value, a typo - was counted as a negative simply because it was
            not a positive, and a null or NaN score was quietly filed in the top bin. Both
            returned a plausible threshold computed from data the in-memory path would have
            refused outright.
        """
        with log.verbosity(self.verbosity):
            return self._optimize_threshold(df, score_col, label_col)

    def _optimize_threshold(self, df: "DataFrame", score_col: str, label_col: str) -> float:
        """Aggregate, refuse what cannot be swept, and sweep the rest.

        Split from `optimize_threshold` so that the level this instance was built with
        covers the whole run, including the refusals, without indenting all of it.

        Args:
            df: the DataFrame holding the scores and their true classes.
            score_col: name of the column holding the scores.
            label_col: name of the column holding the ground-truth classes.

        Returns:
            The threshold.
        """
        functions = _require_pyspark()

        negative_label, positive_label = self.labels or (NEGATIVE_LABEL, POSITIVE_LABEL)
        is_positive = functions.col(label_col) == functions.lit(positive_label)
        is_negative = functions.col(label_col) == functions.lit(negative_label)

        # One pass for the shape of the data: how many rows, of which classes, over what
        # range, and how much of it is unusable. Everything after this is decided from
        # counts, including the refusals - which is why the counts are gathered together
        # rather than a pass at a time.
        summary = df.agg(
            functions.count(functions.lit(1)).alias("rows"),
            functions.sum(is_positive.cast("long")).alias("positives"),
            functions.sum(is_negative.cast("long")).alias("negatives"),
            functions.sum(_is_undefined(functions, df, score_col).cast("long")).alias("bad_scores"),
            functions.sum(functions.col(label_col).isNull().cast("long")).alias("null_labels"),
            functions.min(functions.col(score_col)).alias("lowest"),
            functions.max(functions.col(score_col)).alias("highest"),
        ).first()

        if summary is None or not summary["rows"]:
            raise EmptyInputError

        self._reject_unusable_rows(df, summary, label_col, is_positive, is_negative)

        positives = summary["positives"] or 0
        negatives = summary["negatives"] or 0
        if positives == 0 or negatives == 0:
            raise SingleClassError(positive_label if positives else negative_label)

        log.info(
            "Aggregating {} rows over [{}, {}] with {}.",
            f"{summary['rows']:,}",
            summary["lowest"],
            summary["highest"],
            self.algorithm.full_name,
        )

        if self.algorithm.id == "hist":
            return self._binned(df, score_col, is_positive, summary)
        return self._by_distinct_score(df, score_col, is_positive)

    def _reject_unusable_rows(
        self, df: "DataFrame", summary: Any, label_col: str, is_positive: Any, is_negative: Any
    ) -> None:
        """Refuse the rows the in-memory path would refuse, in the same order.

        The class counts come from equality against the two declared labels, so anything
        matching neither - a null, a third class, a string where a number was meant - is
        simply absent from both. Nothing downstream notices: it lands in the negative
        count by omission, because the sweep works from `rows - positives`. Comparing the
        two counts against the row count is what turns that silence into a refusal, and it
        costs nothing extra because all three came from the one aggregation.

        Args:
            df: the source DataFrame, read again only to name offending labels.
            summary: the first-pass aggregate.
            label_col: name of the label column.
            is_positive: a column expression true for the positive class.
            is_negative: a column expression true for the negative class.

        Returns:
            None. This is a guard - it either passes silently or raises.

        Raises:
            UndefinedScoresError, MissingLabelsError, UnexpectedLabelsError: for each case
                in turn. All are `InvalidInputError`, and so also `ValueError`.
        """
        bad_scores = summary["bad_scores"] or 0
        if bad_scores:
            raise UndefinedScoresError(bad_scores)

        null_labels = summary["null_labels"] or 0
        if null_labels:
            raise MissingLabelsError(null_labels)

        matched = (summary["positives"] or 0) + (summary["negatives"] or 0)
        if matched != summary["rows"]:
            # Only now is a second pass worth it: naming the values beats reporting a
            # count, and this one runs on the way out.
            offending = (
                df.select(label_col)
                .where(~(is_positive | is_negative))
                .distinct()
                .limit(UNEXPECTED_LABEL_SAMPLE)
                .collect()
            )
            raise UnexpectedLabelsError([row[label_col] for row in offending])

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

        threshold, _ = sweep_bins(
            negatives_per_bin, positives_per_bin, lowest=lowest, highest=float(summary["highest"])
        )
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
            log.warning(
                "'exact' collected {} distinct scores to the driver. The 'hist' algorithm "
                "answers the same question from a fixed number of bins, whatever the input.",
                f"{len(rows):,}",
            )

        counts: dict[float, tuple[int, int]] = {}
        for row in rows:
            positives_here = int(row["positives"] or 0)
            counts[float(row["score"])] = (int(row["total"]) - positives_here, positives_here)

        threshold, _ = sweep_class_counts(counts)
        return threshold

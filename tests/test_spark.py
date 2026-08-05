"""The Spark interface.

The property that matters, and what most of these assert: Spark changes *where* the
counting happens, never the answer. A run over a DataFrame must return exactly what the
in-memory algorithm returns for the same data - not close, the same float.

These need PySpark and a JVM, and skip themselves where either is missing.
"""

import random
from collections.abc import Callable, Iterator
from typing import Any

import pytest

import thresher
from thresher.exceptions import (
    ConfigurationError,
    EmptyInputError,
    InsufficientDataError,
    MissingLabelsError,
    SingleClassError,
    ThresherError,
    UndefinedScoresError,
    UnexpectedLabelsError,
    UnknownAlgorithmError,
)

Dataset = tuple[list[float], list[int]]
DatasetFactory = Callable[..., Dataset]

pyspark = pytest.importorskip("pyspark", reason="PySpark is not installed")

# Safe at module level only after the guard above, which is why the older tests each
# import it themselves.
from thresher.spark import SparkThresher  # noqa: E402

DISTRIBUTABLE = ["hist", "exact"]


@pytest.fixture(scope="module")
def spark() -> Iterator[Any]:
    """A two-core local Spark session, started once for this module.

    The bind address is pinned because Spark otherwise picks an interface it cannot always
    bind to on a developer machine, and fails after sixteen retries.
    """
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]")
        .appName("thresher-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture
def frame(spark: Any) -> Callable[[list[float], list[int]], Any]:
    """Build a DataFrame from parallel score and class sequences."""

    def _make(scores: list[float], actual_classes: list[int]) -> Any:
        return spark.createDataFrame(list(zip(scores, actual_classes, strict=True)), ["score", "label"])

    return _make


@pytest.fixture
def noisy() -> DatasetFactory:
    """Overlapping classes, so the best threshold still gets some wrong."""

    def _make(n: int, seed: int = 0, flip: float = 0.15) -> Dataset:
        rng = random.Random(seed)
        scores = [rng.random() for _ in range(n)]
        labels = [1 if score > 0.5 else -1 for score in scores]
        return scores, [(-lab if rng.random() < flip else lab) for lab in labels]

    return _make


class TestSameAnswer:
    """Spark must agree with memory exactly, not approximately."""

    @pytest.mark.parametrize("algorithm_name", DISTRIBUTABLE)
    def test_matches_the_in_memory_result(
        self, frame: Any, noisy: DatasetFactory, algorithm_name: str
    ) -> None:
        from thresher.spark import SparkThresher

        scores, actual_classes = noisy(5000, seed=1)

        in_memory = thresher.Thresher(algorithm=algorithm_name).optimize_threshold(scores, actual_classes)
        distributed = SparkThresher(algorithm_name).optimize_threshold(
            frame(scores, actual_classes), "score", "label"
        )

        assert distributed == in_memory

    @pytest.mark.parametrize("algorithm_name", DISTRIBUTABLE)
    def test_agrees_on_data_full_of_ties(self, frame: Any, algorithm_name: str) -> None:
        """Duplicates land in the same bin or the same score group however partitioned."""
        from thresher.spark import SparkThresher

        rng = random.Random(2)
        scores = [round(rng.random(), 2) for _ in range(3000)]
        actual_classes = [rng.choice([-1, 1]) for _ in range(3000)]

        in_memory = thresher.Thresher(algorithm=algorithm_name).optimize_threshold(scores, actual_classes)
        distributed = SparkThresher(algorithm_name).optimize_threshold(
            frame(scores, actual_classes), "score", "label"
        )

        assert distributed == in_memory

    def test_partition_count_does_not_change_the_answer(self, frame: Any, noisy: DatasetFactory) -> None:
        """Where the shard boundaries fall is an execution detail, not an input."""
        from thresher.spark import SparkThresher

        scores, actual_classes = noisy(4000, seed=3)
        df = frame(scores, actual_classes)

        results = {
            SparkThresher("hist").optimize_threshold(df.repartition(partitions), "score", "label")
            for partitions in (1, 3, 8)
        }

        assert len(results) == 1, f"partitioning changed the result: {results}"

    def test_resolution_is_honoured(self, frame: Any, noisy: DatasetFactory) -> None:
        from thresher.spark import SparkThresher

        scores, actual_classes = noisy(4000, seed=4)
        df = frame(scores, actual_classes)

        for bins in (16, 256):
            in_memory = thresher.Thresher(
                algorithm="hist", algorithm_params={"no_of_bins": bins}
            ).optimize_threshold(scores, actual_classes)
            distributed = SparkThresher("hist", {"no_of_bins": bins}).optimize_threshold(df, "score", "label")
            assert distributed == in_memory, f"disagreed at {bins} bins"

    def test_custom_labels(self, frame: Any) -> None:
        from thresher.spark import SparkThresher

        scores = [0.1, 0.3, 0.4, 0.7]
        zero_one = [0, 0, 1, 1]

        result = SparkThresher("exact", labels=(0, 1)).optimize_threshold(
            frame(scores, zero_one), "score", "label"
        )

        assert 0.3 <= result < 0.4

    def test_column_names_are_not_assumed(self, spark: Any) -> None:
        from thresher.spark import SparkThresher

        df = spark.createDataFrame(
            [(1, 0.1, "a", -1), (2, 0.3, "b", -1), (3, 0.4, "c", 1), (4, 0.7, "d", 1)],
            ["id", "probability", "note", "outcome"],
        )

        result = SparkThresher("exact").optimize_threshold(df, "probability", "outcome")

        assert 0.3 <= result < 0.4


class TestRejections:
    """What it refuses, and whether it says why."""

    @pytest.mark.parametrize("algorithm_name", ["ls", "grid", "sgrid", "gen", "sgd"])
    def test_algorithms_that_cannot_distribute_are_refused(self, algorithm_name: str) -> None:
        """Refused rather than quietly run on a sample or on the driver."""
        from thresher.spark import SparkThresher

        with pytest.raises(ConfigurationError) as excinfo:
            SparkThresher(algorithm_name)

        message = str(excinfo.value)
        assert algorithm_name in message
        assert "hist" in message and "exact" in message, "should name what can be used"

    def test_an_unknown_name_is_still_an_unknown_algorithm(self) -> None:
        from thresher.spark import SparkThresher

        with pytest.raises(UnknownAlgorithmError):
            SparkThresher("does-not-exist")

    def test_refusals_are_thresher_errors(self) -> None:
        from thresher.spark import SparkThresher

        with pytest.raises(ThresherError):
            SparkThresher("sgd")

    def test_empty_frame(self, spark: Any) -> None:
        from thresher.spark import SparkThresher

        empty = spark.createDataFrame([], "score double, label int")

        with pytest.raises(EmptyInputError):
            SparkThresher("hist").optimize_threshold(empty, "score", "label")

    def test_single_class(self, frame: Any) -> None:
        from thresher.spark import SparkThresher

        with pytest.raises(SingleClassError):
            SparkThresher("hist").optimize_threshold(frame([0.1, 0.2, 0.3], [-1, -1, -1]), "score", "label")


class TestWhatItReports:
    """`verbose` and the warnings are user-facing, so they should actually say something."""

    def test_verbose_describes_the_aggregation(self, frame: Any, caplog: pytest.LogCaptureFixture) -> None:
        from thresher.spark import SparkThresher

        with caplog.at_level("INFO", logger="thresher.spark"):
            SparkThresher("hist", verbose=True).optimize_threshold(
                frame([0.1, 0.3, 0.4, 0.7], [-1, -1, 1, 1]), "score", "label"
            )

        assert "4 rows" in caplog.text, "should say how much it is aggregating"
        assert "Histogram" in caplog.text, "and which algorithm is deciding"

    def test_silent_unless_asked(self, frame: Any, caplog: pytest.LogCaptureFixture) -> None:
        from thresher.spark import SparkThresher

        with caplog.at_level("INFO", logger="thresher.spark"):
            SparkThresher("hist").optimize_threshold(
                frame([0.1, 0.3, 0.4, 0.7], [-1, -1, 1, 1]), "score", "label"
            )

        assert caplog.text == ""

    def test_a_wide_collect_suggests_the_bounded_algorithm(
        self, frame: Any, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`exact` ships one row per distinct score, which stops being cheap eventually.

        The real limit is a million distinct scores; lowering it here is what makes the
        path reachable without building a dataset that large.
        """
        from thresher.spark import SparkThresher

        monkeypatch.setattr("thresher.spark.DISTINCT_SCORE_WARNING", 2)

        with caplog.at_level("WARNING", logger="thresher.spark"):
            SparkThresher("exact").optimize_threshold(
                frame([0.1, 0.3, 0.4, 0.7], [-1, -1, 1, 1]), "score", "label"
            )

        assert "hist" in caplog.text, "the warning should name the alternative"

    @pytest.mark.parametrize("bins", [0, -1])
    def test_a_bin_count_below_one_is_rejected(self, frame: Any, bins: int) -> None:
        # Reaches the aggregation before it is caught, so it has to be a real error rather
        # than an empty list of bins that sweeps to nothing.
        from thresher.spark import SparkThresher

        with pytest.raises(InsufficientDataError, match="at least 1"):
            SparkThresher("hist", {"no_of_bins": bins}).optimize_threshold(
                frame([0.1, 0.3, 0.4, 0.7], [-1, -1, 1, 1]), "score", "label"
            )


class TestEdges:
    def test_all_scores_identical(self, frame: Any) -> None:
        from thresher.spark import SparkThresher

        result = SparkThresher("hist").optimize_threshold(
            frame([0.5] * 6, [-1, -1, -1, 1, 1, 1]), "score", "label"
        )

        assert isinstance(result, float)

    def test_reaches_the_classify_everything_positive_split(self, frame: Any) -> None:
        from thresher.spark import SparkThresher

        scores, actual_classes = [0.1, 0.2, 0.3], [1, 1, -1]

        result = SparkThresher("exact").optimize_threshold(frame(scores, actual_classes), "score", "label")

        assert result < min(scores)

    def test_scores_outside_the_unit_interval(self, frame: Any) -> None:
        from thresher.spark import SparkThresher

        scores, actual_classes = [-5.0, -4.0, 12.0, 13.0], [-1, -1, 1, 1]

        result = SparkThresher("hist").optimize_threshold(frame(scores, actual_classes), "score", "label")

        assert min(scores) <= result <= max(scores)

    @pytest.mark.parametrize("alias", ["hist", "histogram", "bins", "exact", "sweep"])
    def test_aliases_resolve(self, alias: str) -> None:
        from thresher.spark import SparkThresher

        assert SparkThresher(alias).algorithm.id in DISTRIBUTABLE


class TestParameterValidation:
    """The Spark interface takes `algorithm_params` too, and validates them the same way.

    A mistyped key here would leave the default in place across an entire cluster run
    (#34), which is a long way to travel to be told nothing.
    """

    def test_a_mistyped_param_is_rejected(self) -> None:
        from thresher.spark import SparkThresher

        with pytest.raises(ValueError, match="no_of_bin"):
            SparkThresher(algorithm_params={"no_of_bin": 4096})

    def test_exact_takes_no_parameters(self) -> None:
        from thresher.spark import SparkThresher

        with pytest.raises(ValueError, match="nothing to tune"):
            SparkThresher("exact", algorithm_params={"no_of_bins": 4096})

    def test_the_documented_parameter_is_accepted(
        self, frame: Callable[[list[float], list[int]], Any]
    ) -> None:
        from thresher.spark import SparkThresher

        data = frame([0.1, 0.2, 0.8, 0.9], [-1, -1, 1, 1])

        assert SparkThresher(algorithm_params={"no_of_bins": 64}).optimize_threshold(data)


class TestRefusesWhatTheMemoryPathRefuses:
    """Rows the in-memory path rejects, fixed in 0.7.1 (#21, #24).

    The class counts come from equality against the two declared labels, so anything
    matching neither - a null, a third value, a typo - was simply absent from both and
    landed in the negative count by omission. Nulls and NaNs in the score column went the
    same way: Spark's `least` skips nulls, so a row with no score at all was filed in the
    top bin, and NaN sorts above everything, so it became the maximum and collapsed the
    whole computation. Each returned a plausible threshold computed from data the
    in-memory path refuses outright.
    """

    def test_a_third_label_value_is_refused(self, spark: Any) -> None:
        frame = spark.createDataFrame(
            [(0.1, 0), (0.2, 0), (0.6, 1), (0.9, 1), (0.95, 2), (0.97, 2)],
            "score double, label int",
        )

        with pytest.raises(UnexpectedLabelsError) as excinfo:
            SparkThresher(labels=(0, 1)).optimize_threshold(frame, "score", "label")

        # Naming the offending value beats reporting a count.
        assert 2 in excinfo.value.unexpected

    def test_it_used_to_move_the_answer(self, spark: Any) -> None:
        """The clean rows alone give one answer; the unusable ones used to shift it."""
        clean = spark.createDataFrame([(0.1, 0), (0.2, 0), (0.6, 1), (0.9, 1)], "score double, label int")
        expected = SparkThresher(labels=(0, 1)).optimize_threshold(clean, "score", "label")

        polluted = spark.createDataFrame(
            [(0.1, 0), (0.2, 0), (0.6, 1), (0.9, 1), (0.95, 2), (0.97, 2), (0.99, 2)],
            "score double, label int",
        )

        with pytest.raises(ThresherError):
            SparkThresher(labels=(0, 1)).optimize_threshold(polluted, "score", "label")

        assert expected == pytest.approx(0.2, abs=0.01), "the clean answer, for contrast"

    def test_null_labels_are_refused(self, spark: Any) -> None:
        frame = spark.createDataFrame(
            [(0.1, 0), (0.2, 0), (0.6, 1), (0.9, 1), (0.95, None), (0.97, None)],
            "score double, label int",
        )

        with pytest.raises(MissingLabelsError) as excinfo:
            SparkThresher(labels=(0, 1)).optimize_threshold(frame, "score", "label")

        assert excinfo.value.count == 2

    def test_labels_matching_nothing_are_not_reported_as_a_single_class(self, spark: Any) -> None:
        """Both counts land at zero, which used to read as "only -1 present" - of nothing."""
        frame = spark.createDataFrame([(0.1, 7), (0.2, 7), (0.6, 9), (0.9, 9)], "score double, label int")

        with pytest.raises(UnexpectedLabelsError):
            SparkThresher().optimize_threshold(frame, "score", "label")

    def test_null_scores_are_refused(self, spark: Any) -> None:
        frame = spark.createDataFrame(
            [(0.1, -1), (0.2, -1), (0.6, 1), (None, 1), (0.9, 1)], "score double, label int"
        )

        with pytest.raises(UndefinedScoresError) as excinfo:
            SparkThresher().optimize_threshold(frame, "score", "label")

        assert excinfo.value.count == 1

    def test_nan_scores_are_refused(self, spark: Any) -> None:
        frame = spark.createDataFrame(
            [(0.1, -1), (0.2, -1), (0.6, 1), (float("nan"), 1), (0.9, 1)],
            "score double, label int",
        )

        with pytest.raises(UndefinedScoresError):
            SparkThresher().optimize_threshold(frame, "score", "label")

    def test_exact_refuses_them_too(self, spark: Any) -> None:
        """`exact` collects by distinct score, where a null died as a bare TypeError."""
        frame = spark.createDataFrame(
            [(0.1, -1), (0.2, -1), (0.6, 1), (None, 1), (0.9, 1)], "score double, label int"
        )

        with pytest.raises(UndefinedScoresError):
            SparkThresher("exact").optimize_threshold(frame, "score", "label")

    def test_an_integer_score_column_still_works(self, spark: Any) -> None:
        """`isnan` is undefined for integer types, so the check has to skip it there."""
        frame = spark.createDataFrame([(1, -1), (2, -1), (6, 1), (9, 1)], "score int, label int")

        assert SparkThresher().optimize_threshold(frame, "score", "label")

    @pytest.mark.parametrize("algorithm_name", DISTRIBUTABLE)
    def test_clean_data_is_unaffected(
        self, spark: Any, frame: Callable[[list[float], list[int]], Any], algorithm_name: str
    ) -> None:
        scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        actual_classes = [-1, -1, -1, 1, 1, 1]

        distributed = SparkThresher(algorithm_name).optimize_threshold(
            frame(scores, actual_classes), "score", "label"
        )
        in_memory = thresher.Thresher(algorithm=algorithm_name).optimize_threshold(scores, actual_classes)

        assert distributed == in_memory

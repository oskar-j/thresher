"""numpy and pandas input, which is what this library is actually given.

The README opens on a `predict_proba` array and the package depends on numpy and pandas
outright, yet until 0.7.2 no test anywhere passed either an `ndarray` or a `Series` to
`optimize_threshold`. Two defects lived in that gap: both types were copied to lists on
the way in - the `O(n)` allocation that makes `hist`'s bounded memory pointless, for
precisely the two types it matters most for - and the answer came back as an `np.float64`
from some algorithms and a `float` from others.

Neither could be caught where the existing tests look. `tests/test_histogram.py`
demonstrates the bounded memory by calling `histogram.run` directly with lists, which
never touches the interface where the copying happened; every other module passes lists
too. So the tests here go through the public entry point on purpose.
"""

import random
import tracemalloc
from collections.abc import Iterable, Iterator

import numpy as np
import pandas as pd
import pytest

import thresher
from thresher.utils import as_sequence

ALL_ALGORITHMS = ["exact", "hist", "ls", "sgd", "gen", "grid", "sgrid"]

Dataset = tuple[list[float], list[int]]
# `Iterable` rather than `Sequence`, and not out of generosity: mypy rejects an ndarray
# where a `Sequence` is asked for, which is the same fact `as_sequence` exists to work
# around. `optimize_threshold` accepts `Iterable`, so that is what these hold.
Containers = dict[str, tuple[Iterable[float], Iterable[int]]]


@pytest.fixture
def dataset() -> Dataset:
    """A few hundred rows with a boundary at 0.5, small enough for the O(n^2) solver."""
    rng = random.Random(11)
    scores = [round(rng.random(), 3) for _ in range(400)]
    return scores, [1 if score > 0.5 else -1 for score in scores]


@pytest.fixture
def containers(dataset: Dataset) -> Containers:
    """The same data in every shape a caller might hold it in.

    The last one is the case that makes pandas awkward rather than merely uncopied: a
    Series that has been filtered keeps the labels of the rows that survived, so its index
    no longer counts from zero. `series[0]` is a *label* lookup, and on this one it raises.
    """
    scores, actual_classes = dataset
    gaps = range(0, 2 * len(scores), 2)
    return {
        "list": (scores, actual_classes),
        "ndarray": (np.array(scores), np.array(actual_classes)),
        "series": (pd.Series(scores), pd.Series(actual_classes)),
        "gapped series": (pd.Series(scores, index=gaps), pd.Series(actual_classes, index=gaps)),
    }


def solve(algorithm_name: str, scores: Iterable[float], actual_classes: Iterable[int]) -> float:
    """Run one algorithm on one container, from a fixed random state.

    Four of the seven solvers sample, so without seeding they would differ between
    containers for reasons that have nothing to do with the container.
    """
    random.seed(3)
    return thresher.Thresher(algorithm=algorithm_name).optimize_threshold(scores, actual_classes)


class TestTheContainerDoesNotChangeTheAnswer:
    """A list, an array and a Series hold the same numbers, so they must agree."""

    @pytest.mark.parametrize("algorithm_name", ALL_ALGORITHMS)
    def test_every_algorithm_agrees_across_every_container(
        self, algorithm_name: str, containers: Containers
    ) -> None:
        answers = {name: solve(algorithm_name, *data) for name, data in containers.items()}

        assert len(set(answers.values())) == 1, f"{algorithm_name} disagreed with itself: {answers}"

    @pytest.mark.parametrize("algorithm_name", ["sgd", "gen", "sgrid"])
    def test_a_filtered_series_is_read_by_position(self, algorithm_name: str, containers: Containers) -> None:
        """The three solvers that sample index into the input, and must do so positionally.

        `stochastic_process` and `grid._get_random_projection` both draw indices from
        `range(len(scores))` and then subscript. Handing a Series over untouched would make
        those label lookups: a KeyError here, and - worse, where the labels happen to be
        present - the wrong row without any error at all. `as_sequence` takes the array
        underneath for that reason, which costs nothing since it is a view.
        """
        expected = solve(algorithm_name, *containers["list"])

        assert solve(algorithm_name, *containers["gapped series"]) == expected

    def test_a_label_mapping_applies_to_arrays_too(self, dataset: Dataset) -> None:
        scores, actual_classes = dataset
        zero_one = np.array([0 if actual == -1 else 1 for actual in actual_classes])

        mapped = thresher.Thresher(labels=(0, 1)).optimize_threshold(np.array(scores), zero_one)

        assert mapped == thresher.Thresher().optimize_threshold(scores, actual_classes)


class TestTheResultIsAPlainFloat:
    """`optimize_threshold` is annotated `-> float`, and now returns one."""

    @pytest.mark.parametrize("algorithm_name", ALL_ALGORITHMS)
    def test_numpy_input_does_not_leak_a_numpy_scalar(
        self, algorithm_name: str, containers: Containers
    ) -> None:
        # `np.float64` subclasses `float`, so `isinstance` cannot see this: it passed
        # throughout while `exact`, `hist` and `ls` were handing one back and `grid` was
        # not. Under numpy 2 the difference shows up wherever the result is printed, as
        # `np.float64(0.35)`.
        result = solve(algorithm_name, *containers["ndarray"])

        assert type(result) is float


class TestNothingIsCopiedOnTheWayIn:
    """`hist`'s reason to exist, measured where the copy actually was.

    The companion test in `tests/test_histogram.py` calls the solver directly, so it holds
    even with the interface copying both arguments in full - which is how this went
    unnoticed. These go through `Thresher`.
    """

    @staticmethod
    def _peak_bytes(scores: Iterable[float], actual_classes: Iterable[int]) -> int:
        tracemalloc.start()
        thresher.Thresher(algorithm="hist").optimize_threshold(scores, actual_classes)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    def test_arrays_cost_no_more_than_lists(self) -> None:
        rng = random.Random(4)
        scores = [rng.random() for _ in range(200_000)]
        actual_classes = [1 if score > 0.5 else -1 for score in scores]
        # Built before the measurement starts: the point is what optimizing allocates.
        arrays = (np.array(scores), np.array(actual_classes))
        series = (pd.Series(scores), pd.Series(actual_classes))

        peaks = {
            "list": self._peak_bytes(scores, actual_classes),
            "ndarray": self._peak_bytes(*arrays),
            "series": self._peak_bytes(*series),
        }

        # Copying 200,000 rows costs megabytes - it measured 12.2 MiB for the array and
        # 7.6 MiB for the Series against 18 KiB for the list. A small multiple of the list
        # figure is far below that and far above the noise.
        for name, peak in peaks.items():
            assert peak < 5 * peaks["list"], f"{name} allocated {peak:,} bytes: {peaks}"

    def test_array_allocation_stays_flat_as_the_data_grows(self) -> None:
        rng = random.Random(5)
        peaks = {}
        for size in (50_000, 500_000):
            scores = np.array([rng.random() for _ in range(size)])
            actual_classes = np.array([1 if score > 0.5 else -1 for score in scores])
            peaks[size] = self._peak_bytes(scores, actual_classes)

        growth = peaks[500_000] / peaks[50_000]
        assert growth < 3, f"memory grew {growth:.1f}x for 10x the data: {peaks}"


class TestWhatStillHasToBeCopied:
    """Not every input can be handed over as it stands, and those must keep working."""

    def test_a_generator_is_consumed_into_a_list(self, dataset: Dataset) -> None:
        """It can only be walked once, and every solver needs at least two passes."""
        scores, actual_classes = dataset

        def stream() -> Iterator[float]:
            yield from scores

        result = thresher.Thresher().optimize_threshold(stream(), iter(actual_classes))

        assert result == thresher.Thresher().optimize_threshold(scores, actual_classes)

    def test_a_list_is_handed_over_untouched(self, dataset: Dataset) -> None:
        """The property 0.5.3 added, asserted on the identity rather than on a byte count."""
        scores, _ = dataset

        assert as_sequence(scores) is scores

    def test_an_array_is_handed_over_untouched(self, dataset: Dataset) -> None:
        scores, _ = dataset
        array = np.array(scores)

        # Widened to `object` only so the identity check typechecks: `as_sequence` is
        # annotated `-> Sequence`, and an ndarray is not one as far as mypy is concerned,
        # which is the whole reason this function has to exist.
        handed_over: object = as_sequence(array)

        assert handed_over is array

    def test_a_series_hands_over_a_view_rather_than_a_copy(self, dataset: Dataset) -> None:
        scores, _ = dataset
        series = pd.Series(scores)

        assert np.shares_memory(as_sequence(series), series.to_numpy())

    def test_a_mapping_still_becomes_a_list_of_its_keys(self) -> None:
        """It has `__len__` and `__getitem__`, but iterating it does not yield what
        subscripting it does - so it cannot be handed over on the strength of the dunders.
        """
        mapping = {0.1: "a", 0.9: "b"}

        assert as_sequence(mapping) == [0.1, 0.9]


class TestEmptyArrays:
    """`if not scores` is ambiguous for an array, so the guards count instead."""

    @pytest.mark.parametrize("algorithm_name", ALL_ALGORITHMS)
    def test_an_empty_array_is_refused_the_same_way_an_empty_list_is(self, algorithm_name: str) -> None:
        empty = np.array([], dtype=float)

        with pytest.raises(thresher.exceptions.EmptyInputError):
            thresher.Thresher(algorithm=algorithm_name).optimize_threshold(empty, empty)

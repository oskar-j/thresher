"""The histogram sweep.

It is an approximation, so the tests pin down the shape of the approximation rather than an
exact answer: how close it gets, that closeness improves with resolution, that it does not
hold the data, and that it returns the same answer twice.
"""

import random
import tracemalloc
from collections.abc import Callable

import pytest

import thresher
from thresher.algs.histogram import compute as histogram
from thresher.exceptions import InsufficientDataError

Dataset = tuple[list[float], list[int]]
DatasetFactory = Callable[..., Dataset]


def accuracy(threshold: float, scores: list[float], actual_classes: list[int]) -> float:
    """Fraction of samples the threshold classifies correctly."""
    return sum(
        1
        for score, actual in zip(scores, actual_classes, strict=True)
        if (1 if score > threshold else -1) == actual
    ) / len(scores)


@pytest.fixture
def overlapping() -> DatasetFactory:
    """Classes that overlap, so the best threshold still gets some wrong."""

    def _make(n: int, seed: int = 0, flip: float = 0.15) -> Dataset:
        rng = random.Random(seed)
        scores = [rng.random() for _ in range(n)]
        labels = [1 if score > 0.5 else -1 for score in scores]
        return scores, [(-lab if rng.random() < flip else lab) for lab in labels]

    return _make


class TestAccuracy:
    """How much is given up against the exact answer, and does resolution buy it back."""

    @pytest.mark.parametrize("seed", range(8))
    def test_close_to_the_exact_optimum(self, overlapping: DatasetFactory, seed: int) -> None:
        scores, actual_classes = overlapping(4000, seed)

        exact = thresher.Thresher(algorithm="exact").optimize_threshold(scores, actual_classes)
        binned = thresher.Thresher(algorithm="hist").optimize_threshold(scores, actual_classes)

        lost = accuracy(exact, scores, actual_classes) - accuracy(binned, scores, actual_classes)
        assert 0 <= lost < 0.01, f"gave up {lost:.4f} accuracy at the default resolution"

    def test_more_bins_never_do_worse(self, overlapping: DatasetFactory) -> None:
        """Resolution is the dial, so turning it up should not cost accuracy."""
        scores, actual_classes = overlapping(4000, seed=1)
        exact = accuracy(
            thresher.Thresher(algorithm="exact").optimize_threshold(scores, actual_classes),
            scores,
            actual_classes,
        )

        losses = []
        for bins in (16, 128, 1024, 8192):
            result = thresher.Thresher(
                algorithm="hist", algorithm_params={"no_of_bins": bins}
            ).optimize_threshold(scores, actual_classes)
            losses.append(exact - accuracy(result, scores, actual_classes))

        assert losses[-1] <= losses[0], f"more bins did worse: {losses}"
        assert all(loss >= 0 for loss in losses), "cannot beat the exact optimum"

    def test_resolution_bounds_the_error_on_separable_data(self, separable: DatasetFactory) -> None:
        """Even a perfect boundary cannot be split finer than a bin.

        A threshold can only be placed on a bin edge, so when the true boundary falls
        *inside* a bin the samples on the wrong side of it are unavoidable. That puts a
        number on the approximation: the samples lost track the samples per bin, and
        enough bins removes them entirely.
        """
        scores, actual_classes = separable(2000, seed=3)

        def wrong(bins: int) -> int:
            result = thresher.Thresher(
                algorithm="hist", algorithm_params={"no_of_bins": bins}
            ).optimize_threshold(scores, actual_classes)
            return round((1 - accuracy(result, scores, actual_classes)) * len(scores))

        # 2,000 samples: ~31 per bin at 64 bins, ~2 at 1,024, well under one at 32,768.
        assert wrong(64) <= 2000 / 64 + 1
        assert wrong(1024) <= 2000 / 1024 + 1
        assert wrong(32_768) == 0, "enough bins should separate perfectly separable data"

    def test_imbalanced_data(self, skewed: DatasetFactory) -> None:
        """The case that defeats the sampling-based approximations."""
        scores, actual_classes = skewed(5000, 0.95, seed=1)

        result = thresher.Thresher(algorithm="hist").optimize_threshold(scores, actual_classes)

        assert accuracy(result, scores, actual_classes) > 0.99


class TestBoundedMemory:
    """The reason this algorithm exists: its allocation does not follow the input."""

    def test_allocation_barely_grows_with_the_data(self, overlapping: DatasetFactory) -> None:
        peaks = {}
        for size in (50_000, 500_000):
            scores, actual_classes = overlapping(size)
            tracemalloc.start()
            histogram.run(scores, actual_classes, verbose=False, progress_bar=False, alg_options={})
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peaks[size] = peak

        # Ten times the data must not cost anything like ten times the memory. The counters
        # are fixed; only their integer values grow.
        growth = peaks[500_000] / peaks[50_000]
        assert growth < 3, f"memory grew {growth:.1f}x for 10x the data: {peaks}"

    def test_resolution_sets_the_memory_not_the_input(self, overlapping: DatasetFactory) -> None:
        scores, actual_classes = overlapping(50_000)

        peaks = {}
        for bins in (64, 65_536):
            tracemalloc.start()
            histogram.run(scores, actual_classes, False, False, {"no_of_bins": bins})
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peaks[bins] = peak

        assert peaks[65_536] > peaks[64], "more bins should cost more memory, not less"


class TestBehaviour:
    def test_is_deterministic(self, overlapping: DatasetFactory) -> None:
        """Unlike the sampling approximations, the same data gives the same answer."""
        scores, actual_classes = overlapping(2000, seed=5)

        results = {
            thresher.Thresher(algorithm="hist").optimize_threshold(scores, actual_classes) for _ in range(5)
        }

        assert len(results) == 1

    def test_reaches_the_classify_everything_positive_split(self) -> None:
        # Score and class run contrary to each other, so no threshold inside the data wins.
        scores, actual_classes = [0.1, 0.2, 0.3], [1, 1, -1]

        result = thresher.Thresher(algorithm="hist").optimize_threshold(scores, actual_classes)

        assert result < min(scores)
        assert accuracy(result, scores, actual_classes) == pytest.approx(2 / 3)

    def test_all_scores_identical(self) -> None:
        """Everything lands in one bin, and an answer still has to come back."""
        scores = [0.5] * 6
        actual_classes = [-1, -1, -1, 1, 1, 1]

        result = thresher.Thresher(algorithm="hist").optimize_threshold(scores, actual_classes)

        assert isinstance(result, float)

    def test_scores_outside_the_unit_interval(self) -> None:
        """Bins are laid over the observed range, not assumed to be [0, 1] like grid."""
        scores = [-5.0, -4.0, 12.0, 13.0]
        actual_classes = [-1, -1, 1, 1]

        result = thresher.Thresher(algorithm="hist").optimize_threshold(scores, actual_classes)

        assert accuracy(result, scores, actual_classes) == 1.0

    @pytest.mark.parametrize("alias", ["hist", "histogram", "binned", "bins"])
    def test_aliases(self, alias: str) -> None:
        assert thresher.Thresher(algorithm=alias).get_current_algorithm()["name"] == "hist"

    @pytest.mark.parametrize("bins", [0, -1])
    def test_a_useless_resolution_is_rejected(self, bins: int) -> None:
        with pytest.raises(InsufficientDataError, match="at least 1"):
            thresher.Thresher(algorithm="hist", algorithm_params={"no_of_bins": bins}).optimize_threshold(
                [0.1, 0.9], [-1, 1]
            )

    def test_no_scores(self) -> None:
        with pytest.raises(InsufficientDataError):
            histogram.run([], [], verbose=False, progress_bar=False, alg_options={})

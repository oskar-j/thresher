"""The histogram sweep.

It is an approximation, so the tests pin down the shape of the approximation rather than an
exact answer: how close it gets, that closeness improves with resolution, that it does not
hold the data, and that it returns the same answer twice.
"""

import math
import random
import tracemalloc
from collections.abc import Callable

import pytest

import thresher
from thresher.algs.histogram import compute as histogram
from thresher.algs.histogram.compute import _boundary_threshold, bin_index, sweep_bins
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
            histogram.run(scores, actual_classes, progress_bar=False, alg_options={})
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
            histogram.run(scores, actual_classes, False, {"no_of_bins": bins})
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
            histogram.run([], [], progress_bar=False, alg_options={})


class TestBinEdgeAgreesWithPrediction:
    """The returned threshold must achieve the count the sweep reported, fixed in 0.7.1.

    Binning floors, so a score sitting exactly on a bin edge belongs to the bin *above*
    it. Prediction is `score > threshold`, which sends a score sitting exactly on the
    threshold to the class *below* it. Returning the edge itself made the two disagree for
    precisely the samples on that edge: the sweep reported a number of correct predictions
    the threshold does not achieve, and would prefer a worse edge to a better one (#20).
    """

    def test_the_reported_count_is_the_count_achieved(self) -> None:
        """The case from the issue: 8/10 claimed, 6/10 delivered, 8/10 available."""
        scores = [0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0]
        actual_classes = [-1, -1, 1, 1, -1, -1, 1, 1, 1, 1]

        result = thresher.Thresher(algorithm="hist", algorithm_params={"no_of_bins": 4}).optimize_threshold(
            scores, actual_classes
        )

        assert accuracy(result, scores, actual_classes) == pytest.approx(0.8)

    @pytest.mark.parametrize("seed", range(40))
    def test_it_never_claims_more_than_it_delivers(self, seed: int) -> None:
        """Across shapes where score values land on bin edges, which is the trigger."""
        rng = random.Random(seed)
        places = rng.choice([1, 2])
        bins = rng.choice([4, 10, 100])
        size = rng.choice([50, 500])
        scores = [round(rng.random(), places) for _ in range(size)]
        actual_classes = [rng.choice([-1, 1]) for _ in range(size)]
        if len(set(actual_classes)) < 2:
            pytest.skip("needs both classes present")

        negatives, positives = [0] * bins, [0] * bins
        lowest, highest = min(scores), max(scores)
        span = highest - lowest
        for score, actual in zip(scores, actual_classes, strict=True):
            index = bin_index(score, lowest, span, bins)
            if actual == 1:
                positives[index] += 1
            else:
                negatives[index] += 1

        threshold, reported = sweep_bins(negatives, positives, lowest=lowest, highest=highest)
        achieved = sum(
            1
            for score, actual in zip(scores, actual_classes, strict=True)
            if (1 if score > threshold else -1) == actual
        )

        # Never fewer than claimed: that shortfall is the defect. Occasionally more, when
        # a duplicate at the boundary falls the favourable side of it, which harms nobody.
        assert achieved >= reported

    @pytest.mark.parametrize("seed", range(20))
    def test_it_returns_the_best_edge_available_to_it(self, seed: int) -> None:
        """Resolution is the only error it is allowed to have.

        Losing accuracy to a bin width is the documented trade. Losing it to an edge the
        binning could have expressed is the defect.
        """
        rng = random.Random(100 + seed)
        scores = [round(rng.random(), 2) for _ in range(400)]
        actual_classes = [1 if score > 0.5 else -1 for score in scores]
        actual_classes = [-a if rng.random() < 0.1 else a for a in actual_classes]
        if len(set(actual_classes)) < 2:
            pytest.skip("needs both classes present")

        bins = 20
        result = thresher.Thresher(
            algorithm="hist", algorithm_params={"no_of_bins": bins}
        ).optimize_threshold(scores, actual_classes)

        lowest, span = min(scores), max(scores) - min(scores)
        # Every split the binning can express: below everything, each interior boundary,
        # and the maximum itself.
        reachable = [math.nextafter(lowest, -math.inf)]
        reachable += [_boundary_threshold(lowest, span, index + 1, bins) for index in range(bins - 1)]
        reachable.append(lowest + span)
        best_available = max(accuracy(edge, scores, actual_classes) for edge in reachable)

        assert accuracy(result, scores, actual_classes) == pytest.approx(best_available)

    def test_the_maximum_is_taken_as_given_not_rebuilt_from_the_span(self) -> None:
        """`lowest + span` does not always reconstruct the largest score.

        With scores rounded to a few decimal places - the ordinary case, and the one that
        triggered this defect in the first place - `0.065 + (0.997 - 0.065)` is
        `0.9969999999999999`. A threshold there classifies the largest samples positive
        while the counting that chose the split had them negative, so the sweep reported
        4 correct where the threshold delivered 2.
        """
        lowest, highest = 0.065, 0.997
        assert lowest + (highest - lowest) != highest, "the premise: the rebuild is lossy"

        scores = [lowest, 0.5, 0.8, highest, highest]
        actual_classes = [1, -1, -1, -1, -1]
        bins = 4
        span = highest - lowest
        negatives, positives = [0] * bins, [0] * bins
        for score, actual in zip(scores, actual_classes, strict=True):
            index = bin_index(score, lowest, span, bins)
            if actual == 1:
                positives[index] += 1
            else:
                negatives[index] += 1

        threshold, reported = sweep_bins(negatives, positives, lowest=lowest, highest=highest)
        achieved = sum(
            1
            for score, actual in zip(scores, actual_classes, strict=True)
            if (1 if score > threshold else -1) == actual
        )

        assert achieved == reported == 4

    def test_the_topmost_edge_still_means_classify_everything_negative(self) -> None:
        """That split needs the maximum itself, so it is the one edge left alone."""
        scores = [0.1, 0.2, 0.3]
        actual_classes = [-1, -1, 1]

        # Contrary data: every interior split does worse than calling everything negative.
        result = thresher.Thresher(algorithm="hist", algorithm_params={"no_of_bins": 3}).optimize_threshold(
            [0.1, 0.2, 0.3], [1, -1, -1]
        )

        assert accuracy(result, [0.1, 0.2, 0.3], [1, -1, -1]) == pytest.approx(2 / 3)
        assert thresher.Thresher(algorithm="hist").optimize_threshold(scores, actual_classes)

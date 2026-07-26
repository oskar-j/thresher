"""The exact sweep.

The point of this algorithm is that it is *exact*, so the tests compare it against a brute
force search over every distinguishable threshold rather than against a tolerance.
"""

import math
import random
from collections.abc import Callable
from itertools import pairwise

import pytest

import thresher

Dataset = tuple[list[float], list[int]]
DatasetFactory = Callable[..., Dataset]


def accuracy(threshold: float, scores: list[float], actual_classes: list[int]) -> float:
    """Fraction of samples the threshold classifies correctly."""
    return sum(
        1
        for score, actual in zip(scores, actual_classes, strict=True)
        if (1 if score > threshold else -1) == actual
    ) / len(scores)


def best_possible_accuracy(scores: list[float], actual_classes: list[int]) -> float:
    """Brute force over every split a threshold can induce.

    Deliberately naive - it enumerates the midpoint between each pair of adjacent distinct
    values and scores each from scratch, plus both edge splits: the maximum itself
    (everything negative) and just below the minimum (everything positive). That is the
    definition the sweep has to meet.
    """
    unique = sorted(set(scores))
    candidates = [(low + high) / 2 for low, high in pairwise(unique)]
    candidates.append(unique[-1])
    candidates.append(math.nextafter(unique[0], -math.inf))
    return max(accuracy(t, scores, actual_classes) for t in candidates)


@pytest.fixture
def awkward_dataset() -> Callable[[int], Dataset]:
    """Random data built to be inconvenient: duplicates, ties and arbitrary labels."""

    def _make(seed: int) -> Dataset:
        rng = random.Random(seed)
        size = rng.randint(2, 60)
        # Low precision on purpose, so scores repeat and ties are common.
        scores = [round(rng.random(), rng.choice([1, 2, 3])) for _ in range(size)]
        actual_classes = [rng.choice([-1, 1]) for _ in range(size)]
        return scores, actual_classes

    return _make


@pytest.mark.parametrize("seed", range(60))
def test_matches_brute_force(awkward_dataset: Callable[[int], Dataset], seed: int) -> None:
    scores, actual_classes = awkward_dataset(seed)
    if len(set(actual_classes)) < 2:
        pytest.skip("needs both classes present")

    result = thresher.Thresher(algorithm="exact").optimize_threshold(scores, actual_classes)

    assert accuracy(result, scores, actual_classes) == pytest.approx(
        best_possible_accuracy(scores, actual_classes)
    )


@pytest.mark.parametrize("seed", range(40))
def test_never_worse_than_linear_search(awkward_dataset: Callable[[int], Dataset], seed: int) -> None:
    """It replaces linear search, so it must never do worse than it.

    It is sometimes strictly better: linear search only ever considers midpoints between
    adjacent scores, and so cannot express the "everything is negative" split that the
    sweep reaches at max(scores).
    """
    scores, actual_classes = awkward_dataset(seed)
    if len(set(actual_classes)) < 2:
        pytest.skip("needs both classes present")

    exact = thresher.Thresher(algorithm="exact").optimize_threshold(scores, actual_classes)
    linear = thresher.Thresher(algorithm="ls").optimize_threshold(scores, actual_classes)

    assert accuracy(exact, scores, actual_classes) >= accuracy(linear, scores, actual_classes)


def test_handles_all_scores_identical() -> None:
    """No threshold can separate equal scores, but one still has to come back."""
    scores = [0.5] * 6
    actual_classes = [-1, -1, -1, 1, 1, 1]

    result = thresher.Thresher(algorithm="exact").optimize_threshold(scores, actual_classes)

    assert result == 0.5


def test_handles_heavy_ties() -> None:
    """Runs of equal scores are indivisible and must not be split part-way through."""
    scores = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9]
    actual_classes = [-1, -1, -1, 1, 1, 1]

    result = thresher.Thresher(algorithm="exact").optimize_threshold(scores, actual_classes)

    assert accuracy(result, scores, actual_classes) == 1.0
    assert 0.1 <= result < 0.9


def test_reaches_the_classify_everything_positive_split() -> None:
    """The one split that needs a threshold below every score.

    Score and class run contrary to each other here, so every threshold *inside* the data
    does worse than simply calling everything positive. Before 0.4.1 that split was
    unreachable and the sweep returned 0.15, getting 1 of 3 right where 2 of 3 was
    available.
    """
    scores = [0.1, 0.2, 0.3]
    actual_classes = [1, 1, -1]

    result = thresher.Thresher(algorithm="exact").optimize_threshold(scores, actual_classes)

    assert result < min(scores), "a below-minimum threshold is the only way to express this split"
    assert result == math.nextafter(min(scores), -math.inf), "and it should be the closest one"
    assert accuracy(result, scores, actual_classes) == pytest.approx(2 / 3)


def test_prefers_a_threshold_inside_the_data_on_a_tie(separable: DatasetFactory) -> None:
    """The edge split is taken only on a strict improvement, never to break a tie."""
    scores, actual_classes = separable(400, seed=5)

    result = thresher.Thresher(algorithm="exact").optimize_threshold(scores, actual_classes)

    assert min(scores) <= result <= max(scores)


def test_perfectly_separable_data(separable: DatasetFactory) -> None:
    scores, actual_classes = separable(2000, seed=3)

    result = thresher.Thresher(algorithm="exact").optimize_threshold(scores, actual_classes)

    assert accuracy(result, scores, actual_classes) == 1.0


def test_imbalanced_data(skewed: DatasetFactory) -> None:
    """The case the approximate solvers struggle with; exactness makes it unremarkable."""
    scores, actual_classes = skewed(5000, 0.95, seed=1)

    result = thresher.Thresher(algorithm="exact").optimize_threshold(scores, actual_classes)

    assert accuracy(result, scores, actual_classes) == 1.0


def test_is_deterministic(separable: DatasetFactory) -> None:
    """Unlike the stochastic solvers, the same input must always give the same answer."""
    scores, actual_classes = separable(500, seed=11)

    results = {
        thresher.Thresher(algorithm="exact").optimize_threshold(scores, actual_classes) for _ in range(5)
    }

    assert len(results) == 1


@pytest.mark.parametrize("alias", ["exact", "sweep", "exact_sweep", "sorted_sweep"])
def test_aliases(alias: str) -> None:
    assert thresher.Thresher(algorithm=alias).get_current_algorithm()["name"] == "exact"

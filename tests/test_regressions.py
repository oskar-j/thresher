"""Regression tests for defects fixed in 0.2.1 through 0.2.2.

Every case here raised an exception or returned an invalid result before those releases.
They exercise paths the rest of the suite does not reach: algorithms selected explicitly
rather than by the oracle, small inputs, and cleanly separable data.
"""

import math
from collections.abc import Callable

import pytest

import thresher
from thresher import algorithm

Dataset = tuple[list[float], list[int]]
DatasetFactory = Callable[..., Dataset]

ALL_ALGORITHMS = ["exact", "hist", "ls", "sgd", "gen", "grid", "sgrid"]


@pytest.mark.parametrize("algorithm_name", ["gen", "sgrid", "sgd"])
@pytest.mark.parametrize("size", [4, 9, 19, 21, 45])
def test_small_inputs_do_not_divide_by_zero(
    separable: DatasetFactory, algorithm_name: str, size: int
) -> None:
    # int(stoch_ratio * N) floored to 0 below N=50 for 'gen' and N=20 for 'sgrid',
    # producing an empty sample and then a division by zero.
    scores, actual_classes = separable(size)
    result = thresher.Thresher(algorithm=algorithm_name).optimize_threshold(scores, actual_classes)
    assert isinstance(result, float)


@pytest.mark.parametrize("size", [200, 500, 1000])
def test_sgd_on_separable_data(separable: DatasetFactory, size: int) -> None:
    # A perfect stochastic evaluation made 'previous_eval' 0.0, which the gradient update
    # then divided by.
    scores, actual_classes = separable(size)
    result = thresher.Thresher(algorithm="sgd").optimize_threshold(scores, actual_classes)
    assert isinstance(result, float)


@pytest.mark.parametrize("algorithm_name", ALL_ALGORITHMS)
@pytest.mark.parametrize("size", [200, 2000, 5000])
def test_result_stays_within_the_score_range(
    separable: DatasetFactory, algorithm_name: str, size: int
) -> None:
    """A returned threshold must correspond to a split of the data it was given.

    'sgd' used to walk clean out of the input range on separable data and return e.g.
    1.8972 for a predict_proba cut-off - plausible enough to go unnoticed.

    The lower bound is `nextafter(min, -inf)` rather than `min` because since 0.4.1 the
    exact sweep can return exactly that value, which is the only way to express
    "classify everything as positive". Nothing may sit any lower, and nothing at all may
    exceed `max(scores)`.
    """
    scores, actual_classes = separable(size, seed=size)
    result = thresher.Thresher(algorithm=algorithm_name).optimize_threshold(scores, actual_classes)

    assert math.nextafter(min(scores), -math.inf) <= result <= max(scores)


@pytest.mark.parametrize("algorithm_name", [a for a in ALL_ALGORITHMS if a != "exact"])
@pytest.mark.parametrize("size", [200, 2000])
def test_approximate_algorithms_stay_strictly_inside_the_range(
    separable: DatasetFactory, algorithm_name: str, size: int
) -> None:
    """Only the exact sweep has a reason to leave the span of the scores.

    The others have no way to represent an edge split, so a result outside `[min, max]`
    from any of them means the search has wandered, which is the 0.2.2 sgd bug.
    """
    scores, actual_classes = separable(size, seed=size)
    result = thresher.Thresher(algorithm=algorithm_name).optimize_threshold(scores, actual_classes)

    assert min(scores) <= result <= max(scores)


@pytest.mark.parametrize("size", [2000, 5000])
def test_sgd_converges_near_the_optimum(separable: DatasetFactory, size: int) -> None:
    # Guards the step-size cap: without it the walk overshoots, pins against a bound and
    # reports convergence there, landing far from the true threshold.
    scores, actual_classes = separable(size, seed=size)
    reference = thresher.Thresher(algorithm="ls").optimize_threshold(scores, actual_classes)
    result = thresher.Thresher(algorithm="sgd").optimize_threshold(scores, actual_classes)
    assert abs(result - reference) < 0.15


def _error_rate(threshold: float, scores: list[float], actual_classes: list[int]) -> float:
    """Fraction of samples the threshold gets wrong, measured on the full dataset."""
    wrong = sum(
        1
        for score, actual in zip(scores, actual_classes, strict=True)
        if (1 if score > threshold else -1) != actual
    )
    return wrong / len(scores)


@pytest.mark.parametrize("boundary", [0.7, 0.85])
def test_sgd_reaches_an_optimum_far_from_the_mean(skewed: DatasetFactory, boundary: float) -> None:
    """The walk has to travel from the mean of the scores to the real boundary.

    Before 0.3.1 it could not: the step size was scaled by the relative gain, so it
    collapsed as soon as progress slowed and the walk froze part-way. With the boundary
    at 0.85 it returned around 0.56, mis-classifying roughly 29% of samples while
    reporting convergence. These datasets are perfectly separable, so a correct answer
    mis-classifies nothing at all.
    """
    scores, actual_classes = skewed(8000, boundary, seed=int(boundary * 100))

    result = thresher.Thresher(algorithm="sgd").optimize_threshold(scores, actual_classes)

    assert _error_rate(result, scores, actual_classes) < 0.10


def test_sgd_returns_the_best_point_it_visited(separable: DatasetFactory) -> None:
    """Not merely the point it happened to stop on.

    The solver keeps walking through unproductive steps, so its final position is often
    worse than one it already passed through.
    """
    scores, actual_classes = separable(5000, seed=7)
    reference = thresher.Thresher(algorithm="ls").optimize_threshold(scores, actual_classes)

    result = thresher.Thresher(algorithm="sgd").optimize_threshold(scores, actual_classes)

    assert abs(result - reference) < 0.10


def test_get_current_algorithm() -> None:
    # Used 'with' on an Algorithm namedtuple, so it raised TypeError unconditionally.
    current = thresher.Thresher(algorithm="grid").get_current_algorithm()
    assert current["name"] == "grid"
    assert current["object"] == algorithm.available_algorithms["grid"]


@pytest.mark.parametrize("stoch_ratio", [0.05, 0.5])
def test_sgd_sample_ratio_is_configurable(skewed: DatasetFactory, stoch_ratio: float) -> None:
    """`stoch_ratio` was the one knob sgd did not expose, added in 0.4.4.

    It is the documented lever against sgd's weak spot - when one class is rare, a small
    subsample says little about where the boundary lies - so it has to actually reach the
    sampling and not be silently ignored like an unknown key would be.
    """
    scores, actual_classes = skewed(2000, 0.95, seed=1)

    result = thresher.Thresher(
        algorithm="sgd", algorithm_params={"stoch_ratio": stoch_ratio}
    ).optimize_threshold(scores, actual_classes)

    assert min(scores) <= result <= max(scores)


def test_a_larger_sgd_sample_reads_more_of_the_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Proves the option is wired through, rather than merely accepted and dropped."""
    # Imported from where it is defined; sgd.compute merely imports it, and strict mypy
    # will not treat an import as an attribute of the importing module.
    from thresher.algs.common.stochastic import stochastic_process as original

    seen: list[float] = []

    def recording(
        evaluated: float, scores: list[float], classes: list[int], factor: float, miss_class: bool = True
    ) -> float:
        seen.append(factor)
        return original(evaluated, scores, classes, factor, miss_class)

    # Patched by dotted path: `stochastic_process` is imported into that module rather
    # than defined there, and strict mypy will not treat an import as a module attribute.
    monkeypatch.setattr("thresher.algs.sgd.compute.stochastic_process", recording)

    thresher.Thresher(algorithm="sgd", algorithm_params={"stoch_ratio": 0.42}).optimize_threshold(
        [0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 0.95], [-1, -1, -1, -1, 1, 1, 1, 1]
    )

    assert seen, "sgd never sampled at all"
    assert set(seen) == {0.42}, f"expected every sample to use 0.42, saw {sorted(set(seen))}"

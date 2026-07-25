"""Regression tests for defects fixed in 0.2.1 through 0.2.2.

Every case here raised an exception or returned an invalid result before those releases.
They exercise paths the rest of the suite does not reach: algorithms selected explicitly
rather than by the oracle, small inputs, and cleanly separable data.
"""

from collections.abc import Callable

import pytest

import thresher
from thresher import algorithm

Dataset = tuple[list[float], list[int]]
DatasetFactory = Callable[..., Dataset]

ALL_ALGORITHMS = ["ls", "sgd", "gen", "grid", "sgrid"]


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
    """A threshold outside the input range puts every sample in one class.

    'sgd' used to walk out of that range on separable data and return e.g. 1.8972 for a
    predict_proba cut-off - plausible enough to go unnoticed.
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


def test_get_current_algorithm() -> None:
    # Used 'with' on an Algorithm namedtuple, so it raised TypeError unconditionally.
    current = thresher.Thresher(algorithm="grid").get_current_algorithm()
    assert current["name"] == "grid"
    assert current["object"] == algorithm.available_algorithms["grid"]

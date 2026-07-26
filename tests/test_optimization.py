"""The core behaviour: does optimize_threshold find the right cut-off?"""

from collections.abc import Callable

import pytest

import thresher
from thresher import algorithm
from thresher.oracle import run_oracle

Dataset = tuple[list[float], list[int]]
DatasetFactory = Callable[..., Dataset]

# The medium fixture's optimum sits inside this band. The solvers are stochastic, so the
# assertion is a range rather than an exact value.
MEDIUM_LOWER, MEDIUM_UPPER = 0.40, 0.65


@pytest.mark.parametrize(
    ("algorithm_name", "algorithm_params"),
    [
        pytest.param(None, {}, id="oracle-default"),
        pytest.param("exact", {}, id="exact"),
        pytest.param("linear", {}, id="linear"),
        pytest.param("sim", {}, id="genetic"),
        pytest.param("grid", {}, id="grid"),
        pytest.param("sgrid", {"no_of_decimal_places": 2, "stoch_ratio": 0.10}, id="sgrid"),
        pytest.param(
            "sgrid",
            {"no_of_decimal_places": 3, "stoch_ratio": 0.06, "reshuffle": True},
            id="sgrid-reshuffle",
        ),
    ],
)
def test_medium_dataset(
    medium_dataset: Dataset, algorithm_name: str | None, algorithm_params: dict[str, object]
) -> None:
    scores, actual_classes = medium_dataset
    kwargs: dict[str, object] = {"algorithm_params": algorithm_params}
    if algorithm_name is not None:
        kwargs["algorithm"] = algorithm_name

    result = thresher.Thresher(**kwargs).optimize_threshold(scores, actual_classes)

    assert MEDIUM_LOWER <= result < MEDIUM_UPPER


@pytest.mark.parametrize("data_length", [2, 500, 1_000, 50_000, 10_000_000])
def test_oracle_always_picks_the_exact_sweep(data_length: int) -> None:
    """Since 0.4.0 there is no size at which an approximation is preferable.

    The oracle used to route on input volume because the only exact algorithm was O(n²).
    'exact' is exact at every size and cheaper than what it replaced, so the trade-off the
    ladder encoded no longer exists.
    """
    assert run_oracle({"data_length": data_length}) == algorithm.available_algorithms["exact"]


def test_tiny_dataset(tiny_dataset: Dataset) -> None:
    scores, actual_classes = tiny_dataset
    assert 0.3 <= thresher.Thresher().optimize_threshold(scores, actual_classes) < 0.4


def test_custom_labels_are_normalized(tiny_dataset: Dataset) -> None:
    scores, _ = tiny_dataset
    t = thresher.Thresher(labels=(0, 1))
    assert 0.3 <= t.optimize_threshold(scores, [0, 0, 1, 1]) < 0.4


@pytest.mark.parametrize("n_jobs", [2, 3, -1])
def test_linear_search_in_parallel(separable: DatasetFactory, n_jobs: int) -> None:
    # n_jobs=-1 is documented in the README but used to make chunksize negative.
    scores, actual_classes = separable(200)
    t = thresher.Thresher(algorithm="linear", algorithm_params={"n_jobs": n_jobs})
    result = t.optimize_threshold(scores, actual_classes)
    assert min(scores) <= result <= max(scores)


def test_supported_algorithms() -> None:
    supported = thresher.Thresher.get_supported_algorithms()
    assert set(supported) == set(algorithm.available_algorithms)

    as_dict = thresher.Thresher.get_supported_algorithms(as_dict=True)
    assert isinstance(as_dict, dict)
    assert as_dict["grid"] == "Grid search"

"""The core behaviour: does optimize_threshold find the right cut-off?"""

from collections.abc import Callable

import pytest

import thresher
from thresher import algorithm

Dataset = tuple[list[float], list[int]]
DatasetFactory = Callable[..., Dataset]

# The medium fixture's optimum sits inside this band. The solvers are stochastic, so the
# assertion is a range rather than an exact value.
MEDIUM_LOWER, MEDIUM_UPPER = 0.40, 0.65


@pytest.mark.parametrize(
    ("algorithm_name", "algorithm_params"),
    [
        pytest.param(None, {}, id="default"),
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


@pytest.mark.parametrize("alias", ["auto", "default", "default_heuristics"])
def test_the_old_oracle_names_still_select_the_default(alias: str) -> None:
    """0.5.0 removed the oracle, having announced it in 0.4.0.

    The names that used to mean "let the oracle decide" are kept as synonyms of the
    default, so calls written against earlier versions keep working. They resolve to
    `exact` because that is what the oracle had been returning unconditionally since
    0.4.0 anyway.
    """
    assert thresher.Thresher(algorithm=alias).get_current_algorithm()["name"] == "exact"


def test_the_default_is_the_exact_sweep() -> None:
    assert thresher.Thresher().get_current_algorithm()["name"] == "exact"
    assert algorithm.DEFAULT is algorithm.available_algorithms["exact"]


def test_auto_is_no_longer_an_algorithm_in_its_own_right() -> None:
    """It named the oracle, and there is no oracle to name."""
    assert "auto" not in algorithm.available_algorithms
    assert "auto" not in thresher.Thresher.get_supported_algorithms()


def test_every_supported_name_is_a_real_algorithm() -> None:
    """Each one has to reach a dispatch branch, not just resolve."""
    scores, actual_classes = [0.1, 0.3, 0.4, 0.7], [-1, -1, 1, 1]

    for name in thresher.Thresher.get_supported_algorithms():
        result = thresher.Thresher(algorithm=name).optimize_threshold(scores, actual_classes)
        assert isinstance(result, float), f"{name} did not produce a threshold"


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


class TestSlowInputWarning:
    """Crossing an algorithm's `data_vol_thresh` warns rather than refusing.

    The thresholds are order-of-magnitude guidance measured on one machine, so a caller
    may well have good reason to wait. Warning through `logging` rather than `warnings`
    means it is silenced with ordinary logging configuration.
    """

    @staticmethod
    def _rows(count: int) -> Dataset:
        scores = [i / count for i in range(count)]
        return scores, [-1] * (count // 2) + [1] * (count - count // 2)

    def test_warns_past_the_threshold(self, caplog: pytest.LogCaptureFixture) -> None:
        threshold = algorithm.available_algorithms["ls"].data_vol_thresh
        scores, actual_classes = self._rows(threshold + 200)

        with caplog.at_level("WARNING", logger="thresher.dispatch"):
            thresher.Thresher(algorithm="ls").optimize_threshold(scores, actual_classes)

        assert "Linear search is likely to be slow" in caplog.text
        assert f"{threshold:,}" in caplog.text, "the message should say where the limit is"
        assert "exact" in caplog.text, "and point at the faster alternative"

    def test_silent_below_the_threshold(self, caplog: pytest.LogCaptureFixture) -> None:
        scores, actual_classes = self._rows(200)

        with caplog.at_level("WARNING", logger="thresher.dispatch"):
            thresher.Thresher(algorithm="ls").optimize_threshold(scores, actual_classes)

        assert caplog.text == ""

    def test_the_default_algorithm_does_not_warn_on_ordinary_data(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # exact is comfortable well past where the others complain, which is the point.
        scores, actual_classes = self._rows(20_000)

        with caplog.at_level("WARNING", logger="thresher.dispatch"):
            thresher.Thresher().optimize_threshold(scores, actual_classes)

        assert caplog.text == ""

    def test_it_warns_rather_than_refusing(self, caplog: pytest.LogCaptureFixture) -> None:
        threshold = algorithm.available_algorithms["ls"].data_vol_thresh
        scores, actual_classes = self._rows(threshold + 200)

        with caplog.at_level("WARNING", logger="thresher.dispatch"):
            result = thresher.Thresher(algorithm="ls").optimize_threshold(scores, actual_classes)

        assert isinstance(result, float), "the run must still complete"

    def test_every_algorithm_declares_a_threshold(self) -> None:
        for name, entry in algorithm.available_algorithms.items():
            assert entry.data_vol_thresh > 0, f"{name} has no usable threshold"

    def test_the_exhaustive_search_is_the_first_to_complain(self) -> None:
        """Linear search is the only quadratic one, so it should have the lowest bar."""
        thresholds = {name: entry.data_vol_thresh for name, entry in algorithm.available_algorithms.items()}

        assert thresholds["ls"] == min(thresholds.values())
        assert thresholds["exact"] == max(thresholds.values())

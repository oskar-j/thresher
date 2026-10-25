"""The core behaviour: does optimize_threshold find the right cut-off?"""

from collections.abc import Callable

import pytest

import thresher
from thresher import algorithm

Dataset = tuple[list[float], list[int]]
DatasetFactory = Callable[..., Dataset]


def accuracy_of(threshold: float, scores: list[float], actual_classes: list[int]) -> float:
    """Fraction of samples the threshold classifies correctly."""
    return sum(
        1
        for score, actual in zip(scores, actual_classes, strict=True)
        if (1 if score > threshold else -1) == actual
    ) / len(scores)


@pytest.mark.parametrize(
    ("algorithm_name", "algorithm_params", "tolerance"),
    [
        pytest.param(None, {}, 0.0, id="default"),
        pytest.param("exact", {}, 0.0, id="exact"),
        pytest.param("hist", {}, 0.005, id="hist"),
        pytest.param("linear", {}, 0.0, id="linear"),
        pytest.param("sim", {}, 0.05, id="genetic"),
        pytest.param("sgd", {}, 0.08, id="sgd"),
        pytest.param("grid", {}, 0.005, id="grid"),
        pytest.param("sgrid", {"no_of_decimal_places": 2, "stoch_ratio": 0.10}, 0.05, id="sgrid"),
        pytest.param(
            "sgrid",
            {"no_of_decimal_places": 3, "stoch_ratio": 0.06, "reshuffle": True},
            0.08,
            id="sgrid-reshuffle",
        ),
    ],
)
def test_medium_dataset(
    medium_dataset: Dataset,
    algorithm_name: str | None,
    algorithm_params: dict[str, object],
    tolerance: float,
) -> None:
    """Each algorithm has to come within its tolerance of the best accuracy available.

    The tolerance is accuracy given up against the exact optimum - what a caller loses
    by choosing the algorithm - rather than distance along the threshold axis. Until
    0.8.1 this asserted a band of `(0.40, 0.65)` around the optimum, and on this fixture
    0.6499 classifies 77.4% against the optimum's 89.2%: an 11.7-point loss that passed
    (#38). The exact solvers get no slack; the two deterministic approximations get a bin
    width or a grid step of it; the four that sample get what their worst run in 200
    seeds needed, with headroom, so a failure is a change in the solver rather than in
    the draw. `sgd` joined in 0.8.2, having been the one algorithm never run against
    real data; its worst of 200 seeds gave up 5.3%.
    """
    scores, actual_classes = medium_dataset
    kwargs: dict[str, object] = {"algorithm_params": algorithm_params}
    if algorithm_name is not None:
        kwargs["algorithm"] = algorithm_name
    optimum = thresher.Thresher(algorithm="exact").optimize_threshold(scores, actual_classes)
    best = accuracy_of(optimum, scores, actual_classes)

    result = thresher.Thresher(**kwargs).optimize_threshold(scores, actual_classes)

    lost = best - accuracy_of(result, scores, actual_classes)
    assert 0 <= lost <= tolerance, f"gave up {lost:.4f} accuracy against the exact optimum"


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
    """`n_jobs` must leave the answer alone, and the answer has to be the right one.

    `n_jobs=-1` is documented in the README and used to make the chunk size negative.
    Asserting that the result lay inside the score range proved nothing: every candidate
    the search scores lies inside it, so the worst of them would have passed (#38).
    Separable data has exactly one perfect split, which pins the answer.
    """
    scores, actual_classes = separable(200)
    sequential = thresher.Thresher(algorithm="linear").optimize_threshold(scores, actual_classes)

    result = thresher.Thresher(algorithm="linear", algorithm_params={"n_jobs": n_jobs}).optimize_threshold(
        scores, actual_classes
    )

    assert result == sequential
    assert accuracy_of(result, scores, actual_classes) == 1.0


def test_supported_algorithms() -> None:
    supported = thresher.Thresher.get_supported_algorithms()
    assert set(supported) == set(algorithm.available_algorithms)

    as_dict = thresher.Thresher.get_supported_algorithms(as_dict=True)
    assert isinstance(as_dict, dict)
    assert as_dict["grid"] == "Grid search"


class TestSlowInputWarning:
    """Crossing an algorithm's `data_vol_thresh` warns rather than refusing.

    The thresholds are order-of-magnitude guidance measured on one machine, so a caller
    may well have good reason to wait. It is logged rather than `warnings.warn`-ed, so it
    is silenced by asking for less - `set_verbosity('error')` - rather than by a filter.
    """

    @staticmethod
    def _rows(count: int) -> Dataset:
        scores = [i / count for i in range(count)]
        return scores, [-1] * (count // 2) + [1] * (count - count // 2)

    def test_warns_past_the_threshold(self, logs: list[str]) -> None:
        threshold = algorithm.available_algorithms["ls"].data_vol_thresh
        scores, actual_classes = self._rows(threshold + 200)

        thresher.Thresher(algorithm="ls").optimize_threshold(scores, actual_classes)

        text = "".join(logs)
        assert "Linear search is likely to be slow" in text
        assert f"{threshold:,}" in text, "the message should say where the limit is"
        assert "exact" in text, "and point at the faster alternative"

    def test_it_is_a_warning_and_so_survives_the_default_level(self, logs: list[str]) -> None:
        """The one thing this package says without being asked, so it has to be a WARNING.

        Nothing sets a level here: the default is `'warning'`, and a message logged any
        lower would be dropped before it reached a sink.
        """
        scores, actual_classes = self._rows(algorithm.available_algorithms["ls"].data_vol_thresh + 200)

        thresher.Thresher(algorithm="ls").optimize_threshold(scores, actual_classes)

        assert [line for line in logs if line.startswith("WARNING")]

    def test_asking_for_less_silences_it(self, logs: list[str]) -> None:
        """What the message itself tells you to do."""
        scores, actual_classes = self._rows(algorithm.available_algorithms["ls"].data_vol_thresh + 200)

        thresher.Thresher(algorithm="ls", verbosity="error").optimize_threshold(scores, actual_classes)

        assert logs == []

    def test_silent_below_the_threshold(self, logs: list[str]) -> None:
        scores, actual_classes = self._rows(200)

        thresher.Thresher(algorithm="ls").optimize_threshold(scores, actual_classes)

        assert logs == []

    def test_the_default_algorithm_does_not_warn_on_ordinary_data(self, logs: list[str]) -> None:
        # exact is comfortable well past where the others complain, which is the point.
        scores, actual_classes = self._rows(20_000)

        thresher.Thresher().optimize_threshold(scores, actual_classes)

        assert logs == []

    def test_it_warns_rather_than_refusing(self, logs: list[str]) -> None:
        threshold = algorithm.available_algorithms["ls"].data_vol_thresh
        scores, actual_classes = self._rows(threshold + 200)

        result = thresher.Thresher(algorithm="ls").optimize_threshold(scores, actual_classes)

        assert logs, "the warning is what makes this the interesting case"
        assert isinstance(result, float), "the run must still complete"

    def test_every_algorithm_declares_a_threshold(self) -> None:
        for name, entry in algorithm.available_algorithms.items():
            assert entry.data_vol_thresh > 0, f"{name} has no usable threshold"

    def test_the_thresholds_rank_by_how_the_work_actually_grows(self) -> None:
        """The order should follow the algorithms' costs rather than being set arbitrarily.

        Linear search is the only quadratic one, so it complains first. The histogram
        sweep neither sorts nor holds the data, so it lasts longest - further than the
        exact sweep, which keeps one entry per distinct score.
        """
        thresholds = {name: entry.data_vol_thresh for name, entry in algorithm.available_algorithms.items()}

        assert thresholds["ls"] == min(thresholds.values())
        assert thresholds["hist"] == max(thresholds.values())
        assert thresholds["hist"] > thresholds["exact"] > thresholds["ls"]

"""Regression tests for defects fixed in 0.2.1 through 0.2.2.

Every case here raised an exception or returned an invalid result before those releases.
They exercise paths the rest of the suite does not reach: algorithms selected explicitly
rather than by the oracle, small inputs, and cleanly separable data.
"""

import math
import random
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


def accuracy_of(threshold: float, scores: list[float], actual_classes: list[int]) -> float:
    """Fraction of samples the threshold classifies correctly."""
    return sum(
        1
        for score, actual in zip(scores, actual_classes, strict=True)
        if (1 if score > threshold else -1) == actual
    ) / len(scores)


def reading_all_the_data(algorithm_name: str) -> dict[str, float]:
    """`algorithm_params` making a sampling solver read every row it is given.

    The tests below are about *where a solver looks* - whether its candidates are laid
    over the data or over a hardcoded `[0, 1]`. `sgrid`, `sgd` and `gen` answer that
    question through a subsample, and at the default ratios those subsamples are tiny:
    `sgrid` scores each candidate against 5 of 100 rows, or against a single row on the
    three-row inputs here. So the assertions were being decided by the draw as much as by
    the grid, and `sgrid` missed exact accuracy for 43 of 200 starting random states -
    which is a fact about sampling, not about the bug these tests were written for.

    Raising the ratio to 1 removes that interference and leaves the property itself, which
    is the one the test names. It costs nothing: the inputs are at most a hundred rows.
    """
    return {"stoch_ratio": 1.0} if algorithm_name in ("sgrid", "sgd", "gen") else {}


class TestScoresOutsideTheUnitInterval:
    """Scores need not be probabilities, fixed in 0.6.4 (#25).

    `grid` and `sgrid` laid their candidates over a hardcoded `[0, 1]`. Given logits,
    margins or any other scale, every candidate fell outside the data, so the answer was
    whichever edge scored better - chance accuracy, returned without a warning.
    """

    NEGATIVE_RANGE: Dataset = ([-5.0, -4.0, -3.0, -2.0] * 25, [-1, -1, 1, 1] * 25)

    @pytest.mark.parametrize("algorithm_name", ALL_ALGORITHMS)
    def test_every_algorithm_separates_data_far_from_the_unit_interval(self, algorithm_name: str) -> None:
        scores, actual_classes = self.NEGATIVE_RANGE

        result = thresher.Thresher(
            algorithm=algorithm_name, algorithm_params=reading_all_the_data(algorithm_name)
        ).optimize_threshold(scores, actual_classes)

        # The data is cleanly separable, so anything short of 1.0 means the solver never
        # looked where the boundary is. grid/sgrid used to score 0.5 here.
        assert accuracy_of(result, scores, actual_classes) == 1.0

    @pytest.mark.parametrize("algorithm_name", ["grid", "sgrid"])
    @pytest.mark.parametrize("scale", [1e-4, 1000.0])
    def test_the_grid_follows_the_scale_of_the_data(self, algorithm_name: str, scale: float) -> None:
        scores = [value * scale for value in (0.1, 0.2, 0.8, 0.9)] * 25
        actual_classes = [-1, -1, 1, 1] * 25

        result = thresher.Thresher(
            algorithm=algorithm_name, algorithm_params=reading_all_the_data(algorithm_name)
        ).optimize_threshold(scores, actual_classes)

        assert accuracy_of(result, scores, actual_classes) == 1.0

    @pytest.mark.parametrize("algorithm_name", ["grid", "sgrid"])
    def test_the_everything_positive_split_is_still_reachable(self, algorithm_name: str) -> None:
        """A threshold below every score is the only way to express it.

        The old `[0, 1]` grid reached it by accident for data sitting above 0; a grid
        spanning the data has to carry the candidate deliberately.
        """
        scores, actual_classes = [0.1, 0.2, 0.3], [1, 1, -1]

        result = thresher.Thresher(
            algorithm=algorithm_name, algorithm_params=reading_all_the_data(algorithm_name)
        ).optimize_threshold(scores, actual_classes)

        assert result < min(scores)
        assert accuracy_of(result, scores, actual_classes) == pytest.approx(2 / 3)

    @pytest.mark.parametrize("algorithm_name", ["grid", "sgrid"])
    def test_identical_scores_do_not_break_the_grid(self, algorithm_name: str) -> None:
        # min == max leaves no range to divide into candidates.
        result = thresher.Thresher(
            algorithm=algorithm_name, algorithm_params=reading_all_the_data(algorithm_name)
        ).optimize_threshold([0.5] * 6, [-1, -1, -1, 1, 1, 1])

        assert result <= 0.5

    def test_a_tie_keeps_the_threshold_inside_the_data(self, separable: DatasetFactory) -> None:
        # The below-minimum candidate is evaluated last, so it wins only on a strict
        # improvement - the rule `exact` follows.
        scores, actual_classes = separable(400, seed=5)

        result = thresher.Thresher(algorithm="grid").optimize_threshold(scores, actual_classes)

        assert min(scores) <= result <= max(scores)


class TestSgdStepScalesWithTheData:
    """The walk's reach follows the score range, fixed in 0.6.4 (#26).

    The first step was a constant 0.05 and only ever decays, so the walk's total travel
    was bounded at roughly 4.3 score units however far away the boundary was. On data
    spanning thousands it stopped short of the boundary every single run - a deterministic
    starvation, distinct from the sampling noise this solver is already known for.
    """

    @staticmethod
    def imbalanced(scale: float, seed: int = 7) -> Dataset:
        """Separable data at a given scale, with the boundary far above the mean."""
        rng = random.Random(seed)
        scores = [rng.uniform(0, 0.9) * scale for _ in range(900)]
        scores += [rng.uniform(0.95, 1.0) * scale for _ in range(100)]
        return scores, [-1] * 900 + [1] * 100

    @pytest.mark.parametrize("trial", range(4))
    def test_the_answer_scales_exactly_with_the_data(self, trial: int) -> None:
        """Multiplying every score by 1000 must multiply the answer by 1000, and no more.

        Seeding the global RNG makes both runs draw the same subsamples, so any remaining
        difference is the step size failing to follow the data rather than sampling noise.
        Before the fix the larger scale returned a wholly different, far worse threshold.
        """
        random.seed(100 + trial)
        small = thresher.Thresher(algorithm="sgd").optimize_threshold(*self.imbalanced(1.0))
        random.seed(100 + trial)
        large = thresher.Thresher(algorithm="sgd").optimize_threshold(*self.imbalanced(1000.0))

        assert large == pytest.approx(small * 1000.0)

    def test_a_distant_optimum_is_reachable_at_a_large_scale(self) -> None:
        scores, actual_classes = self.imbalanced(1000.0)

        results = [
            thresher.Thresher(algorithm="sgd").optimize_threshold(scores, actual_classes) for _ in range(9)
        ]
        accuracies = sorted(accuracy_of(result, scores, actual_classes) for result in results)

        # The median rather than the worst case: this solver's sampling weakness on a rare
        # class is documented and unrelated, and shows at every scale. What is being pinned
        # is that the walk arrives at all - it used to score ~0.61 here on every run.
        assert accuracies[len(accuracies) // 2] > 0.95

    def test_step_ratio_is_configurable(self) -> None:
        scores, actual_classes = self.imbalanced(1000.0)

        result = thresher.Thresher(algorithm="sgd", algorithm_params={"step_ratio": 0.2}).optimize_threshold(
            scores, actual_classes
        )

        assert min(scores) <= result <= max(scores)

    def test_a_tiny_step_ratio_starves_the_walk(self) -> None:
        """The knob has to reach the walk, not be accepted and ignored.

        A step this small cannot cross the gap between the mean and the boundary within
        the iteration budget, so the result stays near where it started - the old
        behaviour, now reachable only by asking for it.
        """
        scores, actual_classes = self.imbalanced(1000.0)
        starting_point = sum(scores) / len(scores)

        result = thresher.Thresher(algorithm="sgd", algorithm_params={"step_ratio": 1e-9}).optimize_threshold(
            scores, actual_classes
        )

        assert abs(result - starting_point) < abs(max(scores) - starting_point) / 2

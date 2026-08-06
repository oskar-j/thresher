"""The evolutionary solver's selection and mutation mechanics.

Three defects fixed in 0.7.3 (#31), all in how a generation turns into the next one and
into an answer:

* `sus_factor` was subtracted from the population size and the result used directly as a
  slice bound, so values at or above the population either failed with a bare stdlib
  `ValueError` or were silently read as a much gentler cull;
* the mutation was drawn from `[0, mutation_factor)`, so it could only ever raise a trait;
* the answer was the mean of a population bred *after* the last round of scoring, and so
  never measured at all.

The solver samples, so the tests here pin down properties that hold whatever the draw -
an exception raised, a direction taken, a range never left - rather than an exact answer.
"""

import math
import random

import pytest

import thresher
from thresher.algs.common.meta_optimizer import calculate_range_mean
from thresher.algs.genetic.compute import Agent, _mutate
from thresher.exceptions import ConfigurationError, ThresherError

Dataset = tuple[list[float], list[int]]


@pytest.fixture
def overlapping() -> Dataset:
    """Classes that overlap, so the optimum is a real trade rather than a gap."""
    rng = random.Random(3)
    scores = [rng.random() for _ in range(1500)]
    labels = [1 if score > 0.5 else -1 for score in scores]
    for index in range(0, len(labels), 7):
        labels[index] = -labels[index]
    return scores, labels


def accuracy(threshold: float, scores: list[float], actual_classes: list[int]) -> float:
    """Fraction of samples the threshold classifies correctly."""
    return sum(
        1
        for score, actual in zip(scores, actual_classes, strict=True)
        if (1 if score > threshold else -1) == actual
    ) / len(scores)


def solve(overlapping: Dataset, **params: float) -> float:
    """Run the genetic solver on the fixture with the given parameters."""
    return thresher.Thresher(algorithm="gen", algorithm_params=params).optimize_threshold(*overlapping)


class TestParameterValidation:
    """A setting the simulation cannot run on is refused, and refused as ours.

    `survivor_count = population_size - sus_factor` fed a slice, and a slice takes a
    negative bound to mean "from the far end". So asking to cull 35 of 30 agents kept 25
    of them - the opposite of the request, applied in silence - while culling exactly 30
    reached `random.sample` and raised "Sample larger than population", a bare stdlib
    `ValueError` that no `except ThresherError` would catch.
    """

    @pytest.mark.parametrize(
        ("params", "reason"),
        [
            ({"sus_factor": 30}, "culls the whole default population"),
            ({"sus_factor": 35}, "used to be read as a gentler cull of 5"),
            ({"sus_factor": 59}, "used to be read as keeping a single survivor"),
            ({"sus_factor": 61}, "used to reach random.sample and fail there"),
            ({"sus_factor": -5}, "a count of agents cannot be negative"),
            ({"sus_factor": 2.5}, "nor a fraction of one"),
            ({"population_size": 0}, "no agents to evolve"),
            ({"population_size": -1}, "fewer than none"),
            ({"population_size": 2.5}, "not a whole number of agents"),
            ({"number_of_generations": 0}, "nothing would ever be measured"),
            ({"number_of_iterations": 0}, "fitness would be the mean of no samples"),
            ({"population_size": 5, "sus_factor": 5}, "the cull matches a smaller population"),
        ],
    )
    def test_unusable_settings_are_refused(
        self, overlapping: Dataset, params: dict[str, float], reason: str
    ) -> None:
        with pytest.raises(ConfigurationError) as excinfo:
            solve(overlapping, **params)

        # Both bases matter: `ThresherError` so it can be caught precisely, `ValueError`
        # so code written before 0.4.5 - including this package's own command line -
        # still catches it. A bare stdlib ValueError satisfies only the second.
        assert isinstance(excinfo.value, ThresherError), reason
        assert isinstance(excinfo.value, ValueError), reason

    @pytest.mark.parametrize("sus_factor", [0, 1, 29])
    def test_a_cull_that_leaves_someone_to_breed_is_accepted(
        self, overlapping: Dataset, sus_factor: int
    ) -> None:
        """29 of 30 is the boundary: one survivor is enough to cross over with itself."""
        assert isinstance(solve(overlapping, sus_factor=sus_factor), float)

    def test_a_smaller_population_moves_the_boundary_with_it(self, overlapping: Dataset) -> None:
        assert isinstance(solve(overlapping, population_size=5, sus_factor=4), float)

    def test_the_message_names_the_two_values_that_disagree(self, overlapping: Dataset) -> None:
        with pytest.raises(ConfigurationError) as excinfo:
            solve(overlapping, population_size=8, sus_factor=12)

        message = str(excinfo.value)
        assert "12" in message and "8" in message, message


class TestMutationGoesBothWays:
    """It was a ratchet: `mutation_factor * random.random()` is never negative.

    Measured against `exact` over 40 seeds of 4,000 rows, the returned threshold sat
    `+0.0061` above the optimum with mutation off, `+0.0069` at the default chance and
    `+0.0120` at `0.5` - the more often it fired, the further up it pushed.
    """

    @staticmethod
    def _nudges(count: int, mutation_factor: float = 0.5) -> list[float]:
        """Apply one mutation to a flat population `count` times, returning the moves."""
        moves: list[float] = []
        for _ in range(count):
            population = [Agent(id=f"agent_{i}", trait=1.0) for i in range(5)]
            _mutate(population, mutation_factor)
            moves.extend(agent.trait - 1.0 for agent in population if agent.trait != 1.0)
        return moves

    def test_traits_move_down_as_well_as_up(self) -> None:
        random.seed(0)

        moves = self._nudges(400)

        assert any(move < 0 for move in moves), "the nudge is still one-way"
        assert any(move > 0 for move in moves)

    def test_the_nudge_is_bounded_by_the_factor_in_both_directions(self) -> None:
        random.seed(1)

        assert all(abs(move) <= 0.5 for move in self._nudges(400))

    def test_only_one_agent_is_nudged(self) -> None:
        random.seed(2)
        population = [Agent(id=f"agent_{i}", trait=1.0) for i in range(30)]

        _mutate(population, 0.5)

        assert sum(1 for agent in population if agent.trait != 1.0) == 1

    def test_the_bias_no_longer_grows_with_the_mutation_rate(self, overlapping: Dataset) -> None:
        """The signed error against `exact`, at three mutation rates.

        A one-way nudge shows up here as a bias that climbs with the rate. Averaged over
        seeds so a single unlucky run cannot decide it.
        """
        scores, actual_classes = overlapping
        best = thresher.Thresher(algorithm="exact").optimize_threshold(scores, actual_classes)

        biases = {}
        for chance in (0.0, 0.5, 1.0):
            errors = []
            for seed in range(12):
                random.seed(seed)
                errors.append(solve(overlapping, mutation_chance=chance) - best)
            biases[chance] = sum(errors) / len(errors)

        # Before the fix these ran +0.0061, +0.0120 and higher, always positive. The bound
        # is loose enough for sampling noise and far tighter than a ratchet could reach.
        for chance, bias in biases.items():
            assert abs(bias) < 0.01, f"mutation_chance={chance} biased the answer by {bias:+.4f}"


class TestTheAnswerWasMeasured:
    """The returned trait is one the simulation actually scored.

    Each generation evaluated, selected, then bred a fresh population - and after the
    *last* one that fresh population was returned without ever being scored. One crossover
    and one mutation therefore reached the answer with no selection in front of them.
    """

    @pytest.mark.parametrize("mutation_factor", [0.1, 5.0, 50.0])
    def test_a_violent_mutation_cannot_leave_the_data(
        self, overlapping: Dataset, mutation_factor: float
    ) -> None:
        """With the nudge firing every generation, this used to return 1.3188 on [0, 1]."""
        scores, _ = overlapping

        for seed in range(8):
            random.seed(seed)
            result = solve(overlapping, mutation_chance=1.0, mutation_factor=mutation_factor)

            assert min(scores) <= result <= max(scores), f"seed {seed} returned {result}"

    @pytest.mark.parametrize("mutation_factor", [0.1, 5.0, 50.0])
    def test_a_violent_mutation_cannot_cost_much_accuracy(
        self, overlapping: Dataset, mutation_factor: float
    ) -> None:
        """The issue's table: 0.8985 achievable, 0.8383 at factor 5 and 0.5310 at 50."""
        scores, actual_classes = overlapping
        best = thresher.Thresher(algorithm="exact").optimize_threshold(scores, actual_classes)
        achievable = accuracy(best, scores, actual_classes)

        for seed in range(8):
            random.seed(seed)
            result = solve(overlapping, mutation_chance=1.0, mutation_factor=mutation_factor)

            lost = achievable - accuracy(result, scores, actual_classes)
            assert lost < 0.1, f"seed {seed} lost {lost:.4f} of accuracy at factor {mutation_factor}"

    def test_a_single_generation_returns_one_of_the_agents_it_scored(self, overlapping: Dataset) -> None:
        """The sharpest form of it: with one generation there is nothing but the seeds.

        The initial population is drawn between the two class means, so every trait that
        was ever measured lies in that range. Anything outside it can only have come from
        the unscored crossover this used to return.
        """
        scores, actual_classes = overlapping
        lowest, highest = sorted(
            (
                calculate_range_mean(scores, actual_classes, -1),
                calculate_range_mean(scores, actual_classes, 1),
            )
        )

        for seed in range(8):
            random.seed(seed)
            result = solve(overlapping, number_of_generations=1, mutation_chance=1.0, mutation_factor=50.0)

            assert lowest <= result <= highest, f"seed {seed} returned an unmeasured {result}"

    def test_the_answer_is_finite_and_a_float(self, overlapping: Dataset) -> None:
        result = solve(overlapping)

        assert isinstance(result, float)
        assert math.isfinite(result)

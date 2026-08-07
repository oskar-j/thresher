"""An evolutionary algorithm over a population of candidate thresholds.

Each generation scores every agent against random subsamples, discards the least fit, and
breeds replacements by crossover between the survivors, with an occasional mutation. Like
the other stochastic solvers it trades exactness for speed on larger inputs.
"""

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from thresher import log
from thresher.algs.common.meta_optimizer import calculate_range_mean
from thresher.algs.common.stochastic import stochastic_process
from thresher.exceptions import ConfigurationError
from thresher.progress import make_progress
from thresher.utils import get_or_default

population_size_default = 30
number_of_generations_default = 20
number_of_iterations_default = 10
sus_factor_default = 2
stoch_ratio_default = 0.02
mutation_chance_default = 0.05
mutation_factor_default = 0.10

#: Every `algorithm_params` key this solver reads. Anything else is a typo, and is
#: reported as one - see `dispatch.validate_algorithm_params`.
known_params = frozenset(
    {
        "population_size",
        "number_of_generations",
        "number_of_iterations",
        "sus_factor",
        "stoch_ratio",
        "mutation_chance",
        "mutation_factor",
    }
)

INVALID_COUNT = (
    "The '{name}' parameter of the evolutionary algorithm counts {counts}, so it must be a "
    "whole number of at least 1 - got {got!r}. Anything else leaves the simulation nothing "
    "to run on, and used to end in a bare error out of the arithmetic instead."
)

UNUSABLE_SUS_FACTOR = (
    "sus_factor is {got!r}. It counts the least fit agents left child-less at the end of a "
    "generation, so it has to be a whole number and cannot be negative - 0 is the value "
    "that spares all of them, and is what a negative one was silently being read as."
)

INVALID_SUS_FACTOR = (
    "sus_factor is {got} against a population_size of {population}, which leaves "
    "{survivors} agents to breed the next generation from. It counts the least fit agents "
    "left child-less at the end of a generation, so it has to be smaller than the "
    "population - culling all of it leaves nothing to cross over. Values at or above it "
    "used to fail with a bare 'Sample larger than population' from the standard library, "
    "or, where the subtraction went negative and was read as a slice from the far end, to "
    "apply a far gentler cull without saying so."
)


@dataclass
class Agent:
    """A candidate threshold and its measured fitness.

    'samples' and 'fitness' are deliberately separate fields. They were once a single
    key that started as a list of samples and was overwritten with the aggregate, which
    is how the fitness ended up being computed from the wrong value entirely (fixed in
    0.2.1). Keeping them apart makes that class of mistake impossible to express.
    """

    id: str
    trait: float
    samples: list[float] = field(default_factory=list)
    fitness: float = 0.0


def _positive_count(name: str, counts: str, given: Any) -> int:
    """Read one of the simulation's size parameters, refusing what cannot be one.

    Args:
        name: the parameter's name, for the message.
        counts: what it counts, for the message.
        given: the value the caller supplied.

    Returns:
        The value, now known to be a whole number of at least 1.

    Raises:
        ConfigurationError: if it is not. It is a `ValueError`. `bool` is excluded along
            with the non-integers: it *is* an `int` in Python, but `population_size=True`
            is a mistake rather than a request for one agent.
    """
    if not isinstance(given, int) or isinstance(given, bool) or given < 1:
        raise ConfigurationError(INVALID_COUNT.format(name=name, counts=counts, got=given))
    return given


def _survivor_count(population_size: int, sus_factor: Any) -> int:
    """Work out how many agents breed, refusing the settings where that is none.

    `population_size - sus_factor` was used directly as a slice bound, and a slice absorbs
    almost anything: a negative bound counts from the far end, so `sus_factor=35` against a
    population of 30 asked for `[0:-5]` and quietly kept 25 survivors - a far gentler cull
    than the one requested, applied without a word. Where the arithmetic landed on 0, or on
    a bound past the population, `random.sample` raised "Sample larger than population"
    instead: a bare `ValueError` from the standard library rather than anything this
    package defines, and so outside `ThresherError` entirely.

    Args:
        population_size: how many agents each generation holds, already checked.
        sus_factor: how many of the least fit are left child-less.

    Returns:
        The number of agents that go on to breed, always at least 1.

    Raises:
        ConfigurationError: if `sus_factor` is not a whole number, is negative, or would
            leave nothing to cross over. It is a `ValueError`.
    """
    if not isinstance(sus_factor, int) or isinstance(sus_factor, bool) or sus_factor < 0:
        raise ConfigurationError(UNUSABLE_SUS_FACTOR.format(got=sus_factor))

    survivors = population_size - sus_factor
    if survivors < 1:
        raise ConfigurationError(
            INVALID_SUS_FACTOR.format(got=sus_factor, population=population_size, survivors=survivors)
        )

    return survivors


def _mutate(population: list[Agent], mutation_factor: float) -> None:
    """Nudge one agent of the population at random, in either direction.

    Either direction is the point. The nudge was drawn from `[0, mutation_factor)`, so it
    could only ever raise a trait - a ratchet rather than a mutation, and one that pushed
    the returned threshold above the optimum by more the more often it fired. Measured
    against `exact` over 40 seeds of 4,000 rows, the bias went from `+0.0061` at
    `mutation_chance=0` to `+0.0120` at `0.5`, and to `+0.82` where it fired every
    generation with a large factor.

    Args:
        population: the agents to choose from. Mutated in place.
        mutation_factor: the furthest the chosen agent's trait can move, either way.

    Returns:
        None. One agent's `trait` is changed.
    """
    chosen = population[int(len(population) * random.random())]
    chosen.trait += mutation_factor * random.uniform(-1, 1)


def run(
    scores: Sequence[float],
    actual_classes: Sequence[int],
    progress_bar: bool,
    alg_options: Mapping[str, Any],
) -> float:
    """Evolve a population of candidate thresholds and return the fittest one measured.

    The initial population is seeded across the range between the mean score of the
    negative class and that of the positive class, so the search starts where the
    boundary is likely to lie.

    Args:
        scores: the values being split.
        actual_classes: the matching ground-truth classes, as -1 and 1.
        progress_bar: draw a progress bar on stderr, one step per generation. It is not
            drawn while the log is at DEBUG, which is where the per-generation detail
            goes: the two write to the same stream. That rule used to live here and now
            applies to every solver - see `thresher.progress`.
        alg_options: recognised keys, each falling back to its module-level default:
            `population_size` (30) agents per generation; `number_of_generations` (20)
            rounds of selection; `number_of_iterations` (10) fitness samples drawn per
            agent per generation; `sus_factor` (2) how many of the least fit are left
            child-less, and so must be below `population_size`; `stoch_ratio` (0.02)
            fraction of the data each fitness sample reads; `mutation_chance` (0.05)
            probability that one agent per generation is nudged; `mutation_factor` (0.10)
            how far that nudge can move it, in either direction.

    Returns:
        The trait of the fittest agent that was actually measured, across every
        generation. Until 0.7.3 this was the mean of the final population - which is bred
        after the last round of scoring and so never evaluated at all, letting one
        crossover and one mutation reach the answer with no selection in front of them.
        With `mutation_chance=1.0` and `mutation_factor=50` that returned 1.3188 on data
        spanning [0, 1].

    Raises:
        ConfigurationError: if any of the four counts is below 1, or if `sus_factor` would
            leave no agent to breed from. It is a `ValueError`.
    """
    # Every size the simulation runs on is checked before any of it starts, so a setting
    # that cannot work is reported as such rather than as whatever the arithmetic made of
    # it several seconds in.
    population_size = _positive_count(
        "population_size",
        "agents per generation",
        get_or_default(alg_options, "population_size", population_size_default),
    )
    number_of_generations = _positive_count(
        "number_of_generations",
        "rounds of selection",
        get_or_default(alg_options, "number_of_generations", number_of_generations_default),
    )
    number_of_iterations = _positive_count(
        "number_of_iterations",
        "fitness samples drawn per agent per generation",
        get_or_default(alg_options, "number_of_iterations", number_of_iterations_default),
    )
    # how many agents should die child-less after a generation
    survivor_count = _survivor_count(
        population_size, get_or_default(alg_options, "sus_factor", sus_factor_default)
    )

    population_initial_range = (
        calculate_range_mean(scores, actual_classes, -1),
        calculate_range_mean(scores, actual_classes, 1),
    )

    stoch_ratio: float = get_or_default(alg_options, "stoch_ratio", stoch_ratio_default)
    # random ratio - the lower, the faster sim

    mutation_factor: float = get_or_default(alg_options, "mutation_factor", mutation_factor_default)

    # Build the population
    population = [
        Agent(
            id=f"agent_{i}",
            trait=random.uniform(population_initial_range[0], population_initial_range[1]),
        )
        for i in range(population_size)
    ]

    # The answer comes from here rather than from wherever the simulation happens to stop,
    # for the same reason the sgd walk returns its best point rather than its last: fitness
    # is sampled, so a generation can be worse than one already seen.
    best_trait, best_fitness = population[0].trait, math.inf

    # Each generation is bred from the previous one's survivors and then scored, so every
    # population that exists has been measured by the time the loop ends. Breeding at the
    # *end* of a generation instead left the last one unscored, which is where an
    # unmeasured crossover and mutation used to reach the answer.
    survivors: list[Agent] = []

    log.info("Evolving {} agents over {} generations.", population_size, number_of_generations)

    with make_progress(number_of_generations, "Evolving", enabled=progress_bar) as bar:
        for generation_no in range(number_of_generations):
            log.debug("Running generation no {}", generation_no)
            bar.update(generation_no)

            if survivors:
                # do crossover
                population = []
                for i in range(population_size):
                    left = random.sample(survivors, 1)[0].trait
                    right = random.sample(survivors, 1)[0].trait
                    if left > right:
                        left, right = right, left
                    new_trait = left + ((right - left) * random.random())
                    population.append(Agent(id=f"agent_{i}", trait=new_trait))

                if random.random() < get_or_default(alg_options, "mutation_chance", mutation_chance_default):
                    _mutate(population, mutation_factor)

                # Asked before it is built: this line lists every agent's trait, so at any
                # other level it would be a list comprehension over the whole population,
                # formatted and thrown away, once per generation.
                if log.is_enabled_for("debug"):
                    log.debug("Population for gen: {} - {}", generation_no, [_.trait for _ in population])

            for _iteration_no in range(number_of_iterations):
                for agent in population:
                    # for every iteration, get a stochastic fitness score
                    agent.samples.append(
                        stochastic_process(agent.trait, scores, actual_classes, random_factor=stoch_ratio)
                    )

            # calculate fitness score - the mean mis-classification ratio over this
            # generation's iterations, so that lower is fitter
            for agent in population:
                agent.fitness = float(np.mean(agent.samples))
                if agent.fitness < best_fitness:
                    best_trait, best_fitness = agent.trait, agent.fitness

            # select most fit (SUS)
            survivors = sorted(population, key=lambda a: a.fitness)[0:survivor_count]

    log.info("Fittest agent measured: {} at a mis-classification ratio of {}", best_trait, best_fitness)

    return best_trait

"""An evolutionary algorithm over a population of candidate thresholds.

Each generation scores every agent against random subsamples, discards the least fit, and
breeds replacements by crossover between the survivors, with an occasional mutation. Like
the other stochastic solvers it trades exactness for speed on larger inputs.
"""

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from thresher.algs.common.meta_optimizer import calculate_range_mean
from thresher.algs.common.stochastic import stochastic_process
from thresher.utils import get_or_default, print_progress_bar

population_size_default = 30
number_of_generations_default = 20
number_of_iterations_default = 10
sus_factor_default = 2
stoch_ratio_default = 0.02
mutation_chance_default = 0.05
mutation_factor_default = 0.10


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


def run(
    scores: Sequence[float],
    actual_classes: Sequence[int],
    verbose: bool,
    progress_bar: bool,
    alg_options: Mapping[str, Any],
) -> float:
    """Evolve a population of candidate thresholds and return the population's mean.

    The initial population is seeded across the range between the mean score of the
    negative class and that of the positive class, so the search starts where the
    boundary is likely to lie.

    Args:
        scores: the values being split.
        actual_classes: the matching ground-truth classes, as -1 and 1.
        verbose: print the population after each generation. Enabling this disables the
            progress bar, since the two would fight over the terminal.
        progress_bar: draw a progress bar on stdout, one step per generation.
        alg_options: recognised keys, each falling back to its module-level default:
            `population_size` (30) agents per generation; `number_of_generations` (20)
            rounds of selection; `number_of_iterations` (10) fitness samples drawn per
            agent per generation; `sus_factor` (2) how many of the least fit are left
            child-less; `stoch_ratio` (0.02) fraction of the data each fitness sample
            reads; `mutation_chance` (0.05) probability that one agent per generation is
            nudged; `mutation_factor` (0.10) the size of that nudge.

    Returns:
        The mean trait of the final population. Because it averages the survivors rather
        than picking the single fittest, a population that has not converged pulls the
        result toward the middle of its spread.
    """
    if verbose and progress_bar:
        print("Warning! Enabling verbosity automatically disables a progress bar.")
        progress_bar = False

    # Defining the population size
    population_size: int = get_or_default(alg_options, "population_size", population_size_default)
    population_initial_range = (
        calculate_range_mean(scores, actual_classes, -1),
        calculate_range_mean(scores, actual_classes, 1),
    )

    number_of_generations: int = get_or_default(
        alg_options, "number_of_generations", number_of_generations_default
    )
    number_of_iterations: int = get_or_default(
        alg_options, "number_of_iterations", number_of_iterations_default
    )

    survivor_count = population_size - get_or_default(alg_options, "sus_factor", sus_factor_default)
    # how many agents should die child-less after a generation

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

    for generation_no in range(number_of_generations):
        if verbose:
            print(f"Running generation no {generation_no}")

        if progress_bar:
            print_progress_bar(generation_no, number_of_generations)

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

        # select most fit (SUS)
        sort_by_fit = sorted(population, key=lambda a: a.fitness)[0:survivor_count]

        # do crossover
        population = []
        for i in range(population_size):
            left = random.sample(sort_by_fit, 1)[0].trait
            right = random.sample(sort_by_fit, 1)[0].trait
            if left > right:
                left, right = right, left
            new_trait = left + ((right - left) * random.random())
            population.append(Agent(id=f"agent_{i}", trait=new_trait))

        # mutate
        if random.random() < get_or_default(alg_options, "mutation_chance", mutation_chance_default):
            population[int(len(population) * random.random())].trait += mutation_factor * random.random()

        if verbose:
            print(f"Population after gen: {generation_no} - {[_.trait for _ in population]}")

    if progress_bar:
        print_progress_bar(number_of_generations, number_of_generations)

    return float(np.mean([_.trait for _ in population]))

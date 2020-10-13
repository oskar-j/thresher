import random
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


def run(scores, actual_classes, verbose, progress_bar, alg_options) -> float:
    if verbose and progress_bar:
        print('Warning! Enabling verbosity automatically disables a progress bar.')
        progress_bar = False

    # Defining the population size
    population_size = get_or_default(alg_options, 'population_size', population_size_default)
    population_initial_range = (calculate_range_mean(scores, actual_classes, -1),
                                calculate_range_mean(scores, actual_classes, 1))

    number_of_generations = get_or_default(alg_options, 'number_of_generations', number_of_generations_default)
    number_of_iterations = get_or_default(alg_options, 'number_of_iterations', number_of_iterations_default)

    sus_factor = population_size - get_or_default(alg_options, 'sus_factor', sus_factor_default)
    # how many agents should die child-less after a generation

    stoch_ratio = get_or_default(alg_options, 'stoch_ratio', stoch_ratio_default)
    # random ratio - the lower, the faster sim

    mutation_factor = get_or_default(alg_options, 'mutation_factor', mutation_factor_default)

    population = []

    # Build the population
    for i in range(population_size):
        population.append({'id': f'agent_{i}',
                           'trait': random.uniform(population_initial_range[0], population_initial_range[1]),
                           'trait_eff': list()})

    for generation_no in range(number_of_generations):
        if verbose:
            print(f'Running generation no {generation_no}')

        if progress_bar:
            print_progress_bar(generation_no, number_of_generations)

        for iteration_no in range(number_of_iterations):
            for agent in population:
                agent['trait_eff'].append(stochastic_process(agent['trait'],
                                                             scores, actual_classes, random_factor=stoch_ratio))
                # for every iteration, get a stochastic fitness score

        # calculate fitness score
        for agent in population:
            agent['trait_eff'] = np.mean(agent['trait'])

        # select most fit (SUS)
        sort_by_fit = [_ for _ in sorted(population, key=lambda x: x['trait_eff'], reverse=False)][0:sus_factor]

        # do crossover
        population = []
        for i in range(population_size):
            l = random.sample(sort_by_fit, 1)[0]['trait']
            r = random.sample(sort_by_fit, 1)[0]['trait']
            if l > r:
                l, r = r, l
            # new_trait = random.sample([l, l + ((r-l)*random.random()), r], 1)[0]
            new_trait = l + ((r-l)*random.random())
            population.append({'id': f'agent_{i}',
                               'trait': new_trait,
                               'trait_eff': list()})

        # mutate
        if random.random() < get_or_default(alg_options, 'mutation_chance', mutation_chance_default):
            population[int(len(population) * random.random())]['trait'] += mutation_factor * random.random()

        if verbose:
            print(f'Population after gen: {generation_no} - {[_["trait"] for _ in population]}')

    if progress_bar:
        print_progress_bar(number_of_generations, number_of_generations)

    return float(np.mean([_["trait"] for _ in population]))

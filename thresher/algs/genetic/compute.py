import random
import numpy as np
from thresher.algs.common.stochastic import stochastic_process


def run(scores, actual_classes, verbose, progress_bar) -> float:
    # Defining the population size
    population_size = 30

    number_of_generations = 20
    number_of_iterations = 10

    sus_factor = population_size - 2  # how many agents should die child-less after a generation
    stoch_ratio = 0.02  # random ratio - the lower, the faster sim

    population = []

    # Build the population
    for i in range(population_size):
        population.append({'id': f'agent_{i}',
                           'trait': random.random(),
                           'trait_eff': list()})

    for generation_no in range(number_of_generations):
        if verbose:
            print(f'Running generation no {generation_no}')

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
        if random.random() < 0.05:
            population[int(len(population) * random.random())]['trait'] += random.randrange(1, 10) / 100.0

        if verbose:
            print(f'Population after gen: {generation_no} - {[_["trait"] for _ in population]}')

    return float(np.mean([_["trait"] for _ in population]))

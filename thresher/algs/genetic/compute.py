import random

from thresher.algs.common.stochastic import stochastic_process


def run(scores, actual_classes, verbose, progress_bar) -> float:
    # Defining the population size
    population_size = 20

    number_of_generations = 20
    number_of_iterations = 40

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
                agent['trait_eff'].append(stochastic_process(agent['trait'], scores, actual_classes, random_factor=0.01))
                # for every iteration, get a stochastic fitness score

        # calculate fitness score

        # select most fit (SUS)

        # do crossover

        # mutate
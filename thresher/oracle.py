from thresher import algorithm
from thresher.algs.linear import compute as linear_compute
from thresher.algs.sgd import compute as sgd_compute
from thresher.algs.genetic import compute as gen_compute
from thresher.algs.grid import compute as grid_compute
from thresher.exceptions import UNKNOWN_ALGORITHM


LINEAR_ALGORITHM = algorithm.available_algorithms['ls']
STOCHASTIC_GRADIENT_DESCENT = algorithm.available_algorithms['sgd']
GENETIC_ALGORITHM = algorithm.available_algorithms['gen']
GRID_SEARCH_ALGORITHM = algorithm.available_algorithms['grid']
STOCHASTIC_GRID_SEARCH_ALGORITHM = algorithm.available_algorithms['sgrid']


def run_oracle(data_traits: dict):
    data_volume = data_traits['data_length']

    # some the 'ThresherPerformanceTest.ipynb' notebook for some thought process behind this
    # an algorithm of recommendation for big datasets is currently 'sgd'

    if data_volume <= algorithm.available_algorithms['ls'].data_vol_thresh:
        chosen_algorithm = algorithm.available_algorithms['ls']
    elif data_volume <= algorithm.available_algorithms['grid'].data_vol_thresh:
        chosen_algorithm = algorithm.available_algorithms['grid']
    else:
        chosen_algorithm = algorithm.available_algorithms['sgd']

    return chosen_algorithm


def run_computations(chosen_algorithm: algorithm.Algorithm, scores, actual_classes,
                     verbose, progress_bar, allow_parallel, alg_options) -> float:
    assert set(actual_classes) == {-1, 1}

    if verbose:
        print(f'Executing the {chosen_algorithm.full_name} algorithm... please wait for the result.')

    if chosen_algorithm == LINEAR_ALGORITHM:
        if allow_parallel and ('n_jobs' in alg_options) and (alg_options['n_jobs'] != 1):
            return linear_compute.run_parallel(scores, actual_classes, verbose, alg_options['n_jobs'])
        else:
            return linear_compute.run(scores, actual_classes, verbose, progress_bar)
    elif chosen_algorithm == STOCHASTIC_GRADIENT_DESCENT:
        return sgd_compute.run(scores, actual_classes, verbose, progress_bar, alg_options)
    elif chosen_algorithm == GENETIC_ALGORITHM:
        return gen_compute.run(scores, actual_classes, verbose, progress_bar, alg_options)
    elif chosen_algorithm == GRID_SEARCH_ALGORITHM:
        return grid_compute.run(scores, actual_classes, verbose, progress_bar, alg_options)
    elif chosen_algorithm == STOCHASTIC_GRID_SEARCH_ALGORITHM:
        return grid_compute.run_stoch(scores, actual_classes, verbose, progress_bar, alg_options)
    else:
        raise NotImplementedError(UNKNOWN_ALGORITHM)

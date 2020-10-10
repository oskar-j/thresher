from thresher import algorithm
from thresher.algs.linear import compute as linear_compute
from thresher.algs.sgd import compute as sgd_compute


LINEAR_ALGORITHM = algorithm.available_algorithms['ls']
STOCHASTIC_GRADIENT_DESCENT = algorithm.available_algorithms['sgd']


def run_oracle(data_traits: dict):
    data_volume = data_traits['data_length']

    if data_volume <= algorithm.available_algorithms['ls'].data_vol_thresh:
        chosen_algorithm = algorithm.available_algorithms['ls']
    elif data_volume <= algorithm.available_algorithms['sgd'].data_vol_thresh:
        chosen_algorithm = algorithm.available_algorithms['sgd']
    else:
        chosen_algorithm = algorithm.available_algorithms['gen']

    return chosen_algorithm


def run_computations(chosen_algorithm: algorithm.Algorithm, scores, actual_classes, verbose, progress_bar) -> float:
    assert set(actual_classes) == {-1, 1}

    if verbose:
        print(f'Executing the {chosen_algorithm.full_name} algorithm... please wait for the result.')

    if chosen_algorithm == LINEAR_ALGORITHM:
        return linear_compute.run(scores, actual_classes, verbose, progress_bar)
    elif chosen_algorithm == STOCHASTIC_GRADIENT_DESCENT:
        return sgd_compute.run(scores, actual_classes, verbose, progress_bar)

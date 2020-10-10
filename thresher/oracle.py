from thresher import algorithm
from thresher.algs.linear import compute as linear_compute


LINEAR_ALGORITHM = algorithm.available_algorithms['ls']


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
    if chosen_algorithm == LINEAR_ALGORITHM:
        assert set(actual_classes) == {-1, 1}
        print(f'Executing the {chosen_algorithm.full_name} algorithm... please wait for the result.')
        return linear_compute.run(scores, actual_classes, verbose, progress_bar)

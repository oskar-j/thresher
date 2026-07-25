from collections.abc import Mapping, Sequence
from typing import Any

from thresher import algorithm
from thresher.algs.genetic import compute as gen_compute
from thresher.algs.grid import compute as grid_compute
from thresher.algs.linear import compute as linear_compute
from thresher.algs.sgd import compute as sgd_compute
from thresher.exceptions import UNKNOWN_ALGORITHM
from thresher.utils import validate_actual_classes

LINEAR_ALGORITHM = algorithm.available_algorithms["ls"]
STOCHASTIC_GRADIENT_DESCENT = algorithm.available_algorithms["sgd"]
GENETIC_ALGORITHM = algorithm.available_algorithms["gen"]
GRID_SEARCH_ALGORITHM = algorithm.available_algorithms["grid"]
STOCHASTIC_GRID_SEARCH_ALGORITHM = algorithm.available_algorithms["sgrid"]


def run_oracle(data_traits: Mapping[str, Any]) -> algorithm.Algorithm:
    """Pick an algorithm from the traits of the input data.

    The bounds come from each algorithm's 'data_vol_thresh', so changing those values in
    'algorithm.py' changes routing here.
    """
    data_volume = data_traits["data_length"]

    # some the 'ThresherPerformanceTest.ipynb' notebook for some thought process behind this
    # an algorithm of recommendation for big datasets is currently 'sgd'

    linear_threshold = LINEAR_ALGORITHM.data_vol_thresh
    grid_threshold = GRID_SEARCH_ALGORITHM.data_vol_thresh
    assert linear_threshold is not None and grid_threshold is not None

    if data_volume <= linear_threshold:
        return LINEAR_ALGORITHM
    if data_volume <= grid_threshold:
        return GRID_SEARCH_ALGORITHM
    return STOCHASTIC_GRADIENT_DESCENT


def run_computations(
    chosen_algorithm: algorithm.Algorithm,
    scores: Sequence[float],
    actual_classes: Sequence[int],
    verbose: bool,
    progress_bar: bool,
    allow_parallel: bool,
    alg_options: Mapping[str, Any],
) -> float:
    validate_actual_classes(actual_classes)

    if verbose:
        print(f"Executing the {chosen_algorithm.full_name} algorithm... please wait for the result.")

    if chosen_algorithm == LINEAR_ALGORITHM:
        if allow_parallel and ("n_jobs" in alg_options) and (alg_options["n_jobs"] != 1):
            return linear_compute.run_parallel(scores, actual_classes, verbose, alg_options["n_jobs"])
        return linear_compute.run(scores, actual_classes, verbose, progress_bar)
    if chosen_algorithm == STOCHASTIC_GRADIENT_DESCENT:
        return sgd_compute.run(scores, actual_classes, verbose, progress_bar, alg_options)
    if chosen_algorithm == GENETIC_ALGORITHM:
        return gen_compute.run(scores, actual_classes, verbose, progress_bar, alg_options)
    if chosen_algorithm == GRID_SEARCH_ALGORITHM:
        return grid_compute.run(scores, actual_classes, verbose, progress_bar, alg_options)
    if chosen_algorithm == STOCHASTIC_GRID_SEARCH_ALGORITHM:
        return grid_compute.run_stoch(scores, actual_classes, verbose, progress_bar, alg_options)
    raise NotImplementedError(UNKNOWN_ALGORITHM)

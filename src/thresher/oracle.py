"""Algorithm selection and dispatch.

Two responsibilities sit here: `run_oracle` picks an algorithm from the shape of the data,
and `run_computations` routes to the chosen implementation. This is the only module that
imports the individual solvers.
"""

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

    The bounds come from each algorithm's `data_vol_thresh`, so changing those values in
    `algorithm.py` changes the routing here. The reasoning behind the current bounds is in
    `examples/performance_test/ThresherPerformanceTest.ipynb`.

    Args:
        data_traits: measurements of the input. Only `data_length`, the number of scores,
            is consulted today.

    Returns:
        Linear search at 1,000 rows or fewer, where an exact search is affordable; grid
        search up to 50,000; stochastic gradient descent above that, where only a
        subsample-based solver stays cheap.

    Raises:
        TypeError: if either routing threshold is unset in the registry, which would make
            the comparisons below meaningless.
    """
    data_volume = data_traits["data_length"]

    linear_threshold = LINEAR_ALGORITHM.data_vol_thresh
    grid_threshold = GRID_SEARCH_ALGORITHM.data_vol_thresh
    if linear_threshold is None or grid_threshold is None:
        # A plain check rather than an assert, so it survives `python -O`.
        raise TypeError("The 'ls' and 'grid' algorithms must both define a data_vol_thresh.")

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
    """Validate the labels, then run the chosen algorithm.

    Args:
        chosen_algorithm: the algorithm to run, as an `Algorithm` from the registry.
        scores: the values being split.
        actual_classes: the matching ground-truth classes. Must already be normalized to
            -1 and 1; `Thresher.optimize_threshold` does that before calling here.
        verbose: print progress information.
        progress_bar: draw a progress bar on stdout, where the solver supports one.
        allow_parallel: permit multiprocessing. Only linear search acts on this, and only
            when `alg_options` also carries an `n_jobs` other than 1.
        alg_options: the user's `algorithm_params`. Each solver reads the keys it knows
            and silently ignores the rest.

    Returns:
        The threshold chosen by the algorithm.

    Raises:
        ValueError: if the labels are empty, single-class, or outside (-1, 1).
        NotImplementedError: if `chosen_algorithm` has no dispatch branch here - which is
            what happens when an algorithm is added to the registry but not wired up.
    """
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

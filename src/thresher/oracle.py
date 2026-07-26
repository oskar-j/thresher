"""Algorithm selection and dispatch.

Two responsibilities sit here: `run_oracle` picks an algorithm from the shape of the data,
and `run_computations` routes to the chosen implementation. This is the only module that
imports the individual solvers.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from thresher import algorithm
from thresher.algs.exact import compute as exact_compute
from thresher.algs.genetic import compute as gen_compute
from thresher.algs.grid import compute as grid_compute
from thresher.algs.linear import compute as linear_compute
from thresher.algs.sgd import compute as sgd_compute
from thresher.backends import Backend, LocalBackend
from thresher.exceptions import UNKNOWN_ALGORITHM
from thresher.utils import validate_actual_classes

EXACT_ALGORITHM = algorithm.available_algorithms["exact"]
LINEAR_ALGORITHM = algorithm.available_algorithms["ls"]
STOCHASTIC_GRADIENT_DESCENT = algorithm.available_algorithms["sgd"]
GENETIC_ALGORITHM = algorithm.available_algorithms["gen"]
GRID_SEARCH_ALGORITHM = algorithm.available_algorithms["grid"]
STOCHASTIC_GRID_SEARCH_ALGORITHM = algorithm.available_algorithms["sgrid"]


def run_oracle(data_traits: Mapping[str, Any]) -> algorithm.Algorithm:
    """Pick an algorithm from the traits of the input data.

    Always `exact` since 0.4.0. The oracle used to trade accuracy against input size,
    routing to linear search below 1,000 rows, grid search below 50,000 and stochastic
    gradient descent above that, because the only exact algorithm available was O(n²) and
    became unaffordable. `exact` removed that trade-off: it is exact at every size *and*
    cheaper than the approximations it replaced, so there is nothing left to weigh up.

    The other algorithms remain selectable by name. This function is kept, rather than
    inlined into the caller, because a future algorithm might reintroduce a genuine
    trade-off - a metric other than accuracy, say - and this is where that decision
    belongs.

    Args:
        data_traits: measurements of the input. `data_length`, the number of scores, is
            no longer consulted.

    Returns:
        The exact sweep.
    """
    return EXACT_ALGORITHM


def run_computations(
    chosen_algorithm: algorithm.Algorithm,
    scores: Sequence[float],
    actual_classes: Sequence[int],
    verbose: bool,
    progress_bar: bool,
    allow_parallel: bool,
    alg_options: Mapping[str, Any],
    backend: Backend | None = None,
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
        backend: where the counting happens. `exact`, `ls` and `grid` accept it; the
            stochastic and sequential solvers do not, because distributing them would
            change their sampling and so their answers. They run in-process whatever is
            passed.

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

    resolved_backend = backend or LocalBackend()

    if chosen_algorithm == EXACT_ALGORITHM:
        return exact_compute.run(scores, actual_classes, verbose, progress_bar, alg_options, resolved_backend)
    if chosen_algorithm == LINEAR_ALGORITHM:
        # An explicit n_jobs asks for local multiprocessing, which predates backends and
        # would only contend with a cluster, so a non-local backend takes precedence.
        wants_processes = allow_parallel and ("n_jobs" in alg_options) and (alg_options["n_jobs"] != 1)
        if wants_processes and resolved_backend.name == "local":
            return linear_compute.run_parallel(scores, actual_classes, verbose, alg_options["n_jobs"])
        return linear_compute.run(scores, actual_classes, verbose, progress_bar, resolved_backend)
    if chosen_algorithm == STOCHASTIC_GRADIENT_DESCENT:
        return sgd_compute.run(scores, actual_classes, verbose, progress_bar, alg_options)
    if chosen_algorithm == GENETIC_ALGORITHM:
        return gen_compute.run(scores, actual_classes, verbose, progress_bar, alg_options)
    if chosen_algorithm == GRID_SEARCH_ALGORITHM:
        return grid_compute.run(
            scores, actual_classes, verbose, progress_bar, alg_options, backend=resolved_backend
        )
    if chosen_algorithm == STOCHASTIC_GRID_SEARCH_ALGORITHM:
        return grid_compute.run_stoch(scores, actual_classes, verbose, progress_bar, alg_options)
    raise NotImplementedError(UNKNOWN_ALGORITHM)

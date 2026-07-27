"""Dispatch: validate the input, then run the chosen algorithm.

This is the only module that imports the individual solvers.

It was `oracle.py` until 0.5.0. The oracle chose an algorithm from the size of the input,
because the only exact algorithm was O(n²) and stopped being affordable - so accuracy had
to be traded against volume. `exact` removed that trade-off in 0.4.0 by being exact at
every size *and* cheaper than the approximations, at which point the oracle had nothing
left to decide and was announced for removal. The algorithm is now settled when a
`Thresher` is built, not per call.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from thresher import algorithm
from thresher.algs.exact import compute as exact_compute
from thresher.algs.genetic import compute as gen_compute
from thresher.algs.grid import compute as grid_compute
from thresher.algs.histogram import compute as hist_compute
from thresher.algs.linear import compute as linear_compute
from thresher.algs.sgd import compute as sgd_compute
from thresher.backends import Backend, LocalBackend
from thresher.exceptions import AlgorithmNotWiredError
from thresher.utils import validate_actual_classes, validate_lengths

logger = logging.getLogger(__name__)

SLOW_FOR_THIS_MUCH_DATA = (
    "%s is likely to be slow on %s rows - it is usually comfortable up to about %s. "
    "The 'exact' algorithm is exact and O(n log n), and is the default for this reason. "
    "Silence this with logging.getLogger('thresher').setLevel(logging.ERROR)."
)

EXACT_ALGORITHM = algorithm.available_algorithms["exact"]
HISTOGRAM_ALGORITHM = algorithm.available_algorithms["hist"]
LINEAR_ALGORITHM = algorithm.available_algorithms["ls"]
STOCHASTIC_GRADIENT_DESCENT = algorithm.available_algorithms["sgd"]
GENETIC_ALGORITHM = algorithm.available_algorithms["gen"]
GRID_SEARCH_ALGORITHM = algorithm.available_algorithms["grid"]
STOCHASTIC_GRID_SEARCH_ALGORITHM = algorithm.available_algorithms["sgrid"]


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

    Logs:
        A warning if the input is larger than the chosen algorithm's `data_vol_thresh`,
        naming a faster alternative. It is not raised, because the thresholds are guidance
        rather than limits.

    Raises:
        ValueError: if the labels are empty, single-class, outside (-1, 1), or a different
            length from the scores.
        AlgorithmNotWiredError: if `chosen_algorithm` has no dispatch branch here - what
            happens when an algorithm is added to the registry but not wired up. It is a
            `NotImplementedError`.
    """
    validate_lengths(scores, actual_classes)
    validate_actual_classes(actual_classes)

    # A warning rather than a refusal: the thresholds are order-of-magnitude guidance from
    # one machine, and a caller may well have reason to wait.
    if len(scores) > chosen_algorithm.data_vol_thresh:
        logger.warning(
            SLOW_FOR_THIS_MUCH_DATA,
            chosen_algorithm.full_name,
            f"{len(scores):,}",
            f"{chosen_algorithm.data_vol_thresh:,}",
        )

    if verbose:
        print(f"Executing the {chosen_algorithm.full_name} algorithm... please wait for the result.")

    resolved_backend = backend or LocalBackend()

    if chosen_algorithm == EXACT_ALGORITHM:
        return exact_compute.run(scores, actual_classes, verbose, progress_bar, alg_options, resolved_backend)
    if chosen_algorithm == HISTOGRAM_ALGORITHM:
        return hist_compute.run(scores, actual_classes, verbose, progress_bar, alg_options)
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
    raise AlgorithmNotWiredError

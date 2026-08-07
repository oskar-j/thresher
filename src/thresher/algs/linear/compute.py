"""Linear search: evaluate every candidate threshold exactly.

The most accurate solver, and the reference the others are measured against, but it costs
O(n^2) - each of the n-1 candidate thresholds is scored against all n samples. Superseded
by `exact`, which returns the same answer, or a marginally better one, in O(n log n).
"""

from collections.abc import Sequence

from thresher import log
from thresher.backends import Backend, LocalBackend, MultiprocessingBackend
from thresher.backends.mp_backend import resolve_worker_count
from thresher.exceptions import InsufficientDataError
from thresher.progress import make_progress
from thresher.utils import pairwise

#: The one `algorithm_params` key linear search reads. It is consulted by
#: `run_computations` rather than here - it selects between `run` and `run_parallel`
#: instead of being passed into either - but it belongs to this algorithm and is
#: documented under it. See `dispatch.validate_algorithm_params`.
known_params = frozenset({"n_jobs"})


def run_parallel(scores: Sequence[float], actual_classes: Sequence[int], n_jobs: int) -> float:
    """Run the linear search across several processes.

    Selected by `run_computations` when `allow_parallel` is set and `n_jobs != 1`.

    Since 0.7.0 this is the ordinary search running on the `mp` backend, rather than a
    second implementation of it. Two things follow. The answer no longer depends on
    whether the search was parallelised: this used to evaluate the raw scores as
    thresholds where the sequential path evaluates the midpoints between them, so the two
    returned different - though equally valid - answers for the same data. And the
    `__main__` guard that separate processes need is now enforced with an explanation
    instead of hanging.

    Args:
        scores: the values being split.
        actual_classes: the matching ground-truth classes, as -1 and 1.
        n_jobs: number of worker processes, or -1 for every available processor bar one.

    Returns:
        The threshold with the highest accuracy - the same one the sequential path finds.

    Raises:
        ConfigurationError: if `n_jobs` is 0 or below -1. It is a `ValueError`.
        ParallelBootstrapError: if the workers could not start, which usually means a
            missing `__main__` guard. It is a `RuntimeError`.
    """
    backend = MultiprocessingBackend(num_workers=n_jobs)

    log.info(
        "Doing linear search with {} scores, running in parallel over {} processes.",
        len(scores),
        resolve_worker_count(n_jobs),
    )

    # No bar: the work happens in other processes, and there is no single loop here to
    # step one - see the note in `thresher.progress`.
    return run(scores, actual_classes, progress_bar=False, backend=backend)


def run(
    scores: Sequence[float],
    actual_classes: Sequence[int],
    progress_bar: bool,
    backend: Backend | None = None,
) -> float:
    """Evaluate the midpoint between every pair of adjacent scores, exactly.

    Unlike the other solvers this one takes no `alg_options`; its only parameter, `n_jobs`,
    selects `run_parallel` instead.

    Args:
        scores: the values being split.
        actual_classes: the matching ground-truth classes, as -1 and 1.
        progress_bar: draw a progress bar on stderr. Since 0.4.2 the candidates are scored
            in one batch, so this brackets the work rather than advancing through it.
        backend: where the counting happens. Defaults to in-process.

    Returns:
        The midpoint threshold with the highest accuracy. Where several tie, the first one
        found wins.

    Raises:
        InsufficientDataError: if fewer than two scores were given, leaving no midpoint
            to evaluate. It is a `ValueError`.
    """
    batch_size = len(scores)

    log.info(
        "Doing linear search with {} iterations. It can take some time, depending on the data volume.",
        batch_size,
    )

    # Every midpoint between adjacent sorted scores, duplicates included, exactly as
    # before - scoring them is now one batched call instead of a nested loop.
    candidates = [(low + high) / 2 for low, high in pairwise(sorted(scores))]

    if not candidates:
        # 'pairwise' yields nothing for fewer than two scores, so there was no candidate
        # threshold to evaluate at all.
        raise InsufficientDataError("At least two scores are needed to evaluate a threshold.")

    # One batched call, so the bar brackets it rather than stepping through it: 0% while
    # the counting runs, 100% when it returns.
    with make_progress(batch_size, "Linear search", enabled=progress_bar) as bar:
        bar.update(0)
        tallies = (backend or LocalBackend()).tally_candidates(candidates, scores, actual_classes)

    # max() over indices returns the first maximum, keeping the original tie-breaking.
    return candidates[max(range(len(tallies)), key=tallies.__getitem__)]

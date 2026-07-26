"""Linear search: evaluate every candidate threshold exactly.

The most accurate solver, and the reference the others are measured against, but it costs
O(n^2) - each of the n-1 candidate thresholds is scored against all n samples. The oracle
picks it only for inputs of 1,000 rows or fewer.
"""

import multiprocessing as mp
from collections.abc import Iterator, Sequence
from functools import partial

from thresher.backends import Backend, LocalBackend
from thresher.utils import pairwise, print_progress_bar


def process_batch(
    scores: Sequence[float], actual_classes: Sequence[int], data_point: float
) -> tuple[float, float]:
    """Score one candidate threshold against the whole dataset.

    Defined at module level rather than as a closure because it is the function handed to
    `multiprocessing.Pool.map`, and workers have to be able to pickle it.

    Args:
        scores: the values being split.
        actual_classes: the matching ground-truth classes, as -1 and 1.
        data_point: the candidate threshold to evaluate.

    Returns:
        A `(threshold, accuracy)` pair, where accuracy is the fraction of samples the
        threshold classifies correctly. The threshold is passed back out so the caller can
        identify results arriving from workers out of order.
    """
    count_correct, count_incorrect = 0, 0

    for score, actual in zip(scores, actual_classes, strict=False):
        predicted = 1 if score > data_point else -1
        if predicted == actual:
            count_correct += 1
        else:
            count_incorrect += 1

    accuracy = count_correct / (count_correct + count_incorrect)

    return data_point, accuracy


def run_parallel(scores: Sequence[float], actual_classes: Sequence[int], verbose: bool, n_jobs: int) -> float:
    """Run the linear search across several processes.

    Selected by `run_computations` when `allow_parallel` is set and `n_jobs != 1`. Note
    that this evaluates the scores themselves as thresholds, whereas the single-process
    `run` evaluates the midpoints between adjacent scores, so the two can return slightly
    different - equally valid - answers.

    Args:
        scores: the values being split.
        actual_classes: the matching ground-truth classes, as -1 and 1.
        verbose: print progress information.
        n_jobs: number of worker processes, or -1 for every available processor bar one.

    Returns:
        The threshold with the highest accuracy.
    """
    batch_size = len(scores)
    number_of_processors = mp.cpu_count()

    if (n_jobs < -1) or (n_jobs > number_of_processors):
        print(
            "Improper value for n_jobs. It must be either -1, or at most, the number of available processors"
        )

    # Resolve n_jobs=-1 to a real process count first, and derive the chunk size from
    # that. Dividing the batch by n_jobs directly makes the chunk size negative when
    # n_jobs is -1, which makes pool.map return Nones.
    number_of_processes = max(1, number_of_processors - 1 if n_jobs == -1 else n_jobs)
    chunk_size = max(1, batch_size // number_of_processes)

    if verbose:
        print(
            f"Doing linear search with {batch_size} iterations, "
            f"running in parallel over {number_of_processes} processes."
        )

    def iterate_through_scores() -> Iterator[float]:
        """Feed the scores to the pool one at a time, without copying the sequence.

        Yields:
            Each score in turn, as a candidate threshold for a worker to evaluate.
        """
        yield from scores

    mp_func = partial(process_batch, scores, actual_classes)
    with mp.Pool(processes=number_of_processes) as pool:
        results = pool.map(func=mp_func, iterable=iterate_through_scores(), chunksize=chunk_size)

    return next(i[0] for i in sorted(results, key=lambda x: x[1], reverse=True))


def run(
    scores: Sequence[float],
    actual_classes: Sequence[int],
    verbose: bool,
    progress_bar: bool,
    backend: Backend | None = None,
) -> float:
    """Evaluate the midpoint between every pair of adjacent scores, exactly.

    Unlike the other solvers this one takes no `alg_options`; its only parameter, `n_jobs`,
    selects `run_parallel` instead.

    Args:
        scores: the values being split.
        actual_classes: the matching ground-truth classes, as -1 and 1.
        verbose: print progress information.
        progress_bar: draw a progress bar on stdout. Since 0.4.2 the candidates are scored
            in one batch, so this brackets the work rather than advancing through it.
        backend: where the counting happens. Defaults to in-process.

    Returns:
        The midpoint threshold with the highest accuracy. Where several tie, the first one
        found wins.

    Raises:
        ValueError: if fewer than two scores were given, leaving no midpoint to evaluate.
    """
    batch_size = len(scores)

    if verbose:
        print(
            f"Doing linear search with {batch_size} iterations. "
            f"It can take some time, depending on the data volume."
        )

    # Every midpoint between adjacent sorted scores, duplicates included, exactly as
    # before - scoring them is now one batched call instead of a nested loop.
    candidates = [(low + high) / 2 for low, high in pairwise(sorted(scores))]

    if not candidates:
        # 'pairwise' yields nothing for fewer than two scores, so there was no candidate
        # threshold to evaluate at all.
        raise ValueError("At least two scores are needed to evaluate a threshold.")

    if progress_bar:
        print_progress_bar(0, batch_size)

    tallies = (backend or LocalBackend()).tally_candidates(candidates, scores, actual_classes)

    if progress_bar:
        print_progress_bar(batch_size, batch_size)

    # max() over indices returns the first maximum, keeping the original tie-breaking.
    return candidates[max(range(len(tallies)), key=tallies.__getitem__)]

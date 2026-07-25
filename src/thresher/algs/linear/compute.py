import multiprocessing as mp
from collections.abc import Iterator, Sequence
from functools import partial

from thresher.utils import pairwise, print_progress_bar


def process_batch(
    scores: Sequence[float], actual_classes: Sequence[int], data_point: float
) -> tuple[float, float]:
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
        yield from scores

    mp_func = partial(process_batch, scores, actual_classes)
    with mp.Pool(processes=number_of_processes) as pool:
        results = pool.map(func=mp_func, iterable=iterate_through_scores(), chunksize=chunk_size)

    return next(i[0] for i in sorted(results, key=lambda x: x[1], reverse=True))


def run(scores: Sequence[float], actual_classes: Sequence[int], verbose: bool, progress_bar: bool) -> float:
    best_threshold: float | None = None
    best_accuracy: float = -1.0

    batch_size = len(scores)

    if verbose:
        print(
            f"Doing linear search with {batch_size} iterations. "
            f"It can take some time, depending on the data volume."
        )

    for iteration, (a, b) in enumerate(pairwise(sorted(scores)), start=1):
        if progress_bar:
            print_progress_bar(iteration, batch_size)

        middle = (a + b) / 2

        count_correct, count_incorrect = 0, 0

        # strict=False keeps the historical behaviour of stopping at the shorter input.
        # See "Silent wrong answers" in CLAUDE.md - making this strict would reject
        # mismatched lengths, which is a deliberate behaviour change, not a lint fix.
        for score, actual in zip(scores, actual_classes, strict=False):
            predicted = 1 if score > middle else -1
            if predicted == actual:
                count_correct += 1
            else:
                count_incorrect += 1

        accuracy = count_correct / (count_correct + count_incorrect)

        if accuracy > best_accuracy:
            best_threshold, best_accuracy = middle, accuracy

    if progress_bar:
        print_progress_bar(batch_size, batch_size)

    if best_threshold is None:
        # 'pairwise' yields nothing for fewer than two scores, so there was no candidate
        # threshold to evaluate at all.
        raise ValueError("At least two scores are needed to evaluate a threshold.")

    return best_threshold

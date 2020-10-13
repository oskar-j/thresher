from thresher.utils import pairwise, print_progress_bar
import multiprocessing as mp
from functools import partial


def process_batch(scores, actual_classes, data_point):
    count_correct, count_incorrect = 0, 0

    for l, r in zip(scores, actual_classes):
        predicted = 1 if l > data_point else -1
        if predicted == r:
            count_correct += 1
        else:
            count_incorrect += 1

    accuracy = count_correct / (count_correct + count_incorrect)

    return data_point, accuracy


def run_parallel(scores, actual_classes, verbose, n_jobs) -> float:
    batch_size = len(scores)
    number_of_processors = mp.cpu_count()

    if (n_jobs < -1) or (n_jobs > number_of_processors):
        print(f'Improper value for n_jobs. It must be either -1, or at most, the number of available processors')

    if verbose:
        print(f'Doing linear search with {batch_size} iterations, running in parallel {n_jobs} jobs.')

    def iterate_through_scores():
        for score in scores:
            yield score

    pool = mp.Pool(processes=number_of_processors-1 if n_jobs == -1 else n_jobs)
    mp_func = partial(process_batch, scores, actual_classes)
    results = pool.map(func=mp_func, iterable=iterate_through_scores(), chunksize=int(batch_size/n_jobs))

    return next(i[0] for i in sorted(results, key=lambda x: x[1], reverse=True))


def run(scores, actual_classes, verbose, progress_bar) -> float:
    best_threshold, best_accuracy, iteration = None, -1, 0

    batch_size = len(scores)

    if verbose:
        print(f'Doing linear search with {batch_size} iterations. It can take some time, depending on the data volume.')

    for a, b in pairwise(sorted(scores)):
        iteration += 1

        if progress_bar:
            print_progress_bar(iteration, batch_size)

        middle = (a + b) / 2

        count_correct, count_incorrect = 0, 0

        for l, r in zip(scores, actual_classes):
            predicted = 1 if l > middle else -1
            if predicted == r:
                count_correct += 1
            else:
                count_incorrect += 1

        accuracy = count_correct / (count_correct + count_incorrect)

        if accuracy > best_accuracy:
            best_threshold, best_accuracy = middle, accuracy

    if progress_bar:
        print_progress_bar(batch_size, batch_size)

    return best_threshold

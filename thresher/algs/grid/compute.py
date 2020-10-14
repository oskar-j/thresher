import numpy as np
from thresher.utils import get_or_default, print_progress_bar

no_of_decimal_places_default = 2


def run_stoch(scores, actual_classes, verbose, progress_bar, alg_options):
    return run(scores, actual_classes, verbose, progress_bar, alg_options, True)


def run(scores, actual_classes, verbose, progress_bar, alg_options, stochastic=False):
    best_threshold, best_accuracy, iteration = None, -1, 0

    no_of_decimal_places = get_or_default(alg_options, 'no_of_decimal_places', no_of_decimal_places_default)
    batch_size = (10**no_of_decimal_places)+1

    for single_point in np.linspace(0, 1, batch_size):
        iteration += 1

        if progress_bar:
            print_progress_bar(iteration, batch_size)

        count_correct, count_incorrect = 0, 0

        for l, r in zip(scores, actual_classes):
            predicted = 1 if l > single_point else -1
            if predicted == r:
                count_correct += 1
            else:
                count_incorrect += 1

        accuracy = count_correct / (count_correct + count_incorrect)

        if accuracy > best_accuracy:
            best_threshold, best_accuracy = single_point, accuracy

    if progress_bar:
        print_progress_bar(batch_size, batch_size)

    return best_threshold

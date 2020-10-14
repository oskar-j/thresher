import numpy as np
from thresher.utils import get_or_default, print_progress_bar
import random

no_of_decimal_places_default = 2
stoch_ratio_default = 0.05
reshuffle_default = False


def run_stoch(scores, actual_classes, verbose, progress_bar, alg_options):
    return run(scores, actual_classes, verbose, progress_bar, alg_options, True)


def run(scores, actual_classes, verbose, progress_bar, alg_options, stochastic=False):
    best_threshold, best_accuracy, iteration = None, -1, 0

    no_of_decimal_places = get_or_default(alg_options, 'no_of_decimal_places', no_of_decimal_places_default)
    stoch_ratio = get_or_default(alg_options, 'stoch_ratio', stoch_ratio_default)
    reshuffle = get_or_default(alg_options, 'reshuffle', reshuffle_default)

    batch_size = (10**no_of_decimal_places)+1

    if verbose:
        print(f'Evaluating {batch_size} solutions. Please wait for results.')

    def get_random_projection(_scores, _actual_classes, _stoch_ratio):
        return random.sample(list(zip(_scores, _actual_classes)), int(_stoch_ratio * len(_scores)))

    if stochastic and (not reshuffle):
        one_time_projection = get_random_projection(scores, actual_classes, stoch_ratio)

    for single_point in np.linspace(0, 1, batch_size):
        iteration += 1

        if progress_bar:
            print_progress_bar(iteration, batch_size)

        count_correct, count_incorrect = 0, 0

        if stochastic:
            if reshuffle:
                projection = get_random_projection(scores, actual_classes, stoch_ratio)
            else:
                projection = one_time_projection
        else:
            projection = zip(scores, actual_classes)

        for l, r in projection:
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

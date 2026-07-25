"""Subsample-based scoring, shared by the stochastic solvers."""

import random
from collections.abc import Sequence


def stochastic_process(
    evaluated: float,
    scores: Sequence[float],
    actual_classes: Sequence[int],
    random_factor: float,
    miss_class: bool = True,
) -> float:
    """Evaluate a candidate threshold against a random subsample of the data.

    This is the shared basis for the speed of the `sgd` and `genetic` solvers on large
    inputs: neither ever reads the whole dataset to score a candidate.

    Args:
        evaluated: the candidate threshold to score.
        scores: the values being split.
        actual_classes: the matching ground-truth classes, as -1 and 1.
        random_factor: fraction of the data to sample, between 0 and 1. The sample is
            floored at one item, so small inputs still produce a usable ratio.
        miss_class: return the mis-classification ratio when True, the accuracy when
            False.

    Returns:
        The ratio of mis-classified samples in the subsample - so lower is fitter - or
        the fraction classified correctly if `miss_class` is False. Being a subsample,
        repeated calls with the same threshold return slightly different values.
    """
    population_size = len(scores)

    # int() alone floors to 0 for small inputs (e.g. the 'gen' default of 0.02 does so
    # below 50 rows), which yields an empty sample and a division by zero below.
    sample_size = min(max(1, int(random_factor * population_size)), population_size)

    sample = random.sample(range(population_size), sample_size)
    number_of_correct, number_of_incorrect = 0, 0
    for idx in sample:
        element = scores[idx]
        actual_class = actual_classes[idx]
        pred = 1 if element > evaluated else -1
        if pred == actual_class:
            number_of_correct += 1
        else:
            number_of_incorrect += 1

    if miss_class:
        return number_of_incorrect / (number_of_incorrect + number_of_correct)  # ratio of mis-class
    return number_of_correct / (number_of_incorrect + number_of_correct)

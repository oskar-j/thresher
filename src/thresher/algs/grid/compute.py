import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from thresher.utils import get_or_default, print_progress_bar

no_of_decimal_places_default = 2
stoch_ratio_default = 0.05
reshuffle_default = False


def _get_random_projection(
    scores: Sequence[float], actual_classes: Sequence[int], stoch_ratio: float
) -> list[tuple[float, int]]:
    # int() alone floors to 0 for small inputs (the default ratio of 0.05 does so
    # below 20 rows), which yields an empty projection and a division by zero below.
    sample_size = min(max(1, int(stoch_ratio * len(scores))), len(scores))
    return random.sample(list(zip(scores, actual_classes, strict=False)), sample_size)


def run_stoch(
    scores: Sequence[float],
    actual_classes: Sequence[int],
    verbose: bool,
    progress_bar: bool,
    alg_options: Mapping[str, Any],
) -> float:
    return run(scores, actual_classes, verbose, progress_bar, alg_options, True)


def run(
    scores: Sequence[float],
    actual_classes: Sequence[int],
    verbose: bool,
    progress_bar: bool,
    alg_options: Mapping[str, Any],
    stochastic: bool = False,
) -> float:
    best_threshold: float | None = None
    best_accuracy: float = -1.0
    iteration = 0

    no_of_decimal_places: int = get_or_default(
        alg_options, "no_of_decimal_places", no_of_decimal_places_default
    )
    stoch_ratio: float = get_or_default(alg_options, "stoch_ratio", stoch_ratio_default)
    reshuffle: bool = get_or_default(alg_options, "reshuffle", reshuffle_default)

    batch_size = (10**no_of_decimal_places) + 1

    if verbose:
        print(f"Evaluating {batch_size} solutions. Please wait for results.")

    one_time_projection: list[tuple[float, int]] | None = None
    if stochastic and not reshuffle:
        one_time_projection = _get_random_projection(scores, actual_classes, stoch_ratio)

    for iteration, single_point in enumerate(np.linspace(0, 1, batch_size), start=1):
        if progress_bar:
            print_progress_bar(iteration, batch_size)

        count_correct, count_incorrect = 0, 0

        projection: Iterable[tuple[float, int]]
        if stochastic:
            if reshuffle:
                projection = _get_random_projection(scores, actual_classes, stoch_ratio)
            else:
                assert one_time_projection is not None
                projection = one_time_projection
        else:
            # strict=False preserves the historical behaviour; see the note in
            # linear/compute.py and "Silent wrong answers" in CLAUDE.md.
            projection = zip(scores, actual_classes, strict=False)

        for score, actual in projection:
            predicted = 1 if score > single_point else -1
            if predicted == actual:
                count_correct += 1
            else:
                count_incorrect += 1

        accuracy = count_correct / (count_correct + count_incorrect)

        if accuracy > best_accuracy:
            best_threshold, best_accuracy = float(single_point), accuracy

    if progress_bar:
        print_progress_bar(batch_size, batch_size)

    if best_threshold is None:
        raise ValueError("The grid produced no candidate thresholds to evaluate.")

    return best_threshold

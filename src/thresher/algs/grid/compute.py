"""Grid search, in exhaustive and stochastic form.

Both share this one implementation: `run` evaluates every point on a fixed grid over
[0, 1] against the whole dataset, and `run_stoch` evaluates each point against a random
subsample instead. Cost depends on the grid resolution rather than the input size, which
is why the oracle prefers grid search for mid-sized inputs.
"""

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
    """Draw a random subsample of the dataset to evaluate a candidate against.

    Args:
        scores: the values being split.
        actual_classes: the matching ground-truth classes, as -1 and 1.
        stoch_ratio: fraction of the data to sample, between 0 and 1.

    Returns:
        A list of `(score, actual_class)` pairs, always holding at least one pair and
        never more than the input size.
    """
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
    """Run the grid search against a random subsample rather than the full dataset.

    This is the `sgrid` algorithm - a thin wrapper that passes `stochastic=True` into
    `run`, since the two share an implementation.

    Args:
        scores: the values being split.
        actual_classes: the matching ground-truth classes, as -1 and 1.
        verbose: print progress information.
        progress_bar: draw a progress bar on stdout.
        alg_options: may hold `no_of_decimal_places`, `stoch_ratio` and `reshuffle`.

    Returns:
        The grid point with the highest measured accuracy.
    """
    return run(scores, actual_classes, verbose, progress_bar, alg_options, True)


def run(
    scores: Sequence[float],
    actual_classes: Sequence[int],
    verbose: bool,
    progress_bar: bool,
    alg_options: Mapping[str, Any],
    stochastic: bool = False,
) -> float:
    """Evaluate every point on a fixed grid over [0, 1] and keep the best.

    Args:
        scores: the values being split.
        actual_classes: the matching ground-truth classes, as -1 and 1.
        verbose: print progress information.
        progress_bar: draw a progress bar on stdout.
        alg_options: recognised keys, each falling back to its module-level default:
            `no_of_decimal_places` (2) sets the grid resolution - the grid holds
            `10**places + 1` points, so 2 gives 101 candidates at 0.01 apart;
            `stoch_ratio` (0.05) is the fraction of data sampled per candidate, used only
            when `stochastic`; `reshuffle` (False) draws a fresh sample for every
            candidate instead of reusing one, again only when `stochastic`.
        stochastic: score each candidate against a subsample rather than all the data.

    Returns:
        The grid point with the highest measured accuracy. Note the grid always spans
        [0, 1], so scores outside that range are only ever split at its edges.

    Raises:
        ValueError: if the grid yielded no candidates at all.
    """
    best_threshold: float | None = None
    best_accuracy: float = -1.0

    no_of_decimal_places: int = get_or_default(
        alg_options, "no_of_decimal_places", no_of_decimal_places_default
    )
    stoch_ratio: float = get_or_default(alg_options, "stoch_ratio", stoch_ratio_default)
    reshuffle: bool = get_or_default(alg_options, "reshuffle", reshuffle_default)

    batch_size = (10**no_of_decimal_places) + 1

    if verbose:
        print(f"Evaluating {batch_size} solutions. Please wait for results.")

    # Drawn once when every candidate is to be judged against the same subsample; left
    # empty, and never read, in the other two modes.
    one_time_projection: list[tuple[float, int]] = (
        _get_random_projection(scores, actual_classes, stoch_ratio) if stochastic and not reshuffle else []
    )

    for iteration, single_point in enumerate(np.linspace(0, 1, batch_size), start=1):
        if progress_bar:
            print_progress_bar(iteration, batch_size)

        count_correct, count_incorrect = 0, 0

        projection: Iterable[tuple[float, int]]
        if not stochastic:
            # strict=False preserves the historical behaviour; see the note in
            # linear/compute.py and "Silent wrong answers" in CLAUDE.md.
            projection = zip(scores, actual_classes, strict=False)
        elif reshuffle:
            projection = _get_random_projection(scores, actual_classes, stoch_ratio)
        else:
            projection = one_time_projection

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

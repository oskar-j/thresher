"""Grid search, in exhaustive and stochastic form.

Both share this one implementation: `run` evaluates every point on a grid spanning the
data against the whole dataset, and `run_stoch` evaluates each point against a random
subsample instead. Cost depends on the grid resolution rather than the input size, which
once made it a good middle ground; `exact` is cheaper still and does not approximate.

Until 0.6.4 the grid spanned [0, 1] whatever the data was, on the assumption that scores
are probabilities. For anything else - logits, margins, distances - every candidate fell
outside the data and the answer was one of the two edges, at chance accuracy and without
a warning. The grid is now derived from the scores themselves, so the algorithm works on
any scale.
"""

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from thresher.backends import Backend, LocalBackend
from thresher.exceptions import InsufficientDataError
from thresher.utils import get_or_default, print_progress_bar

no_of_decimal_places_default = 2
stoch_ratio_default = 0.05
reshuffle_default = False

#: The `algorithm_params` keys each of the two algorithms sharing this file reads.
#: `stoch_ratio` and `reshuffle` are consulted only on the stochastic path, so passing
#: them to the exhaustive `grid` does nothing - which is a typo's failure mode, and is
#: reported as one. See `dispatch.validate_algorithm_params`.
known_params = frozenset({"no_of_decimal_places"})
known_params_stoch = known_params | {"stoch_ratio", "reshuffle"}


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
    # Sampling indices rather than materialised pairs, as `stochastic_process` does.
    # Building the pairs first cost O(n) per candidate however small the sample, which
    # made `reshuffle` slower than the exhaustive search it approximates.
    return [
        (scores[index], actual_classes[index]) for index in random.sample(range(len(scores)), sample_size)
    ]


def _build_grid(scores: Sequence[float], batch_size: int) -> list[float]:
    """Lay `batch_size` evenly spaced candidates across the range of the data.

    Args:
        scores: the values being split.
        batch_size: how many evenly spaced points to place.

    Returns:
        The candidates, in the order they should be evaluated: the even grid from the
        lowest score to the highest, then one point below the lowest. That last one is
        the only way to express "classify everything as positive", and it comes last so
        that the first-maximum tie-breaking both paths use prefers a threshold inside
        the data - the same rule `exact` follows.
    """
    lowest, highest = min(scores), max(scores)
    if lowest == highest:
        # No range to divide: every threshold at or above the value splits the same way,
        # and one below it is the only alternative.
        return [lowest, math.nextafter(lowest, -math.inf)]

    candidates = [float(point) for point in np.linspace(lowest, highest, batch_size)]
    candidates.append(math.nextafter(lowest, -math.inf))
    return candidates


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
    backend: Backend | None = None,
) -> float:
    """Evaluate every point on a grid spanning the data and keep the best.

    Args:
        scores: the values being split.
        actual_classes: the matching ground-truth classes, as -1 and 1.
        verbose: print progress information.
        progress_bar: draw a progress bar on stdout.
        alg_options: recognised keys, each falling back to its module-level default:
            `no_of_decimal_places` (2) sets the grid resolution - the grid holds
            `10**places + 1` evenly spaced points, so 2 gives 101 of them, spanning the
            data rather than a fixed interval; `stoch_ratio` (0.05) is the fraction of
            data sampled per candidate, used only when `stochastic`; `reshuffle` (False)
            draws a fresh sample for every candidate instead of reusing one, again only
            when `stochastic`.
        stochastic: score each candidate against a subsample rather than all the data.
        backend: where the counting happens. Defaults to in-process, and is used only for
            the exhaustive path - the stochastic one draws its own subsamples, which
            sharding would change.

    Returns:
        The grid point with the highest measured accuracy. The grid spans
        `[min(scores), max(scores)]`, so the resolution is spent on the range the data
        actually occupies whatever its scale; one further candidate below the minimum
        expresses "classify everything as positive". Ties go to the leftmost candidate,
        which keeps the answer inside the data unless the edge is strictly better.

    Raises:
        InsufficientDataError: if the grid yielded no candidates at all. It is a
            `ValueError`.
    """
    best_threshold: float | None = None
    best_accuracy: float = -1.0

    no_of_decimal_places: int = get_or_default(
        alg_options, "no_of_decimal_places", no_of_decimal_places_default
    )
    stoch_ratio: float = get_or_default(alg_options, "stoch_ratio", stoch_ratio_default)
    reshuffle: bool = get_or_default(alg_options, "reshuffle", reshuffle_default)

    batch_size = (10**no_of_decimal_places) + 1

    if not scores:
        raise InsufficientDataError("The grid produced no candidate thresholds to evaluate.")

    candidates = _build_grid(scores, batch_size)
    total = len(candidates)

    if verbose:
        print(f"Evaluating {total} solutions over [{min(scores)}, {max(scores)}]. Please wait for results.")

    if not stochastic:
        # The exhaustive path is exactly "score these candidates, keep the best", which is
        # what a backend parallelises. max() over indices takes the first maximum, which
        # is the tie-breaking the loop below also used.
        if progress_bar:
            print_progress_bar(0, total)
        tallies = (backend or LocalBackend()).tally_candidates(candidates, scores, actual_classes)
        if progress_bar:
            print_progress_bar(total, total)
        return candidates[max(range(len(tallies)), key=tallies.__getitem__)]

    # Drawn once when every candidate is to be judged against the same subsample; left
    # empty, and never read, in the other two modes.
    one_time_projection: list[tuple[float, int]] = (
        _get_random_projection(scores, actual_classes, stoch_ratio) if stochastic and not reshuffle else []
    )

    for iteration, single_point in enumerate(candidates, start=1):
        if progress_bar:
            print_progress_bar(iteration, total)

        count_correct, count_incorrect = 0, 0

        projection: Iterable[tuple[float, int]]
        if reshuffle:
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
        print_progress_bar(total, total)

    if best_threshold is None:
        raise InsufficientDataError("The grid produced no candidate thresholds to evaluate.")

    return best_threshold

"""Approximate the optimal threshold from binned class counts.

The exact sweep sorts the scores and then walks them. Sorting is what costs it `O(n log n)`
and what forces it to hold the data. This trades a little precision for neither: divide the
score range into a fixed number of bins, count the classes falling into each in one pass,
and sweep the *bins* with exactly the same running-total argument the exact sweep uses over
distinct scores.

    correct(j) = (negatives in bins below j) + (positives in bins from j upwards)

Nothing is sorted, no row is revisited, and the only thing held is the bin counts - so the
memory is set by the bin count rather than the input size, and stays flat as the data grows.

The cost is resolution. A threshold can only be placed on a bin edge, so the answer is off
by at most one bin width, which `no_of_bins` controls directly and predictably. That makes
it the one approximation here whose error is bounded rather than statistical: it is not
sampling, so repeated runs on the same data return the same answer.

Where `grid` also evaluates a fixed set of candidates, it rescans every row for each one -
`O(c·n)`. This reads each row once, whatever the resolution.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any

from thresher.exceptions import InsufficientDataError
from thresher.utils import POSITIVE_LABEL, get_or_default, print_progress_bar

no_of_bins_default = 1024


def run(
    scores: Sequence[float],
    actual_classes: Sequence[int],
    verbose: bool,
    progress_bar: bool,
    alg_options: Mapping[str, Any],
) -> float:
    """Find a near-optimal threshold from binned counts.

    Args:
        scores: the values being split.
        actual_classes: the matching ground-truth classes, as -1 and 1.
        verbose: print progress information.
        progress_bar: draw a progress bar on stdout.
        alg_options: recognised keys, falling back to the module-level default:
            `no_of_bins` (1024) sets the resolution. The returned threshold is within one
            bin width of the best one, so doubling this halves the worst-case error and
            costs one more counter per bin - and nothing per row.

    Returns:
        The best threshold the binning can express: a bin edge, or a value just below the
        smallest score where classifying everything positive wins. Within one bin width of
        what `exact` would return, and identical across runs on the same data.

    Raises:
        InsufficientDataError: if no scores were given, or `no_of_bins` is below one.
    """
    if not scores:
        raise InsufficientDataError("At least one score is needed to evaluate a threshold.")

    bins: int = get_or_default(alg_options, "no_of_bins", no_of_bins_default)
    if bins < 1:
        raise InsufficientDataError(f"no_of_bins must be at least 1, got {bins}.")

    lowest = min(scores)
    highest = max(scores)
    span = highest - lowest

    if verbose:
        print(f"Binning {len(scores)} scores over [{lowest}, {highest}] into {bins} bins.")

    negatives = [0] * bins
    positives = [0] * bins

    # The one pass. Every row is read here and never looked at again.
    for score, actual in zip(scores, actual_classes, strict=False):
        # Every score shares one bin when they are all equal; otherwise the highest would
        # land one past the end, so it shares the last bin.
        index = 0 if span == 0 else min(int((score - lowest) / span * bins), bins - 1)
        if actual == POSITIVE_LABEL:
            positives[index] += 1
        else:
            negatives[index] += 1

    total_positive = sum(positives)

    # Everything classified positive: the only split that needs a threshold below the data.
    best_correct = total_positive
    best_threshold = math.nextafter(lowest, -math.inf)

    negatives_behind = 0
    positives_behind = 0

    for index in range(bins):
        negatives_behind += negatives[index]
        positives_behind += positives[index]

        if progress_bar:
            print_progress_bar(index + 1, bins)

        # A threshold at this bin's upper edge: everything up to here is predicted
        # negative, everything above it positive.
        correct = negatives_behind + (total_positive - positives_behind)

        if correct > best_correct:
            best_correct = correct
            best_threshold = lowest + span * (index + 1) / bins

    if progress_bar:
        print_progress_bar(bins, bins)

    if verbose:
        print(f"Best threshold {best_threshold} classifies {best_correct}/{len(scores)} correctly.")

    return best_threshold

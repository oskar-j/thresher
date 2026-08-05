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

#: Every `algorithm_params` key this solver reads. Anything else is a typo, and is
#: reported as one - see `dispatch.validate_algorithm_params`.
known_params = frozenset({"no_of_bins"})


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
        index = bin_index(score, lowest, span, bins)
        if actual == POSITIVE_LABEL:
            positives[index] += 1
        else:
            negatives[index] += 1

    best_threshold, best_correct = sweep_bins(
        negatives, positives, lowest=lowest, highest=highest, progress_bar=progress_bar
    )

    if verbose:
        print(f"Best threshold {best_threshold} classifies {best_correct}/{len(scores)} correctly.")

    return best_threshold


def bin_index(score: float, lowest: float, span: float, bins: int) -> int:
    """Place one score in its bin.

    The single definition of the binning, so the counting pass and the threshold that
    reports on it cannot drift apart - which is exactly how they used to disagree.

    Args:
        score: the value to place.
        lowest: the smallest score the bins cover.
        span: the distance from the smallest score to the largest.
        bins: how many bins there are.

    Returns:
        The bin this score falls in. Every score shares one bin when they are all equal;
        otherwise the highest would land one past the end, so it shares the last bin.
    """
    if span == 0:
        return 0
    return min(int((score - lowest) / span * bins), bins - 1)


def _boundary_threshold(lowest: float, span: float, boundary: int, bins: int) -> float:
    """The threshold that separates bin `boundary` from the one below it, exactly.

    The binning floors, so a score sitting on an edge belongs to the bin *above* it, while
    the prediction rule `score > threshold` sends a score sitting on the threshold to the
    class *below* it. Returning the edge itself therefore misclassified precisely the
    samples on that edge, against the counting that chose it (#20).

    Recomputing the edge as `lowest + span * boundary / bins` and stepping one value below
    it is close but not exact: that expression and the `(score - lowest) / span * bins`
    used to bin are not inverses in floating point, so a score can bin above the edge and
    still compare below it. This searches for the boundary with the binning function
    itself, which cannot disagree with the counting by construction.

    Args:
        lowest: the smallest score the bins cover.
        span: the distance from the smallest score to the largest.
        boundary: the first bin to be predicted positive.
        bins: how many bins there are in total.

    Returns:
        The largest value that still bins below `boundary`. Every score in `boundary` or
        above is strictly greater than it, and every score below is not - which is the
        split the counting assumed.
    """
    if span == 0:
        # One occupied bin, so the only split at or above the data is the value itself.
        return lowest

    below, at_or_above = lowest, lowest + span
    # Invariant: `below` bins under `boundary`, `at_or_above` bins at or over it. Halving
    # the interval keeps that true, and floats between two neighbours run out quickly.
    while math.nextafter(below, math.inf) < at_or_above:
        middle = below + (at_or_above - below) / 2
        if middle <= below or middle >= at_or_above:
            break
        if bin_index(middle, lowest, span, bins) >= boundary:
            at_or_above = middle
        else:
            below = middle

    return below


def sweep_bins(
    negatives: Sequence[int],
    positives: Sequence[int],
    lowest: float,
    highest: float,
    progress_bar: bool = False,
) -> tuple[float, int]:
    """Find the best threshold from binned class counts.

    The deciding half of the algorithm, kept separate from the counting half. It needs only
    the per-bin totals and the range they cover, which is a summary whose size is the bin
    count and nothing else - so whatever produced those counts, a single pass here or an
    aggregation across a cluster, this function makes the decision and the answers agree.

    Args:
        negatives: count of negative samples in each bin, lowest bin first.
        positives: count of positive samples in each bin, in the same order.
        lowest: the smallest score the bins cover.
        highest: the largest. The two are taken rather than a span because `lowest + span`
            does not always reconstruct it: for scores rounded to a few decimal places -
            the ordinary case - `0.065 + (0.997 - 0.065)` is `0.9969999999999999`, and a
            threshold there classifies the largest samples positive while the counting
            below has them negative. `span` is derived from the pair, once.
        progress_bar: draw a progress bar on stdout while sweeping.

    Returns:
        The best threshold expressible on a bin edge, and how many samples it classifies
        correctly - a count the threshold is guaranteed to achieve.

    Raises:
        InsufficientDataError: if there are no bins to sweep.
    """
    bins = len(negatives)
    if bins == 0 or bins != len(positives):
        raise InsufficientDataError("Bin counts are missing or do not line up.")

    span = highest - lowest

    total_positive = sum(positives)

    # Everything classified positive: the only split that needs a threshold below the data.
    # `None` stands for it, since it sits below every bin rather than between two.
    best_correct = total_positive
    best_index: int | None = None

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
            best_index = index

    if progress_bar:
        print_progress_bar(bins, bins)

    # Resolved once, for the winner only - the search below is far too expensive to run
    # for every candidate, and only the chosen split needs a threshold at all.
    if best_index is None:
        best_threshold = math.nextafter(lowest, -math.inf)
    elif best_index == bins - 1:
        # The topmost edge is the largest score itself: nothing exceeds it, which is how
        # "classify everything as negative" is expressed. Taken as given rather than
        # rebuilt from the span - see the note on `highest` above.
        best_threshold = highest
    else:
        best_threshold = _boundary_threshold(lowest, span, best_index + 1, bins)

    return best_threshold, best_correct

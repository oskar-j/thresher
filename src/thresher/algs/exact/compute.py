"""Exact threshold search in O(n log n), by sorting once and sweeping.

Linear search is O(n²) because it recomputes the whole confusion matrix from scratch for
every candidate threshold. That work is almost entirely redundant: moving the threshold
past a single sample changes the number of correct predictions by exactly one, in a
direction fixed by that sample's class. So the counts can be carried along the sweep and
updated in constant time, which removes the inner loop altogether.

Sort the samples by score, then walk them in order. At each position, everything to the
left is predicted negative and everything to the right positive, and

    correct(k) = (negatives among the first k) + (positives among the remaining n - k)

Both terms are running totals. The whole search is therefore one pass after the sort, and
the sort dominates: O(n log n) time, O(n) space.

This is the standard exact splitter used to choose a decision-stump threshold, and the
same sweep that generates an ROC curve in one linear scan - see Fawcett, "An introduction
to ROC analysis" (Pattern Recognition Letters, 2006), Algorithm 2, and Google's decision
forests documentation on the exact splitter for numerical features, which states the same
O(n log n) bound "because of the sorting of the feature values".

The result is not an approximation: it is the best threshold available, the same answer
linear search arrives at, and it is reached without ever scoring a candidate twice.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from thresher.utils import POSITIVE_LABEL, print_progress_bar


def run(
    scores: Sequence[float],
    actual_classes: Sequence[int],
    verbose: bool,
    progress_bar: bool,
    alg_options: Mapping[str, Any],
) -> float:
    """Find the threshold with the highest accuracy, exactly.

    Args:
        scores: the values being split.
        actual_classes: the matching ground-truth classes, as -1 and 1.
        verbose: print progress information.
        progress_bar: draw a progress bar on stdout.
        alg_options: accepted for signature compatibility with the other solvers. This
            algorithm has nothing to tune - it is exact, so there is no accuracy to trade
            against speed.

    Returns:
        A threshold yielding the highest achievable fraction of correctly classified
        samples. Where several thresholds tie, the lowest is returned. Interior results
        are the midpoint between the two scores they separate, matching linear search;
        a result equal to `max(scores)` means every sample is best classified negative.

    Raises:
        ValueError: if no scores were given.
    """
    if not scores:
        raise ValueError("At least one score is needed to evaluate a threshold.")

    # One sort, and the sweep below never looks back.
    paired = sorted(zip(scores, actual_classes, strict=False))
    total = len(paired)
    total_positive = sum(1 for _, actual in paired if actual == POSITIVE_LABEL)

    if verbose:
        print(f"Sweeping {total} sorted samples for the exact optimum.")

    negatives_behind = 0
    positives_behind = 0
    best_correct = -1
    best_threshold = float(paired[-1][0])

    for position in range(1, total + 1):
        score, actual = paired[position - 1]
        if actual == POSITIVE_LABEL:
            positives_behind += 1
        else:
            negatives_behind += 1

        if progress_bar:
            print_progress_bar(position, total)

        # A threshold can only sit *between* two different scores. Inside a run of equal
        # scores there is nowhere to put one - those samples are indivisible, and Fawcett
        # makes the same point about ties when generating an ROC curve.
        if position < total and paired[position][0] == score:
            continue

        # Everything up to here is predicted negative, everything after it positive.
        correct = negatives_behind + (total_positive - positives_behind)

        if correct > best_correct:
            best_correct = correct
            if position < total:
                best_threshold = (score + paired[position][0]) / 2
            else:
                # Past the largest score: nothing is predicted positive.
                best_threshold = float(score)

    if progress_bar:
        print_progress_bar(total, total)

    if verbose:
        print(f"Best threshold {best_threshold} classifies {best_correct}/{total} correctly.")

    return best_threshold

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

import math
from collections.abc import Mapping, Sequence
from typing import Any

from thresher.backends import Backend, LocalBackend
from thresher.exceptions import InsufficientDataError
from thresher.utils import print_progress_bar

#: This solver is exact, so it has no parameters: there is no accuracy to trade for
#: speed. Any `algorithm_params` key is therefore a mistake, and is reported as one -
#: see `dispatch.validate_algorithm_params`.
known_params: frozenset[str] = frozenset()


def run(
    scores: Sequence[float],
    actual_classes: Sequence[int],
    verbose: bool,
    progress_bar: bool,
    alg_options: Mapping[str, Any],
    backend: Backend | None = None,
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
        backend: where the counting happens. Defaults to in-process. Only the counting is
            distributed; the sweep over distinct scores is trivial by comparison and stays
            on the driver.

    Returns:
        A threshold yielding the highest achievable fraction of correctly classified
        samples - the best that exists, over every split a threshold can induce.

        Interior results are the midpoint between the two scores they separate, matching
        linear search. Two results sit at the edges: `max(scores)` classifies everything
        negative, and a value just below `min(scores)` classifies everything positive.
        The latter is the only result that can fall outside the span of the input, and it
        is returned only when it beats every threshold inside it, which needs data where
        score and class run contrary to each other.

    Raises:
        InsufficientDataError: if no scores were given. It is a `ValueError`.
    """
    # Length rather than truthiness: `not array` is ambiguous for a numpy array of more
    # than one element, and raises. Since 0.7.2 the input reaches here as the caller's own
    # container, so it need not be a list.
    if len(scores) == 0:
        raise InsufficientDataError("At least one score is needed to evaluate a threshold.")

    # The sweep needs only the class counts at each distinct score, never the samples
    # themselves - which is precisely what makes it distributable.
    counts = (backend or LocalBackend()).class_counts_by_score(scores, actual_classes)

    if verbose:
        print(f"Sweeping {len(counts)} distinct scores from {len(scores)} samples for the exact optimum.")

    best_threshold, best_correct = sweep_class_counts(counts, progress_bar=progress_bar)

    if verbose:
        print(f"Best threshold {best_threshold} classifies {best_correct}/{len(scores)} correctly.")

    return best_threshold


def sweep_class_counts(
    counts: Mapping[float, tuple[int, int]], progress_bar: bool = False
) -> tuple[float, int]:
    """Find the best threshold from class counts, without seeing the samples.

    The whole of the exact search lives here. It takes only "how many of each class sit at
    each distinct score", which is a summary bounded by the number of distinct scores
    rather than by the number of rows - so whatever produced those counts, in this process
    or across a cluster, the decision is made by this one function and the answers agree.

    Args:
        counts: distinct score mapped to its `(negatives, positives)`.
        progress_bar: draw a progress bar on stdout while sweeping.

    Returns:
        The best threshold and how many samples it classifies correctly.

    Raises:
        InsufficientDataError: if there are no counts to sweep.
    """
    if not counts:
        raise InsufficientDataError("At least one score is needed to evaluate a threshold.")

    ordered = sorted(counts)
    distinct = len(ordered)
    total_positive = sum(positives for _, positives in counts.values())

    negatives_behind = 0
    positives_behind = 0
    best_correct = -1
    best_threshold = float(ordered[-1])

    for index, score in enumerate(ordered):
        negatives_here, positives_here = counts[score]
        negatives_behind += negatives_here
        positives_behind += positives_here

        if progress_bar:
            print_progress_bar(index + 1, distinct)

        # Everything up to and including this score is predicted negative, everything
        # above it positive. Runs of equal scores are indivisible, which is automatic
        # here: they are one entry.
        correct = negatives_behind + (total_positive - positives_behind)

        if correct > best_correct:
            best_correct = correct
            # Below the largest score, sit between the two scores being separated; at the
            # largest score, sit on it, which predicts everything negative.
            best_threshold = (score + ordered[index + 1]) / 2 if index + 1 < distinct else float(score)

    if progress_bar:
        print_progress_bar(distinct, distinct)

    # The one split no threshold inside the data can express: everything classified
    # positive, which needs a threshold strictly below the smallest score. nextafter gives
    # the largest float that qualifies, so the answer stays as close to the data as the
    # representation allows. Considered last, and only taken on a strict improvement, so a
    # threshold outside the input range is never returned merely to break a tie.
    if total_positive > best_correct:
        best_correct = total_positive
        best_threshold = math.nextafter(ordered[0], -math.inf)

    return best_threshold, best_correct

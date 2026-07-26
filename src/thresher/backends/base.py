"""The execution-backend contract, and the pure map/reduce steps behind it.

A backend decides *where* the counting happens, never *what* the answer is. Every backend
must return bit-identical results for the same input; only the distribution of the work
changes. That is why the map and reduce steps live here as plain functions rather than
inside any one backend - they are shared verbatim, and can be tested without a cluster.

Two primitives cover every algorithm that can be parallelised:

`tally_candidates`
    Score a fixed list of candidate thresholds against the data. Linear search and grid
    search are both "score these candidates, keep the best", so both reduce to this.

`class_counts_by_score`
    Count the classes at each distinct score. The exact sweep needs only these counts, not
    the samples themselves, so the per-record work distributes and the driver is left with
    one pass over the distinct scores.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol

from thresher.exceptions import ShardMergeError

# Negative and positive counts observed at one score.
ClassCounts = tuple[int, int]


def tally_chunk(
    candidates: Sequence[float], scores: Sequence[float], actual_classes: Sequence[int]
) -> list[int]:
    """Count correct predictions per candidate, for one shard of the data.

    This is the map step of `tally_candidates`, and it is deliberately a free function:
    the local backend calls it directly and the Ray backend ships it to workers, so both
    run exactly the same code.

    Args:
        candidates: the thresholds to score.
        scores: this shard's scores.
        actual_classes: this shard's classes, as -1 and 1.

    Returns:
        One count per candidate, in the same order: how many of *this shard's* samples
        that candidate classifies correctly.
    """
    tallies = [0] * len(candidates)
    for index, candidate in enumerate(candidates):
        correct = 0
        for score, actual in zip(scores, actual_classes, strict=False):
            if (1 if score > candidate else -1) == actual:
                correct += 1
        tallies[index] = correct
    return tallies


def merge_tallies(partials: Iterable[Sequence[int]]) -> list[int]:
    """Add per-shard tallies together elementwise.

    This is the reduce step of `tally_candidates`. Addition is associative and
    commutative, so the order shards arrive in cannot affect the result - which is what
    lets the answer be identical across backends.

    Args:
        partials: one tally list per shard, all the same length.

    Returns:
        The summed tallies.

    Raises:
        ShardMergeError: if no partials were given, or they disagree on length. It is a
            `ValueError`.
    """
    merged: list[int] | None = None
    for partial in partials:
        if merged is None:
            merged = list(partial)
            continue
        if len(partial) != len(merged):
            raise ShardMergeError(f"shard tallies disagree on length: {len(partial)} vs {len(merged)}")
        for index, value in enumerate(partial):
            merged[index] += value

    if merged is None:
        raise ShardMergeError("no shard tallies to merge")
    return merged


def count_chunk(scores: Sequence[float], actual_classes: Sequence[int]) -> dict[float, ClassCounts]:
    """Count negatives and positives at each distinct score, for one shard.

    The map step of `class_counts_by_score`.

    Args:
        scores: this shard's scores.
        actual_classes: this shard's classes, as -1 and 1.

    Returns:
        A mapping of score to `(negatives, positives)` seen at it in this shard.
    """
    counts: dict[float, ClassCounts] = {}
    for score, actual in zip(scores, actual_classes, strict=False):
        negatives, positives = counts.get(score, (0, 0))
        if actual == 1:
            counts[score] = (negatives, positives + 1)
        else:
            counts[score] = (negatives + 1, positives)
    return counts


def merge_counts(partials: Iterable[Mapping[float, ClassCounts]]) -> dict[float, ClassCounts]:
    """Merge per-shard score counts by summing them.

    The reduce step of `class_counts_by_score`, and order-independent for the same reason
    `merge_tallies` is.

    Args:
        partials: one score-to-counts mapping per shard.

    Returns:
        The combined mapping.
    """
    merged: dict[float, ClassCounts] = {}
    for partial in partials:
        for score, (negatives, positives) in partial.items():
            running_negatives, running_positives = merged.get(score, (0, 0))
            merged[score] = (running_negatives + negatives, running_positives + positives)
    return merged


def plan_shards(total: int, workers: int, min_rows: int) -> list[tuple[int, int]]:
    """Work out the shard boundaries for a dataset.

    Kept separate from any backend so the arithmetic can be tested on its own, including
    on machines where Ray cannot be installed.

    Args:
        total: number of samples.
        workers: how many shards are wanted at most, normally the cluster's CPU count.
        min_rows: smallest worthwhile shard. Below this the coordination costs more than
            the work saved, so fewer, larger shards are produced instead.

    Returns:
        A list of `(start, stop)` index pairs covering `range(total)` exactly once, in
        order and without gaps. Empty when `total` is 0.
    """
    if total <= 0:
        return []

    usable = max(1, min(workers, total // max(1, min_rows)))
    size, remainder = divmod(total, usable)

    boundaries: list[tuple[int, int]] = []
    start = 0
    for index in range(usable):
        # Spread the remainder over the first shards, so sizes differ by at most one.
        stop = start + size + (1 if index < remainder else 0)
        if start < stop:
            boundaries.append((start, stop))
        start = stop
    return boundaries


class Backend(Protocol):
    """Where the counting happens.

    Implementations must not change the answer - see the module docstring.
    """

    name: str

    def tally_candidates(
        self,
        candidates: Sequence[float],
        scores: Sequence[float],
        actual_classes: Sequence[int],
    ) -> list[int]:
        """Count correct predictions for each candidate threshold, over all the data."""
        ...

    def class_counts_by_score(
        self, scores: Sequence[float], actual_classes: Sequence[int]
    ) -> dict[float, ClassCounts]:
        """Count negatives and positives at each distinct score, over all the data."""
        ...

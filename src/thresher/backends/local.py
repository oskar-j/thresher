"""The in-process backend, and the default.

Runs the map step once over the whole dataset. There is no reduce to do, because there is
only ever one shard.
"""

from collections.abc import Sequence

from thresher.backends.base import ClassCounts, count_chunk, tally_chunk


class LocalBackend:
    """Do the work here, in this process.

    This is what every version before 0.4.2 did, and what still happens unless a different
    backend is asked for.
    """

    name = "local"

    def tally_candidates(
        self,
        candidates: Sequence[float],
        scores: Sequence[float],
        actual_classes: Sequence[int],
    ) -> list[int]:
        """Count correct predictions for each candidate threshold.

        Args:
            candidates: the thresholds to score.
            scores: the values being split.
            actual_classes: the matching classes, as -1 and 1.

        Returns:
            One count per candidate, in the same order.
        """
        return tally_chunk(candidates, scores, actual_classes)

    def class_counts_by_score(
        self, scores: Sequence[float], actual_classes: Sequence[int]
    ) -> dict[float, ClassCounts]:
        """Count negatives and positives at each distinct score.

        Args:
            scores: the values being split.
            actual_classes: the matching classes, as -1 and 1.

        Returns:
            A mapping of score to `(negatives, positives)`.
        """
        return count_chunk(scores, actual_classes)

"""Small utilities shared by the solvers."""

from collections.abc import Iterable, Iterator


def granularity_of_scores(scores: Iterable[float], number_of_decimal_places: int = 2) -> Iterator[float]:
    """Round scores down to a coarser granularity.

    Reduces a set of scores to the distinct candidate thresholds worth evaluating.
    Nothing in the package calls this today - grid search builds its candidates with
    `numpy.linspace` instead.

    Args:
        scores: the values to round.
        number_of_decimal_places: how many decimal places to keep.

    Yields:
        Each score rounded to `number_of_decimal_places`. Duplicates are not removed.
    """
    for score in scores:
        yield round(score, number_of_decimal_places)

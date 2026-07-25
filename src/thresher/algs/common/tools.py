from collections.abc import Iterable, Iterator


def granularity_of_scores(scores: Iterable[float], number_of_decimal_places: int = 2) -> Iterator[float]:
    for score in scores:
        yield round(score, number_of_decimal_places)

from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import tee
from typing import Any, TypeVar

from thresher.exceptions import EMPTY_INPUT, SINGLE_CLASS_LABELS, UNEXPECTED_LABELS

NEGATIVE_LABEL = -1
POSITIVE_LABEL = 1

T = TypeVar("T")


def validate_actual_classes(actual_classes: Sequence[int]) -> None:
    """Check that the labels are usable before any algorithm runs.

    This was previously a bare `assert set(actual_classes) == {-1, 1}`, which said nothing
    about what was wrong - and, being an assertion, vanished entirely under `python -O`,
    letting malformed input reach the solvers instead.
    """
    present = set(actual_classes)

    if not present:
        raise ValueError(EMPTY_INPUT)

    unexpected = present - {NEGATIVE_LABEL, POSITIVE_LABEL}
    if unexpected:
        raise ValueError(
            UNEXPECTED_LABELS.format(unexpected=", ".join(repr(_) for _ in sorted(unexpected, key=repr)))
        )

    if present != {NEGATIVE_LABEL, POSITIVE_LABEL}:
        raise ValueError(SINGLE_CLASS_LABELS.format(only=repr(next(iter(present)))))


def map_labels(labels: Iterable[Any], mapping: Iterable[Any]) -> Iterator[int]:
    # Declared as Iterable because the caller only knows that much about the option, but
    # a mapping has to be indexable in a defined order - hence the check below, which
    # also narrows the type for the indexing that follows.
    if not isinstance(mapping, list | tuple):
        raise TypeError(f'The "labels" option must be a list or a tuple, got {type(mapping).__name__}.')
    for label in labels:
        if label == mapping[0]:
            yield NEGATIVE_LABEL
        elif label == mapping[1]:
            yield POSITIVE_LABEL
        else:
            raise TypeError("Value not found in the mapping - map_labels() cannot map label classes.")


def get_or_default(options: Mapping[str, Any], key: str, default: T) -> T:
    if key in options:
        value: T = options[key]
        return value
    return default


def pairwise(iterable: Iterable[T]) -> Iterator[tuple[T, T]]:
    a, b = tee(iterable)
    next(b, None)
    # 'b' is deliberately one shorter than 'a' here, so this pairing is never strict.
    return zip(a, b, strict=False)


# Print iterations progress
def print_progress_bar(
    iteration: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    length: int = 100,
    fill: str = "#",
) -> None:
    """Call in a loop to create a terminal progress bar.

    Args:
        iteration: current iteration.
        total: total number of iterations.
        prefix: string printed before the bar.
        suffix: string printed after the bar.
        decimals: number of decimals in the percentage.
        length: character length of the bar.
        fill: bar fill character.
    """
    percent = f"{100 * (iteration / float(total)):.{decimals}f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="\r")
    # Print New Line on Complete
    if iteration == total:
        print()

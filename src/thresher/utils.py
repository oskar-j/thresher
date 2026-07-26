"""Helpers shared across the package: label handling, option lookup and output."""

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import tee
from typing import Any, TypeVar

from thresher.exceptions import (
    LABEL_MAPPING_TYPE,
    LABEL_NOT_IN_MAPPING,
    EmptyInputError,
    LabelMappingError,
    LengthMismatchError,
    MissingLabelsError,
    SingleClassError,
    UnexpectedLabelsError,
)

NEGATIVE_LABEL = -1
POSITIVE_LABEL = 1

T = TypeVar("T")


def validate_lengths(scores: Sequence[float], actual_classes: Sequence[int]) -> None:
    """Check that every score has a class to go with it.

    The solvers pair the two with `zip`, which stops at the shorter sequence, so a
    mismatch used to be absorbed in silence: six scores against four classes simply
    discarded two scores and returned a threshold computed from the rest. That is a wrong
    answer rather than a partial one, and nothing in the result hints at it.

    Args:
        scores: the values being split.
        actual_classes: the matching ground-truth classes.

    Returns:
        None. This is a guard - it either passes silently or raises.

    Raises:
        LengthMismatchError: if the two differ in length. It is a `ValueError`.
    """
    if len(scores) != len(actual_classes):
        raise LengthMismatchError(len(scores), len(actual_classes))


def validate_actual_classes(actual_classes: Sequence[int]) -> None:
    """Check that the labels are usable before any algorithm runs.

    This was previously a bare `assert set(actual_classes) == {-1, 1}`, which said nothing
    about what was wrong - and, being an assertion, vanished entirely under `python -O`,
    letting malformed input reach the solvers instead.

    Args:
        actual_classes: the ground-truth classes, already normalized to -1 and 1.

    Returns:
        None. This is a guard - it either passes silently or raises.

    Raises:
        EmptyInputError, MissingLabelsError, UnexpectedLabelsError, SingleClassError:
            for each of those cases in turn. All are `InvalidInputError`, and so also
            `ValueError`.
    """
    present = set(actual_classes)

    if not present:
        raise EmptyInputError

    # A blank cell in a CSV arrives as NaN. Reporting it as an unrecognised label sends
    # people to the 'labels' option, which cannot help - the value is absent, not
    # differently encoded.
    missing = sum(1 for value in actual_classes if isinstance(value, float) and math.isnan(value))
    if missing:
        raise MissingLabelsError(missing)

    unexpected = present - {NEGATIVE_LABEL, POSITIVE_LABEL}
    if unexpected:
        raise UnexpectedLabelsError(sorted(unexpected, key=repr))

    if present != {NEGATIVE_LABEL, POSITIVE_LABEL}:
        raise SingleClassError(next(iter(present)))


def map_labels(labels: Iterable[Any], mapping: Iterable[Any]) -> Iterator[int]:
    """Translate caller-supplied labels into the internal -1 / 1 pair.

    Args:
        labels: the ground-truth classes as the caller provided them.
        mapping: a two-item list or tuple, negative label first, positive second -
            the value of the `labels` constructor option, e.g. `(0, 1)`.

    Yields:
        -1 for each label matching `mapping[0]`, 1 for each matching `mapping[1]`.

    Raises:
        LabelMappingError: if `mapping` is not a list or tuple, or if a label appears
            that is in neither position of the mapping. It is a `TypeError`.
    """
    # Declared as Iterable because the caller only knows that much about the option, but
    # a mapping has to be indexable in a defined order - hence the check below, which
    # also narrows the type for the indexing that follows.
    if not isinstance(mapping, list | tuple):
        raise LabelMappingError(LABEL_MAPPING_TYPE.format(got=type(mapping).__name__))
    for label in labels:
        if label == mapping[0]:
            yield NEGATIVE_LABEL
        elif label == mapping[1]:
            yield POSITIVE_LABEL
        else:
            raise LabelMappingError(LABEL_NOT_IN_MAPPING)


def get_or_default(options: Mapping[str, Any], key: str, default: T) -> T:
    """Read one algorithm parameter, falling back to its default.

    Note that unknown keys are never reported: a mistyped parameter name silently leaves
    the default in place rather than raising.

    Args:
        options: the user-supplied `algorithm_params` mapping.
        key: the parameter name to read.
        default: the value to use when the key is absent.

    Returns:
        The value stored under `key`, or `default` if there is none.
    """
    if key in options:
        value: T = options[key]
        return value
    return default


def pairwise(iterable: Iterable[T]) -> Iterator[tuple[T, T]]:
    """Yield consecutive overlapping pairs from an iterable.

    `pairwise([1, 2, 3])` yields `(1, 2)` then `(2, 3)`.

    Args:
        iterable: the values to pair up.

    Returns:
        An iterator of adjacent pairs. It is empty for inputs shorter than two items,
        which is why callers have to handle "no candidate found".
    """
    a, b = tee(iterable)
    next(b, None)
    # 'b' is deliberately one shorter than 'a' here, so this pairing is never strict.
    return zip(a, b, strict=False)


def print_progress_bar(
    iteration: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    length: int = 100,
    fill: str = "#",
) -> None:
    """Call in a loop to draw a terminal progress bar in place.

    Args:
        iteration: current iteration.
        total: total number of iterations. A final newline is printed once the two match.
        prefix: string printed before the bar.
        suffix: string printed after the bar.
        decimals: number of decimals in the percentage.
        length: character length of the bar.
        fill: bar fill character.

    Returns:
        None. The bar is written to stdout.
    """
    percent = f"{100 * (iteration / float(total)):.{decimals}f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="\r")
    # Print New Line on Complete
    if iteration == total:
        print()

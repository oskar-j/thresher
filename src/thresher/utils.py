"""Helpers shared across the package: label handling, option lookup and output."""

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import tee
from typing import Any, TypeVar, cast

from thresher.exceptions import (
    LABEL_MAPPING_LENGTH,
    LABEL_MAPPING_TYPE,
    LABEL_NOT_IN_MAPPING,
    EmptyInputError,
    LabelMappingError,
    LengthMismatchError,
    MissingLabelsError,
    SingleClassError,
    UndefinedScoresError,
    UnexpectedLabelsError,
)

NEGATIVE_LABEL = -1
POSITIVE_LABEL = 1

T = TypeVar("T")


def as_sequence(values: Iterable[T]) -> Sequence[T]:
    """Give the solvers something they can measure, iterate twice and index - cheaply.

    The solvers need three things from their input: `len()`, more than one pass over it,
    and integer indexing. A `list` is the obvious way to guarantee all three, and building
    one was what this used to do for anything that was not already a `Sequence`. That
    quietly included the two types this library is built around: neither `numpy.ndarray`
    nor `pandas.Series` is registered as a `Sequence` - they have no `index` or `count` -
    so both were copied on the way in, at `O(n)`, which is the allocation 0.5.3 removed
    from `optimize_threshold` in the first place. `hist` holds a few kilobytes of counters
    however large the data is, and was still paying 12 MB to receive 200,000 rows.

    Args:
        values: the scores or the classes, however the caller holds them.

    Returns:
        The caller's own container where it already does what the solvers need, and a list
        built from it where it does not - a generator, a set, a dict view. A `pandas`
        object is handed over as its underlying array, which is a view rather than a copy
        for the numeric dtypes a score column has.
    """
    if isinstance(values, Sequence):
        return values

    # pandas indexes by *label*: `series[0]` looks up the label 0 rather than the first
    # element, so on a Series that has been filtered - the ordinary case, and one where
    # the labels keep the gaps - it either raises KeyError or returns the wrong row. That
    # matters because `stochastic_process` and `grid` sample positions. `to_numpy()` hands
    # back the buffer underneath, which indexes by position and shares its memory.
    converter = getattr(values, "to_numpy", None)
    candidate: Any = converter() if callable(converter) else values

    # A Mapping is deliberately excluded: it has both dunders, but iterating one yields
    # keys while indexing it yields values, and the solvers do both. It becomes a list of
    # its keys, exactly as before.
    if (
        hasattr(candidate, "__len__")
        and hasattr(candidate, "__getitem__")
        and not isinstance(candidate, Mapping)
    ):
        # An ndarray satisfies the three requirements above without satisfying `Sequence`,
        # which is a wider protocol than anything here uses.
        return cast("Sequence[T]", candidate)

    return list(values)


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


def validate_scores(scores: Sequence[float]) -> None:
    """Check that every score is a number a threshold can be placed against.

    NaN is the case that matters. Every comparison against it is false, so `score > t` is
    false for any threshold, and a NaN that reaches a solver is not merely ignored - it
    propagates. `exact` sorted it into place and handed the NaN back as the answer, a
    "threshold" that classifies the whole dataset negative; `hist` failed instead, but
    with a bare `ValueError` out of its bin arithmetic. One NaN in a `predict_proba`
    column is an ordinary upstream accident, so it is worth one pass to catch here.

    `None` is treated the same way: it is the shape a blank takes in a plain Python list,
    where pandas would have produced NaN, and it used to reach the sort and fail there
    with a bare `TypeError`. The Spark interface refuses a null score for the same reason.

    Infinities are left alone. They order correctly against everything else, so a
    threshold can be placed relative to them, and `exact` handles them already.

    Args:
        scores: the values being split.

    Returns:
        None. This is a guard - it either passes silently or raises.

    Raises:
        UndefinedScoresError: if any score is NaN. It is a `ValueError`.
    """
    # `value != value` is true only for NaN, and unlike `math.isnan` it needs no coercion
    # and holds for numpy's float types as well as Python's. None is counted with it: a
    # blank cell arrives as one from a bare Python list where pandas would have given NaN,
    # and it used to reach the sort and fail there as a bare TypeError.
    undefined = sum(1 for value in scores if value is None or value != value)
    if undefined:
        raise UndefinedScoresError(undefined)


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


def validate_label_mapping(mapping: Any) -> list[Any] | tuple[Any, ...]:
    """Check that a `labels` option is a usable mapping before anything relies on it.

    Shared between `Thresher.__init__`, which validates the option the moment it is
    given, and `map_labels`, which is what a caller mutating the live options dict
    afterwards still runs into.

    Args:
        mapping: the value of the `labels` constructor option.

    Returns:
        The mapping itself, now known to be an indexable two-item pair.

    Raises:
        LabelMappingError: if `mapping` is not a list or tuple, or does not hold exactly
            two values. It is a `TypeError`.
    """
    # A mapping has to be indexable in a defined order - a set, for example, has no
    # "first" - and needs one value per class, no more and no fewer. A one-item mapping
    # previously fell through to a bare IndexError inside map_labels.
    if not isinstance(mapping, list | tuple):
        raise LabelMappingError(LABEL_MAPPING_TYPE.format(got=type(mapping).__name__))
    if len(mapping) != 2:
        raise LabelMappingError(LABEL_MAPPING_LENGTH.format(count=len(mapping)))
    return mapping


def map_labels(labels: Iterable[Any], mapping: Iterable[Any]) -> Iterator[int]:
    """Translate caller-supplied labels into the internal -1 / 1 pair.

    Args:
        labels: the ground-truth classes as the caller provided them.
        mapping: a two-item list or tuple, negative label first, positive second -
            the value of the `labels` constructor option, e.g. `(0, 1)`.

    Yields:
        -1 for each label matching `mapping[0]`, 1 for each matching `mapping[1]`.

    Raises:
        LabelMappingError: if `mapping` is not a two-item list or tuple, or if a label
            appears that is in neither position of the mapping. It is a `TypeError`.
    """
    pair = validate_label_mapping(mapping)
    for label in labels:
        if label == pair[0]:
            yield NEGATIVE_LABEL
        elif label == pair[1]:
            yield POSITIVE_LABEL
        else:
            raise LabelMappingError(LABEL_NOT_IN_MAPPING)


def get_or_default(options: Mapping[str, Any], key: str, default: T) -> T:
    """Read one algorithm parameter, falling back to its default.

    This does not report an unknown key - it cannot tell one from a key meant for a
    different solver. `dispatch.validate_algorithm_params` is what rejects those, when
    the `Thresher` is built and the algorithm is known.

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
    """Draw one frame of the built-in progress bar. Moved to `thresher.progress`.

    Kept here because it has been importable from this module since the first release.
    Everything inside the package now goes through `thresher.progress.make_progress`,
    which picks between this bar and tqdm.

    Args:
        iteration: current iteration.
        total: total number of iterations. The line is ended once the two match.
        prefix: string printed before the bar.
        suffix: string printed after the bar.
        decimals: number of decimals in the percentage.
        length: character length of the bar.
        fill: bar fill character.

    Returns:
        None. The bar is written to stderr - it was stdout until 0.8.0, which is the
        stream the command line prints its answer on.
    """
    # Imported here rather than at module scope. This module is pulled in while the
    # package itself is still initialising, by way of `interface`, and a module-level
    # import would drag `progress` - and through it `log` and loguru - into that chain for
    # the sake of a function nothing inside the package calls any more.
    from thresher.progress import print_progress_bar as draw

    draw(iteration, total, prefix, suffix, decimals, length, fill)

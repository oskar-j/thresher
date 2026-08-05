"""The exceptions this package raises, and the wording they carry.

Everything raised from `thresher` derives from `ThresherError`, so a caller can catch this
package's failures without also catching unrelated ones:

    try:
        Thresher().optimize_threshold(scores, actual_classes)
    except thresher.exceptions.InvalidInputError as exc:
        ...

Each class **also** inherits the builtin it used to be raised as - `ValueError`,
`TypeError`, `AttributeError`, `ImportError`, `NotImplementedError`. That is deliberate
rather than decorative: code written against any earlier version catches those builtins,
and this package's own command line catches `ValueError` and `ImportError`. Dual
inheritance makes the hierarchy an addition rather than a breaking change.

Where an error carries useful detail - how many scores, which labels, what was available -
it is kept on the instance as well as formatted into the message, so callers can act on it
instead of parsing prose.

The message templates stay as module constants. They define the wording, tests assert
against the same strings the user sees, and they were importable before the classes
existed.
"""

from collections.abc import Iterable
from typing import Any

NOT_ITERABLE = 'The "{attribute}" attribute is not an Iterable! Please provide a list-like object'
NOT_IMPLEMENTED_ERROR = NOT_ITERABLE.format(attribute="scores")
UNKNOWN_ALGORITHM = "Unknown algorithm. Run get_supported_algorithms() to get a list of available algorithms."

UNKNOWN_OPTIONS = (
    "Unknown option(s) passed to Thresher: {unknown}. Valid options are: {valid}. "
    "A mistyped name used to be accepted in silence, leaving the default in place."
)

ALGORITHM_PARAMS_TYPE = 'The "algorithm_params" option must be a mapping of name to value, got {got}.'

UNKNOWN_PARAMS = (
    "Unknown algorithm_params key(s) for {algorithm}: {unknown}. It reads: {accepted}. "
    "A mistyped name used to be ignored in silence, so the run continued with the "
    "default the caller believed they had changed."
)

UNKNOWN_ALGORITHM_NAME = (
    "Unknown algorithm {name!r}. Available algorithms are: {available}. "
    "Run get_supported_algorithms() to list them at runtime."
)

UNKNOWN_BACKEND_NAME = "Unknown backend {name!r}. Available backends are: {available}."

PARALLEL_BOOTSTRAP_FAILED = (
    "The worker processes could not start, so no counting was done. This almost always "
    "means the calling script builds its Thresher at module level: on start methods that "
    "re-import __main__ - spawn on macOS and Windows - each worker re-runs the script and "
    "starts its own workers. Put the call inside a function and guard it:\n\n"
    '    if __name__ == "__main__":\n'
    "        main()\n\n"
    "See examples/sample_parallel.py. Nothing to change if you would rather not: "
    "backend='local' is the default and needs no guard."
)

UNEXPECTED_LABELS = (
    'Found {unexpected} in "actual_classes", but only -1 and 1 are supported. '
    'If your data uses different labels, declare them with the "labels" option, '
    "for example Thresher(labels=(0, 1))."
)

SINGLE_CLASS_LABELS = (
    '"actual_classes" contains only {only}. Both -1 and 1 must be present - '
    "a threshold cannot be optimized against a single class."
)

LENGTH_MISMATCH = (
    'Got {scores} scores but {classes} entries in "actual_classes". Each score needs the '
    "class it belongs to, so the two must be the same length."
)

MISSING_LABELS = (
    '"actual_classes" contains {count} missing value(s). Every score needs a known class; '
    "rows with a blank or NaN label have to be filled in or dropped before optimizing."
)

EMPTY_INPUT = '"scores" and "actual_classes" are empty - there is nothing to optimize.'

UNDEFINED_SCORES = (
    '"scores" contains {count} value(s) that are not a number. A threshold cannot be '
    "placed relative to NaN - every comparison against it is false - so rows with a "
    "missing or undefined score have to be filled in or dropped before optimizing."
)

LABEL_MAPPING_TYPE = 'The "labels" option must be a list or a tuple, got {got}.'
LABEL_MAPPING_LENGTH = 'The "labels" option needs exactly two values, negative label first, got {count}.'
LABEL_NOT_IN_MAPPING = "Value not found in the mapping - map_labels() cannot map label classes."


class ThresherError(Exception):
    """Base class for every error raised by this package.

    Catch this to handle anything thresher rejects, without also catching failures from
    numpy, pandas or your own code that happen to use the same builtin types.
    """


class ConfigurationError(ThresherError, ValueError):
    """Something was asked for that does not exist - a mistyped name, usually.

    Raised while an object is being built, before any data is touched.
    """


class UnknownAlgorithmError(ConfigurationError):
    """No algorithm goes by that name, or any of its synonyms."""

    def __init__(self, name: Any, available: Iterable[str]) -> None:
        """Record what was asked for and what would have worked.

        Args:
            name: the name that matched nothing. Usually a mistyped string, but anything
                arrives here - a non-string is as unknown as a wrong spelling.
            available: the algorithm ids that would have.
        """
        self.name = name
        self.available = sorted(available)
        super().__init__(UNKNOWN_ALGORITHM_NAME.format(name=name, available=", ".join(self.available)))


class UnknownBackendError(ConfigurationError):
    """No execution backend goes by that name."""

    def __init__(self, name: Any, available: Iterable[str]) -> None:
        """Record what was asked for and what would have worked.

        Args:
            name: the name that matched nothing.
            available: the backend names that would have.
        """
        self.name = name
        self.available = list(available)
        super().__init__(UNKNOWN_BACKEND_NAME.format(name=name, available=", ".join(self.available)))


class InvalidInputError(ThresherError, ValueError):
    """The data cannot be optimized over as given."""


class EmptyInputError(InvalidInputError):
    """There is nothing to optimize."""

    def __init__(self) -> None:
        super().__init__(EMPTY_INPUT)


class LengthMismatchError(InvalidInputError):
    """The scores and the classes do not line up one to one."""

    def __init__(self, score_count: int, class_count: int) -> None:
        """Record both counts, so a caller can report or repair the difference.

        Args:
            score_count: how many scores were given.
            class_count: how many classes were given.
        """
        self.score_count = score_count
        self.class_count = class_count
        super().__init__(LENGTH_MISMATCH.format(scores=score_count, classes=class_count))


class MissingLabelsError(InvalidInputError):
    """Some scores have no class at all - a blank cell arrives as NaN."""

    def __init__(self, count: int) -> None:
        """Record how many are missing.

        Args:
            count: number of missing values found.
        """
        self.count = count
        super().__init__(MISSING_LABELS.format(count=count))


class UndefinedScoresError(InvalidInputError):
    """Some scores are NaN, so no threshold can be placed relative to them.

    Distinct from `MissingLabelsError`, which is the same problem on the other column.
    Before 0.7.1 this went unchecked and each algorithm failed its own way: `exact`
    returned NaN as though it were an answer - a threshold that classifies everything
    negative, since every comparison against NaN is false - while `hist` raised a bare
    `ValueError` from its bin arithmetic.
    """

    def __init__(self, count: int) -> None:
        """Record how many are undefined.

        Args:
            count: number of NaN scores found.
        """
        self.count = count
        super().__init__(UNDEFINED_SCORES.format(count=count))


class UnexpectedLabelsError(InvalidInputError):
    """Labels outside the -1 / 1 pair the solvers work in."""

    def __init__(self, unexpected: Iterable[Any]) -> None:
        """Record the offending values.

        Args:
            unexpected: the label values that are neither -1 nor 1.
        """
        self.unexpected = list(unexpected)
        formatted = ", ".join(repr(value) for value in self.unexpected)
        super().__init__(UNEXPECTED_LABELS.format(unexpected=formatted))


class SingleClassError(InvalidInputError):
    """Only one of the two classes is present, so there is nothing to separate."""

    def __init__(self, only: Any) -> None:
        """Record the class that was found.

        Args:
            only: the single label value present.
        """
        self.only = only
        super().__init__(SINGLE_CLASS_LABELS.format(only=repr(only)))


class InsufficientDataError(InvalidInputError):
    """Too little data for this algorithm to produce a candidate threshold."""


class LabelMappingError(ThresherError, TypeError):
    """The `labels` option cannot map the classes it was given."""


class NotIterableError(ThresherError, AttributeError):
    """`scores` or `actual_classes` is not something that can be iterated.

    Inherits `AttributeError` because that is what earlier versions raised. `TypeError`
    would fit the failure better, but changing it would break existing `except` clauses
    for no practical gain.
    """

    def __init__(self, attribute: str = "scores") -> None:
        """Record which argument was not iterable, and name it in the message.

        Args:
            attribute: the offending argument - `"scores"` or `"actual_classes"`. The
                default keeps the historical wording, which only ever blamed the former.
        """
        self.attribute = attribute
        super().__init__(NOT_ITERABLE.format(attribute=attribute))


class BackendDependencyError(ThresherError, ImportError):
    """A backend was selected whose optional dependency is not installed."""


class ParallelBootstrapError(ThresherError, RuntimeError):
    """Worker processes could not be started, so the work never ran.

    Inherits `RuntimeError` because that is what `BrokenProcessPool` - the failure this
    replaces - already was, so an `except RuntimeError` written around a parallel run keeps
    working. Before 0.7.0 this situation had no exception at all: `multiprocessing.Pool`
    waited on workers that would never report, and the process simply hung.
    """


class AlgorithmNotWiredError(ThresherError, NotImplementedError):
    """An algorithm is in the registry but has no branch in the dispatcher.

    A mistake in the package rather than in the caller's code: it means
    `available_algorithms` and `run_computations` have drifted apart.
    """

    def __init__(self, message: str = UNKNOWN_ALGORITHM) -> None:
        super().__init__(message)


class ShardMergeError(ThresherError, ValueError):
    """Partial results from a distributed run could not be combined.

    Also a package-level mistake rather than a caller's: the shards disagree about how
    many candidates were scored, which cannot happen within a single run.
    """

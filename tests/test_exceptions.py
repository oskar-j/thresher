"""The exception hierarchy.

Two properties matter here and are tested separately, because they pull in opposite
directions:

* everything this package raises is a `ThresherError`, so it can be caught precisely;
* everything is *also* the builtin it was raised as before 0.4.5, so code written against
  an earlier version still works.

Lose the second and the hierarchy stops being an addition and becomes a breaking change.
"""

import math
from collections.abc import Callable

import pytest

import thresher
from thresher import exceptions as exc
from thresher.algs.exact import compute as exact_compute
from thresher.backends.base import merge_tallies

Dataset = tuple[list[float], list[int]]

# Every way a caller can be told they got something wrong, with the builtin each one has
# to remain compatible with.
FAILURES: list[tuple[str, Callable[[], object], type[Exception], type[Exception]]] = [
    (
        "single class",
        lambda: thresher.Thresher().optimize_threshold([0.1, 0.2], [-1, -1]),
        exc.SingleClassError,
        ValueError,
    ),
    (
        "length mismatch",
        lambda: thresher.Thresher().optimize_threshold([0.1, 0.2, 0.3], [-1, 1]),
        exc.LengthMismatchError,
        ValueError,
    ),
    (
        "undefined scores",
        lambda: thresher.Thresher().optimize_threshold([0.1, math.nan, 0.6, 0.9], [-1, -1, 1, 1]),
        exc.UndefinedScoresError,
        ValueError,
    ),
    (
        "missing labels",
        lambda: thresher.Thresher().optimize_threshold([0.1, 0.9], [math.nan, 1]),
        exc.MissingLabelsError,
        ValueError,
    ),
    (
        "unmapped labels",
        lambda: thresher.Thresher().optimize_threshold([0.1, 0.9], [0, 1]),
        exc.UnexpectedLabelsError,
        ValueError,
    ),
    (
        "empty input",
        lambda: thresher.Thresher().optimize_threshold([], []),
        exc.EmptyInputError,
        ValueError,
    ),
    (
        "unknown algorithm",
        lambda: thresher.Thresher(algorithm="does-not-exist"),
        exc.UnknownAlgorithmError,
        ValueError,
    ),
    (
        "unknown backend",
        lambda: thresher.Thresher(backend="does-not-exist"),
        exc.UnknownBackendError,
        ValueError,
    ),
    (
        "mistyped option name",
        lambda: thresher.Thresher(algoritm="exact"),
        exc.ConfigurationError,
        ValueError,
    ),
    (
        "non-string algorithm",
        lambda: thresher.Thresher(algorithm=123),
        exc.UnknownAlgorithmError,
        ValueError,
    ),
    (
        "mistyped algorithm parameter",
        lambda: thresher.Thresher(algorithm="sgd", algorithm_params={"stoch_ration": 0.5}),
        exc.ConfigurationError,
        ValueError,
    ),
    (
        "meaningless worker count",
        lambda: thresher.Thresher(algorithm="ls", algorithm_params={"n_jobs": 0}).optimize_threshold(
            [0.1, 0.9], [-1, 1]
        ),
        exc.ConfigurationError,
        ValueError,
    ),
    (
        "not iterable",
        lambda: thresher.Thresher().optimize_threshold(7, [-1, 1]),  # type: ignore[arg-type]
        exc.NotIterableError,
        AttributeError,
    ),
    (
        "unusable labels option",
        lambda: thresher.Thresher(labels={0, 1}).optimize_threshold([0.1, 0.9], [0, 1]),
        exc.LabelMappingError,
        TypeError,
    ),
    (
        "too little data",
        lambda: exact_compute.run([], [], progress_bar=False, alg_options={}),
        exc.InsufficientDataError,
        ValueError,
    ),
    (
        "inconsistent shards",
        lambda: merge_tallies([[1, 2], [1, 2, 3]]),
        exc.ShardMergeError,
        ValueError,
    ),
]

IDS = [case[0] for case in FAILURES]


@pytest.mark.parametrize(("_name", "trigger", "specific", "_builtin"), FAILURES, ids=IDS)
def test_everything_is_a_thresher_error(
    _name: str, trigger: Callable[[], object], specific: type[Exception], _builtin: type[Exception]
) -> None:
    """One `except` clause is enough to catch anything this package rejects."""
    with pytest.raises(exc.ThresherError):
        trigger()


@pytest.mark.parametrize(("_name", "trigger", "specific", "builtin"), FAILURES, ids=IDS)
def test_the_original_builtin_still_catches_it(
    _name: str, trigger: Callable[[], object], specific: type[Exception], builtin: type[Exception]
) -> None:
    """Code written before 0.4.5 catches builtins, and must keep working.

    This package's own command line is such code: it catches `ValueError` and
    `ImportError` around `optimize_threshold`.
    """
    with pytest.raises(builtin):
        trigger()


@pytest.mark.parametrize(("_name", "trigger", "specific", "_builtin"), FAILURES, ids=IDS)
def test_each_failure_has_its_own_type(
    _name: str, trigger: Callable[[], object], specific: type[Exception], _builtin: type[Exception]
) -> None:
    """So a caller can distinguish them without matching on message text."""
    with pytest.raises(specific):
        trigger()


def test_ray_missing_is_an_import_error() -> None:
    """Kept out of the table because it only fails where Ray is absent."""
    try:
        import ray  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("Ray is installed, so this path cannot be reached")

    with pytest.raises(exc.BackendDependencyError) as excinfo:
        thresher.Thresher(backend="ray")

    assert isinstance(excinfo.value, ImportError)
    assert isinstance(excinfo.value, exc.ThresherError)


class TestCarriedDetail:
    """An exception should carry its detail, not just describe it in prose."""

    def test_length_mismatch_carries_both_counts(self) -> None:
        with pytest.raises(exc.LengthMismatchError) as excinfo:
            thresher.Thresher().optimize_threshold([0.1, 0.2, 0.3], [-1, 1])

        assert excinfo.value.score_count == 3
        assert excinfo.value.class_count == 2

    def test_missing_labels_carries_the_count(self) -> None:
        with pytest.raises(exc.MissingLabelsError) as excinfo:
            thresher.Thresher().optimize_threshold([0.1, 0.5, 0.9], [math.nan, math.nan, 1])

        assert excinfo.value.count == 2

    def test_unknown_algorithm_carries_the_name_and_the_alternatives(self) -> None:
        with pytest.raises(exc.UnknownAlgorithmError) as excinfo:
            thresher.Thresher(algorithm="does-not-exist")

        assert excinfo.value.name == "does-not-exist"
        assert "exact" in excinfo.value.available

    def test_unknown_backend_carries_the_name_and_the_alternatives(self) -> None:
        with pytest.raises(exc.UnknownBackendError) as excinfo:
            thresher.Thresher(backend="does-not-exist")

        assert excinfo.value.name == "does-not-exist"
        assert "local" in excinfo.value.available

    def test_unexpected_labels_carries_the_offending_values(self) -> None:
        with pytest.raises(exc.UnexpectedLabelsError) as excinfo:
            thresher.Thresher().optimize_threshold([0.1, 0.9], [0, 1])

        assert 0 in excinfo.value.unexpected

    def test_single_class_carries_the_class_found(self) -> None:
        with pytest.raises(exc.SingleClassError) as excinfo:
            thresher.Thresher().optimize_threshold([0.1, 0.2], [-1, -1])

        assert excinfo.value.only == -1


class TestGrouping:
    """The intermediate classes exist so related failures can be caught together."""

    @pytest.mark.parametrize(
        "trigger",
        [
            lambda: thresher.Thresher().optimize_threshold([0.1, 0.2], [-1, -1]),
            lambda: thresher.Thresher().optimize_threshold([0.1, 0.2, 0.3], [-1, 1]),
            lambda: thresher.Thresher().optimize_threshold([], []),
        ],
    )
    def test_data_problems_share_a_base(self, trigger: Callable[[], object]) -> None:
        with pytest.raises(exc.InvalidInputError):
            trigger()

    @pytest.mark.parametrize(
        "trigger",
        [
            lambda: thresher.Thresher(algorithm="does-not-exist"),
            lambda: thresher.Thresher(backend="does-not-exist"),
        ],
    )
    def test_mistyped_names_share_a_base(self, trigger: Callable[[], object]) -> None:
        with pytest.raises(exc.ConfigurationError):
            trigger()

    def test_a_data_problem_is_not_a_configuration_problem(self) -> None:
        # The two bases are siblings, so catching one must not catch the other.
        with pytest.raises(exc.InvalidInputError):
            thresher.Thresher().optimize_threshold([0.1, 0.2], [-1, -1])

        assert not issubclass(exc.InvalidInputError, exc.ConfigurationError)
        assert not issubclass(exc.ConfigurationError, exc.InvalidInputError)


class TestParallelBootstrap:
    """The failure that replaced a hang, added in 0.7.0.

    It cannot be triggered from inside pytest - the situation needs a script whose
    module-level code starts a pool - so the subprocess test lives in test_backends.py.
    What is checked here is the shape of the class, which that test cannot assert on.
    """

    def test_it_is_a_thresher_error_and_a_runtime_error(self) -> None:
        error = exc.ParallelBootstrapError(exc.PARALLEL_BOOTSTRAP_FAILED)

        assert isinstance(error, exc.ThresherError)
        # BrokenProcessPool, which this replaces, is already a RuntimeError.
        assert isinstance(error, RuntimeError)

    def test_the_message_says_how_to_fix_it(self) -> None:
        assert '__name__ == "__main__"' in exc.PARALLEL_BOOTSTRAP_FAILED
        assert "backend='local'" in exc.PARALLEL_BOOTSTRAP_FAILED


def test_the_message_constants_are_still_importable() -> None:
    """They predate the classes and define the wording; the classes format them."""
    assert exc.LENGTH_MISMATCH
    assert exc.NOT_IMPLEMENTED_ERROR
    assert exc.UNKNOWN_ALGORITHM

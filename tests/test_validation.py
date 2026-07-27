"""Input validation and error reporting, fixed in 0.2.3.

Bad input previously surfaced as a bare StopIteration or a message-less AssertionError,
neither of which told the caller what was wrong.
"""

import subprocess
import sys

import pytest

import thresher
from thresher import algorithm


def test_unknown_algorithm_in_constructor() -> None:
    with pytest.raises(ValueError) as excinfo:
        thresher.Thresher(algorithm="does-not-exist")

    message = str(excinfo.value)
    assert "does-not-exist" in message
    # the message should list what the caller could have used instead
    for name in algorithm.available_algorithms:
        assert name in message


def test_unknown_algorithm_in_set_algorithm() -> None:
    # This used to print a warning and silently keep the previous algorithm, so the caller
    # believed a switch had happened when it had not.
    t = thresher.Thresher(algorithm="grid")

    with pytest.raises(ValueError):
        t.set_algorithm("does-not-exist")

    assert t.get_current_algorithm()["name"] == "grid"


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("sim", "gen"),
        ("genetic", "gen"),
        ("linear", "ls"),
        ("linear_search", "ls"),
        ("gs", "grid"),
        ("s-grid", "sgrid"),
        ("curve_fitting", "sgd"),
        ("default", "exact"),
        ("auto", "exact"),
    ],
)
def test_known_aliases_still_resolve(alias: str, expected: str) -> None:
    assert thresher.Thresher(algorithm=alias).get_current_algorithm()["name"] == expected


def test_single_class_labels() -> None:
    with pytest.raises(ValueError, match="single class"):
        thresher.Thresher().optimize_threshold([0.1, 0.2, 0.3], [-1, -1, -1])


def test_unmapped_labels_point_at_the_labels_option() -> None:
    with pytest.raises(ValueError, match="labels"):
        thresher.Thresher().optimize_threshold([0.1, 0.2], [0, 1])


def test_empty_input() -> None:
    with pytest.raises(ValueError):
        thresher.Thresher().optimize_threshold([], [])


@pytest.mark.parametrize("argument", ["scores", "actual_classes"])
def test_non_iterable_arguments(argument: str) -> None:
    kwargs: dict[str, object] = {"scores": [0.1, 0.2], "actual_classes": [-1, 1]}
    kwargs[argument] = 7  # type: ignore[assignment]

    with pytest.raises(AttributeError):
        thresher.Thresher().optimize_threshold(**kwargs)  # type: ignore[arg-type]


def test_validation_survives_optimized_mode() -> None:
    """The old check was an `assert`, which `python -O` strips entirely.

    Malformed input would then reach the solvers instead of being rejected.
    """
    source = (
        "import thresher\n"
        "try:\n"
        "    thresher.Thresher().optimize_threshold([0.1, 0.2, 0.3], [-1, -1, -1])\n"
        "except ValueError as exc:\n"
        "    print(type(exc).__name__)\n"
        "    raise SystemExit(3) from None\n"
        "raise SystemExit(0)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-O", "-c", source], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 3, f"invalid input was accepted under -O: {completed.stderr}"
    # Caught as a plain ValueError, which is what code written before 0.4.5 does, and
    # reported as the specific type, which is what code written after it can do.
    assert completed.stdout.strip() == "SingleClassError"


class TestLengthMismatch:
    """Every score needs the class it belongs to.

    The solvers pair the two with `zip`, which stops at the shorter sequence, so before
    0.4.4 a mismatch was absorbed in silence and the surplus simply discarded. Six scores
    against four classes returned a threshold computed from four of them, with nothing in
    the result to say so.
    """

    def test_more_scores_than_classes(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            thresher.Thresher().optimize_threshold([0.1, 0.2, 0.3, 0.4, 0.9, 0.95], [-1, -1, 1, 1])

    def test_more_classes_than_scores(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            thresher.Thresher().optimize_threshold([0.1, 0.9], [-1, 1, 1, 1])

    def test_the_message_names_both_counts(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            thresher.Thresher().optimize_threshold([0.1, 0.2, 0.3], [-1, 1])

        message = str(excinfo.value)
        assert "3 scores" in message
        assert "2 entries" in message

    @pytest.mark.parametrize("algorithm_name", ["exact", "hist", "ls", "grid", "sgrid", "gen", "sgd"])
    def test_every_algorithm_rejects_it(self, algorithm_name: str) -> None:
        # The check sits in run_computations, so no solver can be reached with ragged input.
        with pytest.raises(ValueError, match="same length"):
            thresher.Thresher(algorithm=algorithm_name).optimize_threshold([0.1, 0.2, 0.3, 0.9], [-1, 1])

    def test_mismatch_is_caught_after_custom_labels_are_mapped(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            thresher.Thresher(labels=(0, 1)).optimize_threshold([0.1, 0.2, 0.3], [0, 1])

    def test_matching_lengths_are_unaffected(self) -> None:
        assert thresher.Thresher().optimize_threshold([0.1, 0.3, 0.4, 0.7], [-1, -1, 1, 1])


class TestMissingLabels:
    """A blank cell in a CSV arrives as NaN, which is absent rather than mis-encoded."""

    def test_missing_labels_are_named_as_missing(self) -> None:
        with pytest.raises(ValueError, match="missing value"):
            thresher.Thresher().optimize_threshold([0.1, 0.5, 0.9], [-1, float("nan"), 1])

    def test_the_message_counts_them(self) -> None:
        with pytest.raises(ValueError, match="2 missing value"):
            thresher.Thresher().optimize_threshold([0.1, 0.5, 0.9, 0.95], [-1, float("nan"), float("nan"), 1])

    def test_it_does_not_send_you_to_the_labels_option(self) -> None:
        # That advice fits a differently-encoded class, not an absent one, and would send
        # someone looking for a mapping that cannot exist.
        with pytest.raises(ValueError) as excinfo:
            thresher.Thresher().optimize_threshold([0.1, 0.9], [float("nan"), 1])

        assert "--labels" not in str(excinfo.value)
        assert "labels=" not in str(excinfo.value)

    def test_real_labels_are_still_reported_as_unmapped(self) -> None:
        with pytest.raises(ValueError, match="labels"):
            thresher.Thresher().optimize_threshold([0.1, 0.9], [0, 1])

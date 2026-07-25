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
        ("default", "auto"),
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
    source = "import thresher;thresher.Thresher().optimize_threshold([0.1, 0.2, 0.3], [-1, -1, -1])"
    completed = subprocess.run(
        [sys.executable, "-O", "-c", source], capture_output=True, text=True, check=False
    )

    assert completed.returncode != 0, "invalid input was accepted under -O"
    assert "ValueError" in completed.stderr

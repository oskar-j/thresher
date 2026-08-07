"""Input validation and error reporting, fixed in 0.2.3.

Bad input previously surfaced as a bare StopIteration or a message-less AssertionError,
neither of which told the caller what was wrong.
"""

import subprocess
import sys
from pathlib import Path

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

    # The message must blame the argument that is actually at fault - it used to name
    # "scores" whichever of the two was passed wrong.
    with pytest.raises(AttributeError, match=argument):
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


class TestConstructorOptions:
    """Mistyped or malformed constructor options fail at construction, fixed in 0.6.2.

    A wrong option name used to be merged into the options dict and never read, so
    `Thresher(algoritm='gen')` silently ran the default algorithm - the caller believed
    they had configured a run they had not.
    """

    def test_a_mistyped_option_name_is_rejected(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            thresher.Thresher(algoritm="gen")

        message = str(excinfo.value)
        assert "algoritm" in message
        # the message should list what the caller could have used instead
        assert "'algorithm'" not in message.split("Valid options")[0]
        assert "algorithm" in message.split("Valid options")[1]

    def test_every_documented_option_is_still_accepted(self) -> None:
        t = thresher.Thresher(
            algorithm="exact",
            allow_parallel=False,
            verbose=False,
            verbosity="warning",
            progress_bar=False,
            algorithm_params={},
            labels=(0, 1),
            backend="local",
        )

        assert t.optimize_threshold([0.1, 0.9], [0, 1]) == pytest.approx(0.5)

    def test_a_non_string_algorithm_is_a_value_error(self) -> None:
        # This used to escape as a bare AttributeError from `.lower()`, which the
        # constructor's documented contract (and the CLI's handlers) never mention.
        with pytest.raises(ValueError):
            thresher.Thresher(algorithm=123)

        with pytest.raises(ValueError):
            thresher.Thresher().set_algorithm(None)  # type: ignore[arg-type]

    def test_non_iterable_labels_are_rejected_at_construction(self) -> None:
        # Previously discarded in silence; the eventual error then told the caller to do
        # the thing they had already done.
        with pytest.raises(TypeError, match="list or a tuple"):
            thresher.Thresher(labels=5)

    @pytest.mark.parametrize("mapping", [(0,), (0, 1, 2)])
    def test_labels_of_the_wrong_length_are_rejected(self, mapping: tuple[int, ...]) -> None:
        # A one-item mapping previously reached map_labels and died as a bare IndexError.
        with pytest.raises(TypeError, match="exactly two"):
            thresher.Thresher(labels=mapping)

    def test_labels_none_means_no_mapping(self) -> None:
        result = thresher.Thresher(labels=None).optimize_threshold([0.1, 0.9], [-1, 1])

        assert result == pytest.approx(0.5)


class TestAlgorithmParams:
    """Mistyped `algorithm_params` keys fail at construction, fixed in 0.6.3.

    Every solver reads its parameters through `get_or_default`, so an unknown key was
    simply absent and the default stayed in place - the run continued with the value the
    caller believed they had replaced. That silence is also how the README came to
    document `optimized_start` for years after the code stopped reading it.
    """

    def test_a_mistyped_param_is_rejected(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            thresher.Thresher(algorithm="sgd", algorithm_params={"stoch_ration": 0.5})

        message = str(excinfo.value)
        assert "stoch_ration" in message
        # and it should say what this algorithm does read
        assert "stoch_ratio," in message or "stoch_ratio" in message.split("It reads:")[1]

    def test_the_phantom_parameter_from_the_readme_is_rejected(self) -> None:
        """`optimized_start` was documented but never read; it now says so."""
        with pytest.raises(ValueError, match="optimized_start"):
            thresher.Thresher(algorithm="gen", algorithm_params={"optimized_start": True})

    @pytest.mark.parametrize(
        ("algorithm_name", "params"),
        [
            ("hist", {"no_of_bins": 64}),
            ("sgd", {"stoch_ratio": 0.5, "num_of_iters": 10, "alpha": 0.02}),
            ("gen", {"population_size": 8, "number_of_generations": 2, "mutation_chance": 0.1}),
            ("grid", {"no_of_decimal_places": 1}),
            ("sgrid", {"no_of_decimal_places": 1, "stoch_ratio": 0.5, "reshuffle": True}),
            ("ls", {"n_jobs": 1}),
            ("exact", {}),
        ],
    )
    def test_documented_parameters_are_accepted(self, algorithm_name: str, params: dict[str, object]) -> None:
        t = thresher.Thresher(algorithm=algorithm_name, algorithm_params=params)

        assert t.optimize_threshold([0.1, 0.2, 0.8, 0.9], [-1, -1, 1, 1])

    def test_exact_takes_no_parameters_at_all(self) -> None:
        # It is exact, so there is no accuracy to trade for speed - and the message says so
        # rather than listing an empty set.
        with pytest.raises(ValueError, match="nothing to tune"):
            thresher.Thresher(algorithm="exact", algorithm_params={"no_of_bins": 64})

    def test_a_stochastic_only_param_is_rejected_for_exhaustive_grid(self) -> None:
        """`grid` and `sgrid` share an implementation, but not their parameters.

        `stoch_ratio` is read only on the stochastic path, so passing it to `grid` does
        nothing - the same failure mode as a typo.
        """
        with pytest.raises(ValueError, match="stoch_ratio"):
            thresher.Thresher(algorithm="grid", algorithm_params={"stoch_ratio": 0.5})

        assert thresher.Thresher(algorithm="sgrid", algorithm_params={"stoch_ratio": 0.5})

    def test_switching_algorithms_revalidates_the_parameters(self) -> None:
        # The params were valid for sgd; they are not for exact, and quietly dropping them
        # here would recreate exactly what construction-time validation prevents.
        t = thresher.Thresher(algorithm="sgd", algorithm_params={"stoch_ratio": 0.5})

        with pytest.raises(ValueError, match="stoch_ratio"):
            t.set_algorithm("exact")

        assert t.get_current_algorithm()["name"] == "sgd", "the switch must not half-happen"

    def test_a_non_mapping_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be a mapping"):
            thresher.Thresher(algorithm_params=[("no_of_bins", 64)])


class TestUndefinedScores:
    """Scores that are not numbers, fixed in 0.7.1 (#23).

    Only the labels were ever checked. A NaN score reached the solvers and each failed its
    own way: `exact` sorted it into place and returned it as the answer - a "threshold"
    that classifies everything negative, since every comparison against NaN is false -
    while `hist` raised a bare `ValueError` from its bin arithmetic. One NaN in a
    `predict_proba` column is an ordinary upstream accident.
    """

    @pytest.mark.parametrize("algorithm_name", ["exact", "hist", "ls", "grid", "sgrid", "gen", "sgd"])
    def test_every_algorithm_refuses_a_nan_score(self, algorithm_name: str) -> None:
        scores = [0.1, float("nan"), 0.6, 0.9] * 30
        actual_classes = [-1, -1, 1, 1] * 30

        with pytest.raises(ValueError, match="not a number"):
            thresher.Thresher(algorithm=algorithm_name).optimize_threshold(scores, actual_classes)

    def test_it_no_longer_returns_nan_as_a_threshold(self) -> None:
        """The failure this replaces: a threshold nothing can ever exceed."""
        with pytest.raises(ValueError):
            thresher.Thresher().optimize_threshold([0.1, float("nan"), 0.6, 0.9], [-1, -1, 1, 1])

    def test_the_message_counts_them(self) -> None:
        with pytest.raises(ValueError, match="2 value"):
            thresher.Thresher().optimize_threshold([0.1, float("nan"), float("nan"), 0.9], [-1, -1, 1, 1])

    def test_none_is_refused_too(self) -> None:
        """What a blank looks like in a plain list; it used to die in the sort.

        Annotated away because a `None` score is exactly what the signature forbids - the
        point is that reaching the solvers with one no longer produces a bare `TypeError`.
        """
        scores: list[float] = [0.1, None, 0.6, 0.9]  # type: ignore[list-item]

        with pytest.raises(ValueError, match="not a number"):
            thresher.Thresher().optimize_threshold(scores, [-1, -1, 1, 1])

    def test_infinities_are_still_allowed(self) -> None:
        """They order against everything else, so a threshold can be placed near them."""
        result = thresher.Thresher().optimize_threshold([0.1, float("inf"), 0.6, 0.9], [-1, 1, 1, -1])

        assert result == pytest.approx(0.35)

    def test_a_length_mismatch_is_still_reported_first(self) -> None:
        # Ragged input is the more basic complaint, and naming it first is more useful.
        with pytest.raises(ValueError, match="same length"):
            thresher.Thresher().optimize_threshold([0.1, float("nan"), 0.6], [-1, 1])

    def test_the_cli_reports_it_rather_than_printing_nan(self, tmp_path: Path) -> None:
        """It used to print `nan` and exit 0, which reads as a successful run."""
        source = tmp_path / "scores.csv"
        source.write_text("score,actual_class\n0.1,-1\nnan,-1\n0.6,1\n0.9,1\n")

        completed = subprocess.run(
            [sys.executable, "-m", "thresher.cli", str(source)],
            capture_output=True,
            text=True,
            check=False,
        )

        assert completed.returncode == 1
        assert "nan" not in completed.stdout
        assert "not a number" in completed.stderr

"""Helpers and reporting paths.

These are the parts the end-to-end tests reach around rather than through: the shared
utilities, and everything guarded by `verbose` or `progress_bar`. They are tested here on
their own terms - what each returns or prints - rather than being exercised incidentally.
"""

import importlib.metadata
import math
import re
from collections.abc import Callable

import pandas as pd
import pytest

import thresher
from thresher.algs.common.meta_optimizer import calculate_range_mean, get_mean_value_for_class_pd
from thresher.algs.common.stochastic import stochastic_process
from thresher.algs.common.tools import granularity_of_scores
from thresher.algs.linear.compute import process_batch
from thresher.utils import get_or_default, map_labels, pairwise, print_progress_bar

Dataset = tuple[list[float], list[int]]
DatasetFactory = Callable[..., Dataset]

REPORTING_ALGORITHMS = ["exact", "hist", "ls", "grid", "sgrid", "gen", "sgd"]


class TestScoreHelpers:
    def test_granularity_rounds_to_the_requested_places(self) -> None:
        assert list(granularity_of_scores([0.123, 0.456, 0.789])) == [0.12, 0.46, 0.79]

    def test_granularity_places_are_configurable(self) -> None:
        assert list(granularity_of_scores([0.123456], number_of_decimal_places=4)) == [0.1235]

    def test_granularity_keeps_duplicates(self) -> None:
        # It reduces precision, not cardinality - callers dedupe if they want to.
        assert list(granularity_of_scores([0.11, 0.12], number_of_decimal_places=1)) == [0.1, 0.1]

    def test_class_mean(self) -> None:
        scores, actual_classes = [0.1, 0.2, 0.8, 0.9], [-1, -1, 1, 1]

        assert calculate_range_mean(scores, actual_classes, -1) == pytest.approx(0.15)
        assert calculate_range_mean(scores, actual_classes, 1) == pytest.approx(0.85)

    def test_class_mean_of_an_absent_class_warns_and_is_not_a_number(self) -> None:
        """numpy averages an empty selection to NaN, and says so.

        The genetic solver seeds its population range from these two means, so an absent
        class would seed it with NaN. `validate_actual_classes` rejects single-class input
        before any solver runs, so this is unreachable through `optimize_threshold` - but
        it is reachable by calling the helper directly, which is worth knowing.
        """
        # numpy emits two RuntimeWarnings here - "Mean of empty slice" and then "invalid
        # value encountered in scalar divide". Matching on the category rather than the
        # text catches both; matching one text re-emits the other, which this suite's
        # `filterwarnings = ["error"]` would then raise.
        with pytest.warns(RuntimeWarning):
            result = calculate_range_mean([0.1, 0.2], [-1, -1], 1)

        assert math.isnan(result)

    def test_class_mean_from_a_dataframe(self) -> None:
        frame = pd.DataFrame({"pred": [0.1, 0.2, 0.8, 0.9], "actual": [-1, -1, 1, 1]})

        assert get_mean_value_for_class_pd(1, "actual", frame, "pred") == pytest.approx(0.85)


class TestStochasticProcess:
    def test_reports_the_mis_classification_ratio(self) -> None:
        # The whole dataset is sampled, so the result is exact and comparable.
        scores, actual_classes = [0.1, 0.2, 0.8, 0.9], [-1, -1, 1, 1]

        assert stochastic_process(0.5, scores, actual_classes, random_factor=1.0) == 0.0

    def test_can_report_accuracy_instead(self) -> None:
        scores, actual_classes = [0.1, 0.2, 0.8, 0.9], [-1, -1, 1, 1]

        assert stochastic_process(0.5, scores, actual_classes, 1.0, miss_class=False) == 1.0

    def test_the_two_modes_are_complementary(self) -> None:
        scores, actual_classes = [0.1, 0.4, 0.6, 0.9], [1, -1, 1, -1]

        missed = stochastic_process(0.5, scores, actual_classes, 1.0)
        hit = stochastic_process(0.5, scores, actual_classes, 1.0, miss_class=False)

        assert missed + hit == pytest.approx(1.0)


class TestUtils:
    def test_pairwise(self) -> None:
        assert list(pairwise([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]

    def test_pairwise_of_a_short_sequence_is_empty(self) -> None:
        assert list(pairwise([1])) == []
        assert list(pairwise([])) == []

    def test_get_or_default(self) -> None:
        assert get_or_default({"a": 1}, "a", 99) == 1
        assert get_or_default({"a": 1}, "b", 99) == 99

    def test_map_labels(self) -> None:
        assert list(map_labels([0, 1, 0], (0, 1))) == [-1, 1, -1]

    def test_map_labels_rejects_a_non_sequence_mapping(self) -> None:
        with pytest.raises(TypeError, match="list or a tuple"):
            list(map_labels([0, 1], {0, 1}))

    def test_map_labels_rejects_an_unmapped_value(self) -> None:
        with pytest.raises(TypeError, match="not found in the mapping"):
            list(map_labels([0, 1, 7], (0, 1)))

    def test_progress_bar_draws_and_finishes(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_progress_bar(5, 10, prefix="working", suffix="done", length=10)
        midway = capsys.readouterr().out

        assert "working" in midway
        assert "50.0%" in midway
        assert not midway.endswith("\n"), "an unfinished bar redraws in place"

        print_progress_bar(10, 10, length=10)
        complete = capsys.readouterr().out

        assert "100.0%" in complete
        assert complete.endswith("\n"), "a finished bar ends the line"


class TestLinearSearchInternals:
    def test_process_batch_scores_one_candidate(self) -> None:
        # Runs inside worker processes during a parallel search, so it is called directly
        # here - coverage and assertions do not reach into a subprocess.
        scores, actual_classes = [0.1, 0.4, 0.6, 0.9], [-1, -1, 1, 1]

        threshold, accuracy = process_batch(scores, actual_classes, 0.5)

        assert threshold == 0.5
        assert accuracy == 1.0

    def test_process_batch_returns_its_threshold(self) -> None:
        # Results come back from the pool out of order, so each carries its own candidate.
        _, accuracy = process_batch([0.1, 0.9], [-1, 1], 0.95)

        assert process_batch([0.1, 0.9], [-1, 1], 0.95)[0] == 0.95
        assert accuracy == 0.5


class TestReporting:
    """`verbose` and `progress_bar` are user-facing options and should do something."""

    @pytest.mark.parametrize("algorithm_name", REPORTING_ALGORITHMS)
    def test_verbose_explains_what_it_is_doing(
        self, capsys: pytest.CaptureFixture[str], algorithm_name: str
    ) -> None:
        scores = [0.1, 0.15, 0.2, 0.22, 0.4, 0.7, 0.8, 0.9]
        actual_classes = [-1, -1, -1, -1, 1, 1, 1, 1]

        thresher.Thresher(algorithm=algorithm_name, verbose=True).optimize_threshold(scores, actual_classes)

        assert capsys.readouterr().out.strip(), f"{algorithm_name} said nothing when asked to"

    @pytest.mark.parametrize("algorithm_name", REPORTING_ALGORITHMS)
    def test_progress_bar_reaches_completion(
        self, capsys: pytest.CaptureFixture[str], algorithm_name: str
    ) -> None:
        scores = [0.1, 0.15, 0.2, 0.22, 0.4, 0.7, 0.8, 0.9]
        actual_classes = [-1, -1, -1, -1, 1, 1, 1, 1]

        thresher.Thresher(algorithm=algorithm_name, progress_bar=True).optimize_threshold(
            scores, actual_classes
        )

        output = capsys.readouterr().out
        # sgd reports through verbose only and never draws a bar; the rest must finish.
        if algorithm_name != "sgd":
            assert "100.0%" in output, f"{algorithm_name} left its progress bar unfinished"

    def test_verbose_names_the_algorithm_that_ran(self, capsys: pytest.CaptureFixture[str]) -> None:
        thresher.Thresher(verbose=True).optimize_threshold([0.1, 0.3, 0.4, 0.7], [-1, -1, 1, 1])

        assert "Exact sweep" in capsys.readouterr().out

    def test_verbose_and_progress_bar_together_on_the_genetic_solver(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # It refuses to do both, because they would fight over the terminal.
        scores = [0.1, 0.15, 0.2, 0.22, 0.4, 0.7, 0.8, 0.9]
        actual_classes = [-1, -1, -1, -1, 1, 1, 1, 1]

        thresher.Thresher(algorithm="gen", verbose=True, progress_bar=True).optimize_threshold(
            scores, actual_classes
        )

        assert "automatically disables a progress bar" in capsys.readouterr().out


class TestEmptyAndDegenerateInput:
    def test_exact_rejects_an_empty_input(self) -> None:
        from thresher.algs.exact import compute as exact_compute

        with pytest.raises(ValueError, match="At least one score"):
            exact_compute.run([], [], verbose=False, progress_bar=False, alg_options={})

    def test_linear_search_needs_two_scores_to_find_a_midpoint(self) -> None:
        from thresher.algs.linear import compute as linear_compute

        with pytest.raises(ValueError, match="At least two scores"):
            linear_compute.run([0.5], [1], verbose=False, progress_bar=False)

    def test_stochastic_grid_with_reshuffling(self) -> None:
        # Draws a fresh subsample per candidate rather than reusing one.
        scores = [0.1, 0.15, 0.2, 0.22, 0.4, 0.7, 0.8, 0.9]
        actual_classes = [-1, -1, -1, -1, 1, 1, 1, 1]

        result = thresher.Thresher(
            algorithm="sgrid", algorithm_params={"reshuffle": True, "stoch_ratio": 0.5}
        ).optimize_threshold(scores, actual_classes)

        assert 0.0 <= result <= 1.0


class TestDispatch:
    def test_set_algorithm_returns_the_instance_for_chaining(self) -> None:
        t = thresher.Thresher()

        chained = t.set_algorithm("grid")

        assert chained is t
        assert t.get_current_algorithm()["name"] == "grid"

    def test_an_unwired_algorithm_is_reported_rather_than_ignored(self) -> None:
        """The guard that fires when the registry and the dispatcher disagree.

        Adding an entry to `available_algorithms` without a matching branch in
        `run_computations` is the mistake this catches - it would otherwise fall off the
        end of the dispatch chain and return None.
        """
        from thresher.algorithm import Algorithm
        from thresher.dispatch import run_computations

        unwired = Algorithm(id="nope", full_name="Not wired up", synonyms=[], data_vol_thresh=1)

        with pytest.raises(NotImplementedError):
            run_computations(
                unwired,
                [0.1, 0.9],
                [-1, 1],
                verbose=False,
                progress_bar=False,
                allow_parallel=False,
                alg_options={},
            )


class TestPackageVersion:
    def test_version_matches_the_installed_distribution(self) -> None:
        """`__version__` must report the distribution's version, not a hardcoded copy.

        `pyproject.toml` is the single source of truth, so the attribute has to come from
        the installed metadata - a literal here would drift on the next release.
        """
        assert thresher.__version__ == importlib.metadata.version("thresher-py")

    def test_version_is_release_shaped(self) -> None:
        assert re.fullmatch(r"\d+\.\d+\.\d+", thresher.__version__)

    def test_version_is_exported(self) -> None:
        assert "__version__" in thresher.__all__

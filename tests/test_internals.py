"""Helpers and reporting paths.

These are the parts the end-to-end tests reach around rather than through: the shared
utilities, and everything guarded by `verbosity` or `progress_bar`. They are tested here
on their own terms - what each returns or reports - rather than being exercised
incidentally.
"""

import importlib.metadata
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import pytest

import thresher
from thresher import algorithm, dispatch
from thresher.algs.common.meta_optimizer import calculate_range_mean, get_mean_value_for_class_pd
from thresher.algs.common.stochastic import stochastic_process
from thresher.algs.common.tools import granularity_of_scores
from thresher.algs.grid import compute as grid_compute
from thresher.progress import print_progress_bar
from thresher.utils import get_or_default, map_labels, pairwise

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
        # stderr since 0.8.0: stdout is where the command line prints its answer.
        midway = capsys.readouterr().err

        assert "working" in midway
        assert "50.0%" in midway
        assert not midway.endswith("\n"), "an unfinished bar redraws in place"

        print_progress_bar(10, 10, length=10)
        complete = capsys.readouterr().err

        assert "100.0%" in complete
        assert complete.endswith("\n"), "a finished bar ends the line"

    def test_it_is_still_importable_from_utils(self, capsys: pytest.CaptureFixture[str]) -> None:
        """It moved to `thresher.progress` in 0.8.0, and has been importable here since
        the first release."""
        from thresher.utils import print_progress_bar as from_utils

        from_utils(1, 2, length=10)

        assert "50.0%" in capsys.readouterr().err


class TestReporting:
    """`verbosity` and `progress_bar` are user-facing options and should do something."""

    SCORES: ClassVar[list[float]] = [0.1, 0.15, 0.2, 0.22, 0.4, 0.7, 0.8, 0.9]
    CLASSES: ClassVar[list[int]] = [-1, -1, -1, -1, 1, 1, 1, 1]

    @pytest.mark.parametrize("algorithm_name", REPORTING_ALGORITHMS)
    def test_verbosity_explains_what_it_is_doing(self, logs: list[str], algorithm_name: str) -> None:
        thresher.Thresher(algorithm=algorithm_name, verbosity="debug").optimize_threshold(
            self.SCORES, self.CLASSES
        )

        assert logs, f"{algorithm_name} said nothing when asked to"

    @pytest.mark.parametrize("algorithm_name", REPORTING_ALGORITHMS)
    def test_every_algorithm_reports_its_own_run_and_not_only_the_dispatcher(
        self, logs: list[str], algorithm_name: str
    ) -> None:
        """Something has to come from the solver itself, not just from the way in.

        `interface` and `dispatch` log two lines for every run whatever the algorithm, so
        asserting only that *something* was logged would pass for a solver that says
        nothing at all.
        """
        thresher.Thresher(algorithm=algorithm_name, verbosity="debug").optimize_threshold(
            self.SCORES, self.CLASSES
        )

        from_solver = [line for line in logs if ":thresher.algs." in line]

        assert from_solver, f"{algorithm_name} reported nothing of its own"

    @pytest.mark.parametrize("algorithm_name", REPORTING_ALGORITHMS)
    def test_nothing_is_logged_at_the_default_level(self, logs: list[str], algorithm_name: str) -> None:
        """The default is 'warning', and an ordinary run has nothing to warn about."""
        thresher.Thresher(algorithm=algorithm_name).optimize_threshold(self.SCORES, self.CLASSES)

        assert logs == []

    @pytest.mark.parametrize("algorithm_name", REPORTING_ALGORITHMS)
    def test_progress_bar_reaches_completion(
        self, capsys: pytest.CaptureFixture[str], algorithm_name: str
    ) -> None:
        thresher.Thresher(algorithm=algorithm_name, progress_bar=True).optimize_threshold(
            self.SCORES, self.CLASSES
        )

        output = capsys.readouterr().err
        # sgd has no step count to report a proportion of; the rest must finish. The
        # percentage is asserted rather than the shape of the bar because either tqdm or
        # the built-in bar may have drawn it - they are formatted to agree on that much.
        if algorithm_name != "sgd":
            assert "100.0%" in output, f"{algorithm_name} left its progress bar unfinished"

    def test_verbosity_names_the_algorithm_that_ran(self, logs: list[str]) -> None:
        thresher.Thresher(verbosity="info").optimize_threshold([0.1, 0.3, 0.4, 0.7], [-1, -1, 1, 1])

        assert any("Exact sweep" in line for line in logs)

    def test_verbose_is_still_accepted_and_means_debug(self, logs: list[str]) -> None:
        """The option `verbosity` replaced. It printed every solver's per-step detail."""
        thresher.Thresher(algorithm="sgd", verbose=True).optimize_threshold(self.SCORES, self.CLASSES)

        assert any(line.startswith("DEBUG") for line in logs)

    def test_one_instance_being_verbose_does_not_make_another_one_verbose(self, logs: list[str]) -> None:
        """The level is per-call, which neither logging system offers on its own."""
        loud = thresher.Thresher(verbosity="debug")
        quiet = thresher.Thresher()

        loud.optimize_threshold(self.SCORES, self.CLASSES)
        spoken = len(logs)
        quiet.optimize_threshold(self.SCORES, self.CLASSES)

        assert spoken > 0
        assert len(logs) == spoken, "the second instance logged on the first one's setting"

    def test_a_progress_bar_gives_way_to_the_detail_it_would_overwrite(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Both go to stderr, so at DEBUG the bar is not drawn - see `thresher.progress`.

        Until 0.8.0 this rule was the genetic solver's alone, and it announced itself with
        a printed warning.
        """
        thresher.Thresher(algorithm="gen", verbosity="debug", progress_bar=True).optimize_threshold(
            self.SCORES, self.CLASSES
        )

        assert "100.0%" not in capsys.readouterr().err


class TestEmptyAndDegenerateInput:
    def test_exact_rejects_an_empty_input(self) -> None:
        from thresher.algs.exact import compute as exact_compute

        with pytest.raises(ValueError, match="At least one score"):
            exact_compute.run([], [], progress_bar=False, alg_options={})

    def test_linear_search_needs_two_scores_to_find_a_midpoint(self) -> None:
        from thresher.algs.linear import compute as linear_compute

        with pytest.raises(ValueError, match="At least two scores"):
            linear_compute.run([0.5], [1], progress_bar=False)

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


class TestDocumentedParameters:
    """The README is the only parameter documentation, so it has to match the code.

    `optimized_start` sat in that list for years after the genetic solver stopped reading
    it (#34): nothing compared the two, and an unknown key was silently ignored, so
    following the documentation produced no error and no effect. These compare them.
    """

    README = Path(__file__).resolve().parent.parent / "README.md"

    def documented_parameters(self) -> set[str]:
        """Every parameter named in the README's algorithm sections.

        Two shapes are in use: the older sections list parameters as `* `name` (default:
        ...)` bullets, and `hist` documents its one parameter as a `| `name` | default |`
        table row. Both count as documentation.
        """
        readme = self.README.read_text()
        documented = set(re.findall(r"^\* `([a-z_0-9]+)`", readme, flags=re.MULTILINE))

        # Only rows under a "Parameter" heading - the README's other tables compare
        # algorithms, and their first column holds algorithm ids in the same backticks.
        in_parameter_table = False
        for line in readme.splitlines():
            if line.startswith("| Parameter |"):
                in_parameter_table = True
                continue
            if in_parameter_table:
                row = re.match(r"^\| `([a-z_0-9]+)` \|", line)
                if row:
                    documented.add(row.group(1))
                elif not line.startswith("|"):
                    in_parameter_table = False
        return documented

    def test_every_documented_parameter_is_read_by_some_algorithm(self) -> None:
        implemented = set().union(*dispatch.KNOWN_PARAMS.values())

        assert self.documented_parameters() - implemented == set(), (
            "the README documents parameters no algorithm reads - passing one does nothing"
        )

    def test_every_implemented_parameter_is_documented(self) -> None:
        implemented = set().union(*dispatch.KNOWN_PARAMS.values())

        assert implemented - self.documented_parameters() == set(), (
            "an algorithm reads a parameter the README does not mention"
        )

    def test_every_algorithm_has_an_entry_in_the_table(self) -> None:
        """A new solver must declare its parameters, even if it has none."""
        assert set(dispatch.KNOWN_PARAMS) == set(algorithm.available_algorithms)


class TestStochasticGridSampling:
    """`reshuffle` draws its subsample by index, fixed in 0.6.4 (#27).

    It used to build every `(score, class)` pair before sampling from them, so each
    candidate cost a full pass over the data however small `stoch_ratio` was. That made
    the option `O(c·n)` against its documented `O(c·r·n)`, and slower than the exhaustive
    grid it exists to approximate.
    """

    def test_it_reads_only_the_sampled_rows(self) -> None:
        """Counting reads is what separates the two implementations.

        A wall-clock assertion would be flaky; this counts instead, and a return to
        materialising the pairs would read every row per candidate and fail.
        """
        size, sample_ratio = 2000, 0.05
        reads = [0] * size

        class CountingScores(list[float]):
            def __getitem__(self, index: Any) -> Any:  # type: ignore[override]
                if isinstance(index, int):
                    reads[index] += 1
                return super().__getitem__(index)

        scores = CountingScores([value / size for value in range(size)])
        actual_classes = [-1 if value < size // 2 else 1 for value in range(size)]

        grid_compute.run_stoch(
            scores,
            actual_classes,
            progress_bar=False,
            alg_options={"stoch_ratio": sample_ratio, "reshuffle": True, "no_of_decimal_places": 1},
        )

        # 11 candidates * 5% of 2,000 rows = ~1,100 reads, plus the two the grid needs for
        # the data's range. Materialising the pairs would be 11 * 2,000 = 22,000.
        assert sum(reads) < size * 2, (
            f"read {sum(reads)} values from {size} rows - a full pass per candidate?"
        )

    def test_the_sample_is_drawn_per_candidate_when_reshuffling(self) -> None:
        draws = []
        original = grid_compute._get_random_projection

        def counting(*args: Any, **kwargs: Any) -> Any:
            draws.append(1)
            return original(*args, **kwargs)

        scores = [value / 100 for value in range(100)]
        actual_classes = [-1 if value < 50 else 1 for value in range(100)]

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(grid_compute, "_get_random_projection", counting)
            grid_compute.run_stoch(
                scores,
                actual_classes,
                progress_bar=False,
                alg_options={"reshuffle": True, "no_of_decimal_places": 1},
            )

        # One per candidate: 10**1 + 1 grid points, plus the below-minimum edge candidate.
        assert len(draws) == 12

    def test_one_sample_is_reused_when_not_reshuffling(self) -> None:
        draws = []
        original = grid_compute._get_random_projection

        def counting(*args: Any, **kwargs: Any) -> Any:
            draws.append(1)
            return original(*args, **kwargs)

        scores = [value / 100 for value in range(100)]
        actual_classes = [-1 if value < 50 else 1 for value in range(100)]

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(grid_compute, "_get_random_projection", counting)
            grid_compute.run_stoch(
                scores,
                actual_classes,
                progress_bar=False,
                alg_options={"reshuffle": False, "no_of_decimal_places": 1},
            )

        assert len(draws) == 1

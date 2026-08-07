"""Which progress bar draws, where it draws, and when it does not.

0.8.0 made tqdm an optional extra and the bar this package carries its fallback. That
creates one property worth pinning down above all others - installing the extra changes
how a run looks and nothing else about it - and a set of places where no bar belongs at
all, which is the other half of what a progress bar being *appropriate* means.
"""

import io
import sys
from typing import Any

import pytest

import thresher
from thresher import log, progress

Dataset = tuple[list[float], list[int]]

SCORES = [0.1, 0.15, 0.2, 0.22, 0.4, 0.7, 0.8, 0.9]
CLASSES = [-1, -1, -1, -1, 1, 1, 1, 1]

#: Every algorithm that draws one. `sgd` has no step count to report a proportion of.
DRAWING_ALGORITHMS = ["exact", "hist", "ls", "grid", "sgrid", "gen"]


@pytest.fixture
def without_tqdm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run as though the `progress` extra had never been installed.

    The extra *is* installed in the dev environment - deliberately, so the tqdm path is
    exercised - which leaves the fallback unreachable without hiding it. Setting the
    module-level handle to None is exactly the state the import guard leaves behind when
    tqdm is absent.
    """
    monkeypatch.setattr(progress, "_tqdm", None)


class TestTheBackendDoesNotChangeTheAnswer:
    """The whole promise of making tqdm optional."""

    @pytest.mark.parametrize("algorithm_name", DRAWING_ALGORITHMS)
    def test_the_threshold_is_the_same_with_and_without_tqdm(
        self, monkeypatch: pytest.MonkeyPatch, algorithm_name: str
    ) -> None:
        import random

        def solve() -> float:
            # The stochastic solvers sample, so the two runs are only comparable from the
            # same starting state.
            random.seed(11)
            return thresher.Thresher(algorithm=algorithm_name, progress_bar=True).optimize_threshold(
                SCORES, CLASSES
            )

        with_tqdm = solve()
        monkeypatch.setattr(progress, "_tqdm", None)
        without = solve()

        assert with_tqdm == without

    def test_a_drawn_bar_does_not_change_the_answer_either(self) -> None:
        drawn = thresher.Thresher(algorithm="hist", progress_bar=True).optimize_threshold(SCORES, CLASSES)
        undrawn = thresher.Thresher(algorithm="hist").optimize_threshold(SCORES, CLASSES)

        assert drawn == undrawn

    def test_tqdm_is_what_draws_when_it_is_installed(self) -> None:
        pytest.importorskip("tqdm")

        bar = progress.make_progress(10, "working", enabled=True, stream=io.StringIO())

        assert isinstance(bar, progress._TqdmBar)
        assert progress.tqdm_available()

    def test_the_built_in_bar_draws_when_it_is_not(self, without_tqdm: None) -> None:
        bar = progress.make_progress(10, "working", enabled=True, stream=io.StringIO())

        assert isinstance(bar, progress._BuiltInBar)
        assert not progress.tqdm_available()

    @pytest.mark.parametrize("hide_tqdm", [False, True])
    def test_both_report_the_percentage_the_same_way(
        self, monkeypatch: pytest.MonkeyPatch, hide_tqdm: bool
    ) -> None:
        """tqdm is given a `bar_format` shaped like the built-in bar, on purpose.

        Without it, installing the extra would change what a script watching stderr sees -
        and the tests here would have to assert two different things. What they cannot
        agree on is *when* a frame is drawn: tqdm coalesces redraws that arrive inside a
        tenth of a second and the built-in bar does not, so only the finished bar is
        guaranteed to have been written by both.
        """
        if hide_tqdm:
            monkeypatch.setattr(progress, "_tqdm", None)
        stream = io.StringIO()

        with progress.make_progress(4, "working", enabled=True, stream=stream) as bar:
            bar.update(2)

        drawn = stream.getvalue()
        assert "100.0%" in drawn
        assert "[4/4]" in drawn, "and both say how many steps that was"
        assert "working" in drawn


class TestWhereABarIsDrawn:
    """A bar is a thing somebody watches, so it goes where somebody is watching."""

    @pytest.mark.parametrize("algorithm_name", DRAWING_ALGORITHMS)
    def test_it_goes_to_stderr_and_never_to_stdout(
        self, capsys: pytest.CaptureFixture[str], algorithm_name: str
    ) -> None:
        """stdout is where the command line prints the threshold, so it has to stay clean.

        The built-in bar wrote there until 0.8.0. Nothing had noticed because the command
        line had no way to ask for a bar - and 0.8.0 adds one.
        """
        capsys.readouterr()

        thresher.Thresher(algorithm=algorithm_name, progress_bar=True).optimize_threshold(SCORES, CLASSES)

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "100.0%" in captured.err

    def test_the_stream_is_read_when_the_bar_is_built_not_when_the_module_loads(self) -> None:
        """Otherwise a redirected stderr - pytest's, click's - would be written past."""
        replacement = io.StringIO()
        original = sys.stderr
        sys.stderr = replacement
        try:
            with progress.make_progress(2, "working", enabled=True) as bar:
                bar.update(1)
        finally:
            sys.stderr = original

        assert "100.0%" in replacement.getvalue()

    def test_nothing_is_drawn_when_none_was_asked_for(self) -> None:
        stream = io.StringIO()

        with progress.make_progress(10, "working", enabled=False, stream=stream) as bar:
            bar.update(5)

        assert stream.getvalue() == ""

    @pytest.mark.parametrize("total", [0, -1])
    def test_nothing_is_drawn_for_a_job_with_no_length(self, total: int) -> None:
        """A proportion of nothing is not a proportion, and would divide by zero."""
        stream = io.StringIO()

        with progress.make_progress(total, "working", enabled=True, stream=stream) as bar:
            bar.update(1)

        assert stream.getvalue() == ""
        assert isinstance(bar, progress._NoProgress)

    def test_the_log_wins_at_debug(self) -> None:
        """Both write to stderr, so at DEBUG the bar would be cut apart by log lines."""
        stream = io.StringIO()

        with log.verbosity("debug"):
            bar = progress.make_progress(10, "working", enabled=True, stream=stream)

        assert isinstance(bar, progress._NoProgress)

    def test_a_bar_is_still_drawn_at_info(self) -> None:
        """INFO is a few lines per run, not one per step, so the two do not collide."""
        stream = io.StringIO()

        with log.verbosity("info"):
            bar = progress.make_progress(10, "working", enabled=True, stream=stream)

        assert not isinstance(bar, progress._NoProgress)

    def test_the_workers_of_a_parallel_run_draw_nothing(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Several processes redrawing one line would interleave into nonsense.

        The functions the `mp` backend ships to workers do no reporting at all, which is
        part of what makes them safe to ship - see `thresher.backends.base`.
        """
        capsys.readouterr()

        thresher.Thresher(backend="mp", progress_bar=True).optimize_threshold(SCORES, CLASSES)

        assert capsys.readouterr().out == ""


class TestTheBuiltInBar:
    """The fallback, on its own terms."""

    def test_it_ends_its_line_exactly_once(self, without_tqdm: None) -> None:
        stream = io.StringIO()

        bar = progress.make_progress(4, "working", enabled=True, stream=stream)
        bar.update(4)
        bar.close()
        bar.close()
        # A late step - `sweep_bins` reports the last bin and then the loop ends - must not
        # draw on the line the finished bar already ended.
        bar.update(4)

        assert stream.getvalue().count("\n") == 1, "a closed bar redrew itself"

    def test_closing_it_early_still_finishes_the_line(self, without_tqdm: None) -> None:
        """Which is what the `with` block does when the work inside it raises."""
        stream = io.StringIO()

        with (
            pytest.raises(RuntimeError),
            progress.make_progress(4, "working", enabled=True, stream=stream) as bar,
        ):
            bar.update(1)
            raise RuntimeError("the search failed")

        assert stream.getvalue().endswith("\n")

    def test_a_bar_that_only_brackets_its_work_shows_both_ends(self, without_tqdm: None) -> None:
        """`ls` and `grid` score every candidate in one batched call, so there is no
        middle to report - the bar says 0% and then 100%."""
        stream = io.StringIO()

        with progress.make_progress(101, "batched", enabled=True, stream=stream) as bar:
            bar.update(0)

        drawn = stream.getvalue()
        assert "0.0%" in drawn
        assert "100.0%" in drawn


class TestTheTqdmBar:
    """The adapter, on its own terms."""

    def test_going_backwards_does_not_rewind_it(self) -> None:
        """tqdm counts increments; the call sites here count from the start.

        `sweep_class_counts` reports "bin 500 of 1,024", so the difference is worked out
        in the adapter. A negative one would be tqdm's way of being told to go back.
        """
        pytest.importorskip("tqdm")
        stream = io.StringIO()

        bar: Any = progress.make_progress(10, "working", enabled=True, stream=stream)
        bar.update(6)
        bar.update(3)

        # Read off the bar rather than off the stream: tqdm coalesces redraws, so an
        # intermediate frame may never have been written even though the count moved.
        assert bar.bar.n == 6
        bar.close()

    def test_it_is_filled_to_the_end_when_it_closes(self) -> None:
        """A search that stops early - sgd's patience, a tie found on the first bin -
        still leaves a finished bar rather than one stuck at 30%."""
        pytest.importorskip("tqdm")
        stream = io.StringIO()

        bar: Any = progress.make_progress(10, "working", enabled=True, stream=stream)
        bar.update(3)
        bar.close()

        assert "100.0%" in stream.getvalue()

"""What the package says, and who decides how much of it.

0.8.0 moved every message onto loguru and put one setting in front of all of them. The
tests here are about that setting - that it selects, that it is validated, that it is
scoped to the caller who set it - and about the property the whole change exists to
create: that nothing in the package writes to a console on its own initiative.
"""

import ast
import threading
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

import thresher
from thresher import log
from thresher.exceptions import ConfigurationError, ThresherError

SOURCE_DIR = Path(thresher.__file__).parent

Dataset = tuple[list[float], list[int]]

TINY: Dataset = ([0.1, 0.3, 0.4, 0.7], [-1, -1, 1, 1])


class TestNothingPrints:
    """The property the release is named for, asserted rather than reviewed for.

    Twenty-odd `print()` calls used to sit behind `if verbose:`, writing to stdout - the
    stream the command line reserves for its answer, and the one stream a library has no
    business claiming. They are gone, and this is what keeps them gone: a new solver
    written in the old style fails here rather than at a code review.
    """

    @staticmethod
    def _print_calls(path: Path) -> list[int]:
        """Line numbers of every `print(...)` in one module."""
        tree = ast.parse(path.read_text(), filename=str(path))
        return [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"
        ]

    def test_no_module_in_the_package_calls_print(self) -> None:
        offenders = {
            path.relative_to(SOURCE_DIR).as_posix(): lines
            for path in sorted(SOURCE_DIR.rglob("*.py"))
            if (lines := self._print_calls(path))
        }

        assert offenders == {}, f"print() is not how this package reports: {offenders}"

    def test_a_default_run_writes_nothing_to_either_stream(self, capsys: pytest.CaptureFixture[str]) -> None:
        capsys.readouterr()

        thresher.Thresher().optimize_threshold(*TINY)

        captured = capsys.readouterr()
        assert (captured.out, captured.err) == ("", "")


class TestTheLevelSelects:
    """`verbosity` names the lowest level that gets through, and means it."""

    @pytest.mark.parametrize(
        ("level", "emitted"),
        [
            ("debug", ["debug", "info", "warning"]),
            ("info", ["info", "warning"]),
            ("warning", ["warning"]),
            ("error", []),
            ("critical", []),
        ],
    )
    def test_each_level_lets_through_itself_and_everything_above_it(
        self, logs: list[str], level: str, emitted: list[str]
    ) -> None:
        with log.verbosity(level):
            log.debug("a debug line")
            log.info("an info line")
            log.warning("a warning line")

        assert [line.split(":")[0].lower() for line in logs] == emitted

    def test_the_default_is_warning(self) -> None:
        assert log.current_verbosity() == "warning"
        assert log.DEFAULT_VERBOSITY == "warning"

    def test_the_case_of_the_name_does_not_matter(self, logs: list[str]) -> None:
        with log.verbosity("INFO"):
            log.info("shouted at")

        assert len(logs) == 1

    def test_arguments_are_interpolated_only_when_the_record_is_emitted(self, logs: list[str]) -> None:
        """A message below the level costs nothing - it never reaches loguru at all.

        The genetic solver's per-generation line lists every agent's trait, and the sgd
        walk logs four lines per step. Formatting those and discarding them would be a
        real cost on a run that asked for none of it.
        """

        class Counted:
            renders = 0

            def __format__(self, spec: str) -> str:
                Counted.renders += 1
                return "rendered"

        log.debug("value: {}", Counted())
        assert Counted.renders == 0, "formatted a record nobody asked for"

        with log.verbosity("debug"):
            log.debug("value: {}", Counted())
        assert Counted.renders == 1


class TestWhoSetsIt:
    """Three ways in, in increasing precedence, each scoped differently."""

    def test_set_verbosity_lasts_until_it_is_changed(self, logs: list[str]) -> None:
        log.set_verbosity("info")

        thresher.Thresher().optimize_threshold(*TINY)

        assert logs, "the process-wide default should have applied"

    def test_an_instance_setting_beats_the_process_default(self, logs: list[str]) -> None:
        log.set_verbosity("info")

        thresher.Thresher(verbosity="error").optimize_threshold(*TINY)

        assert logs == []

    def test_the_level_is_put_back_when_the_call_returns(self, logs: list[str]) -> None:
        thresher.Thresher(verbosity="debug").optimize_threshold(*TINY)
        during = len(logs)

        log.info("after the call")

        assert during > 0
        assert len(logs) == during, "the instance's level outlived its own run"

    def test_a_block_applies_to_the_block(self, logs: list[str]) -> None:
        with log.verbosity("debug"):
            log.debug("inside")
        log.debug("outside")

        assert len(logs) == 1

    def test_two_threads_do_not_read_each_others_setting(self) -> None:
        """A ContextVar rather than a global, which is what makes this true.

        Both logging systems hold their level process-wide, so a verbose `Thresher` in one
        thread would otherwise make every other thread verbose for as long as it ran.
        """
        seen: dict[str, str] = {}
        started = threading.Barrier(2)

        def record(name: str, level: str | None) -> None:
            started.wait()
            if level is None:
                seen[name] = log.current_verbosity()
                return
            with log.verbosity(level):
                seen[name] = log.current_verbosity()

        loud = threading.Thread(target=record, args=("loud", "debug"))
        quiet = threading.Thread(target=record, args=("quiet", None))
        loud.start()
        quiet.start()
        loud.join()
        quiet.join()

        assert seen == {"loud": "debug", "quiet": "warning"}


class TestAnUnusableLevelIsRefused:
    """A mistyped level is a run the caller believes they configured and did not."""

    @pytest.mark.parametrize("given", ["verbose", "DEBUGG", "", 20, None.__class__])
    def test_it_is_refused_where_it_is_given(self, given: Any) -> None:
        with pytest.raises(ConfigurationError):
            thresher.Thresher(verbosity=given)

    def test_it_is_refused_when_the_object_is_built_not_when_it_runs(self) -> None:
        """The same rule the algorithm name and the backend follow."""
        with pytest.raises(ConfigurationError):
            thresher.Thresher(verbosity="chatty")

    def test_set_verbosity_refuses_it_too(self) -> None:
        with pytest.raises(ConfigurationError):
            log.set_verbosity("chatty")

        assert log.current_verbosity() == "warning", "a refused level must not be half-applied"

    def test_it_is_both_a_thresher_error_and_a_value_error(self) -> None:
        with pytest.raises(ConfigurationError) as excinfo:
            log.set_verbosity("chatty")

        assert isinstance(excinfo.value, ThresherError)
        assert isinstance(excinfo.value, ValueError)

    def test_the_message_lists_the_levels_that_would_have_worked(self) -> None:
        with pytest.raises(ConfigurationError) as excinfo:
            log.set_verbosity("chatty")

        message = str(excinfo.value)
        assert "chatty" in message
        for level in log.LEVELS:
            assert level in message


class TestTheOlderBooleanStillWorks:
    """`verbose=True` predates the levels and goes on meaning what it meant."""

    def test_true_means_debug(self) -> None:
        assert log.resolve_verbosity(None, verbose=True) == "debug"

    def test_false_asks_for_nothing_rather_than_for_warning(self) -> None:
        """It has to be "unset", not "warning", or it would override the other two ways.

        `Thresher()` leaves `verbose` at False, so if that meant `'warning'` then every
        instance would silently override a `set_verbosity('info')` around it.
        """
        assert log.resolve_verbosity(None, verbose=False) is None

    def test_verbosity_wins_when_both_are_given(self) -> None:
        assert log.resolve_verbosity("error", verbose=True) == "error"

    def test_both_options_are_accepted_together(self, logs: list[str]) -> None:
        thresher.Thresher(verbose=True, verbosity="error").optimize_threshold(*TINY)

        assert logs == []


class TestReachingOtherLoggingSystems:
    """Loguru is the front door. Neither of the other two ways in is closed."""

    def test_disabling_the_package_the_loguru_way_silences_it(self, logs: list[str]) -> None:
        """What loguru documents for silencing a library, and it works here.

        It works because nothing in this package writes to a console directly - every
        message goes through the global logger, which is what `disable` gates.
        """
        logger.disable("thresher")
        try:
            thresher.Thresher(verbosity="debug").optimize_threshold(*TINY)
        finally:
            logger.enable("thresher")

        assert logs == []

    def test_propagation_puts_the_records_back_into_the_standard_library(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """`logging.getLogger('thresher')` saw these until 0.8.0, and can again."""
        log.propagate_to_logging()
        try:
            with caplog.at_level("INFO", logger="thresher"):
                thresher.Thresher(verbosity="info").optimize_threshold(*TINY)
        finally:
            log.propagate_to_logging(False)

        assert "Exact sweep" in caplog.text

    def test_propagation_is_off_by_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Or an application with both configured would print everything twice."""
        with caplog.at_level("INFO", logger="thresher"):
            thresher.Thresher(verbosity="info").optimize_threshold(*TINY)

        assert caplog.text == ""

    def test_installing_it_twice_does_not_deliver_everything_twice(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        log.propagate_to_logging()
        log.propagate_to_logging()
        try:
            with caplog.at_level("INFO", logger="thresher"):
                thresher.Thresher(verbosity="info").optimize_threshold(*TINY)
        finally:
            log.propagate_to_logging(False)

        assert caplog.text.count("Chosen algorithm") == 1

    def test_removing_it_when_it_was_never_installed_is_harmless(self) -> None:
        log.propagate_to_logging(False)
        log.propagate_to_logging(False)

    def test_only_this_package_is_dragged_across(self, caplog: pytest.LogCaptureFixture) -> None:
        """The bridge is a global loguru handler, so it has to filter to us.

        Another library logging through loguru has not asked to appear in this
        application's `logging` output, and that is not this package's call to make.
        """
        log.propagate_to_logging()
        try:
            with caplog.at_level("DEBUG"):
                # Logged from this module, so loguru names the record after it - which is
                # what any other library's records look like from here.
                logger.info("not ours")
        finally:
            log.propagate_to_logging(False)

        assert "not ours" not in caplog.text


class TestTheRecordsThemselves:
    """Where a record says it came from, which is what makes filtering possible."""

    def test_a_record_is_credited_to_the_module_that_logged_it(self, logs: list[str]) -> None:
        """Not to `thresher.log`, which is where the call physically happens.

        The facade calls loguru on the caller's behalf, so without a depth correction
        every record in the package would claim to come from this one module - and
        `logging.getLogger('thresher.dispatch')` would have nothing to select.
        """
        thresher.Thresher(algorithm="hist", verbosity="info").optimize_threshold(*TINY)

        sources = {line.split(":")[1] for line in logs}

        assert "thresher.algs.histogram.compute" in sources
        assert "thresher.log" not in sources

    def test_every_record_is_named_under_the_package(self, logs: list[str]) -> None:
        thresher.Thresher(verbosity="debug").optimize_threshold(*TINY)

        assert logs
        assert all(line.split(":")[1].startswith("thresher") for line in logs)

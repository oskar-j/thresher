"""Where everything this package has to say goes.

Until 0.8.0 there were two answers to that, neither of them good. Most of it went to
`print()` behind an `if verbose:` - twenty-odd branches threading a boolean down through
the dispatcher into every solver, writing to stdout, unformattable, untimestamped, and
impossible to route anywhere. The rest went to the standard library's `logging`, from
`dispatch` and `spark` only. So the same run reported half of itself one way and half the
other, and a caller who wanted the detail had to take it on stdout - the stream the
command line reserves for the answer.

It all goes through `loguru` now, at a level per message, and one setting decides how much
of it is emitted.

## The level a caller asks for

`verbosity` names the lowest level that gets through: `'debug'`, `'info'`, `'warning'`
(the default), `'error'` or `'critical'`. It can be set three ways, in increasing
precedence:

* `set_verbosity('info')`, which lasts until it is changed;
* `Thresher(verbosity='info')`, which applies to that instance's runs;
* `with verbosity('info'):`, which applies to the block.

The default is `'warning'`, so an ordinary call still says nothing unless something is
worth saying - which preserves the one message that was always emitted, the warning that
the chosen algorithm is slow for this much data.

## Why the level is checked here rather than at a sink

Both logging systems hold their level globally: one `Thresher` cannot be verbose while
another, in the same process, is not. That is the wrong shape for a library whose
verbosity is a per-instance option, so the check happens at the call site, against a
`ContextVar`. Two consequences worth knowing: the level is per-context, so threads and
async tasks do not read each other's setting; and a message below the level costs nothing,
because it is never handed to loguru at all.

## Why no sink is installed

A library that adds a loguru handler takes over an application's console. This adds none:
records go to whatever the application has configured, which for an application that has
configured nothing is loguru's own stderr handler. `logger.disable('thresher')` silences
this package the loguru way, and works because nothing here bypasses the global logger.

Applications that route their logs through the standard library instead can have these
records too - `propagate_to_logging()` bridges them, restoring what
`logging.getLogger('thresher')` used to see. It is off by default because an application
with both configured would otherwise print everything twice.
"""

import logging
import sys
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from loguru import logger

from thresher.exceptions import UNKNOWN_VERBOSITY, ConfigurationError

#: The levels a caller may ask for, lowest first. These are loguru's own names, lowercased
#: - `set_verbosity('DEBUG')` and `set_verbosity('debug')` both work.
LEVELS: tuple[str, ...] = ("debug", "info", "warning", "error", "critical")

#: Severity of each level, matching the standard library's numbers and loguru's. Used only
#: for comparison, so the scale matters and the exact values do not.
SEVERITY: dict[str, int] = {"debug": 10, "info": 20, "warning": 30, "error": 40, "critical": 50}

#: What is emitted when nobody has said otherwise. Warnings and worse, which is what this
#: package emitted before it had a verbosity setting at all.
DEFAULT_VERBOSITY = "warning"

#: Set by `set_verbosity`, read when no context-local level is in force.
_default_verbosity: str = DEFAULT_VERBOSITY

#: The level for the current context, set by the `verbosity` context manager. A ContextVar
#: rather than a plain global so that two threads - or two async tasks - each running a
#: `Thresher` of their own do not overwrite each other's setting.
_verbosity: ContextVar[str | None] = ContextVar("thresher_verbosity", default=None)

#: The bridge into the standard library, while one is installed. See `propagate_to_logging`.
_propagating: int | None = None

#: What the command line prints: the level and the message. A person reading a terminal
#: already knows which program is talking and when they ran it, so loguru's timestamp and
#: `module:function:line` are decoration there - they belong in an application's log file.
CLI_FORMAT = "<level>{level: <8}</level> {message}"


def _validate_level(value: Any) -> str:
    """Read one level name, refusing anything that is not one.

    Args:
        value: the name given, in any case.

    Returns:
        The name, lowercased.

    Raises:
        ConfigurationError: if it is not one of `LEVELS`. It is a `ValueError`. Accepting
            a mistyped level would leave the caller believing they configured a run they
            did not - the same reason a mistyped option name is refused.
    """
    name = str(value).lower()
    if name not in SEVERITY:
        raise ConfigurationError(
            UNKNOWN_VERBOSITY.format(got=value, valid=", ".join(repr(level) for level in LEVELS))
        )
    return name


def resolve_verbosity(verbosity: Any, verbose: Any = None) -> str | None:
    """Turn the two constructor options into one level name.

    Args:
        verbosity: the `verbosity` option: a level name, or None if it was not given.
        verbose: the older `verbose` boolean. True means `'debug'` - it used to print
            every solver's per-iteration detail, which is what that level is for. False
            and None both mean "nothing asked for", so that leaving it at its default
            does not override a `verbosity` set elsewhere.

    Returns:
        The level name, or None if neither option asked for anything.

    Raises:
        ConfigurationError: if `verbosity` is not one of `LEVELS`. It is a `ValueError`.
    """
    if verbosity is not None:
        return _validate_level(verbosity)

    return "debug" if verbose else None


def set_verbosity(level: str) -> None:
    """Set how much this package reports, until it is set again.

    Args:
        level: one of `LEVELS`, case-insensitive. `'error'` is the way to silence the
            "this algorithm is slow for this much data" warning.

    Returns:
        None.

    Raises:
        ConfigurationError: if `level` is not one of `LEVELS`. It is a `ValueError`.
    """
    global _default_verbosity
    _default_verbosity = _validate_level(level)


def current_verbosity() -> str:
    """Report the level in force here.

    Returns:
        The context-local level if a `verbosity` block or a `Thresher` set one, and the
        process-wide default otherwise.
    """
    return _verbosity.get() or _default_verbosity


def is_enabled_for(level: str) -> bool:
    """Say whether a message at this level would be emitted.

    Worth asking before building a message that is expensive to format - the genetic
    solver's per-generation dump of every agent's trait, for instance.

    Args:
        level: one of `LEVELS`, case-insensitive.

    Returns:
        True if the level is at or above the one in force.
    """
    return SEVERITY[level.lower()] >= SEVERITY[current_verbosity()]


@contextmanager
def verbosity(level: str | None) -> Generator[None, None, None]:
    """Apply a level for the duration of a block, then put back what was there.

    This is what `Thresher.optimize_threshold` wraps its run in, so that an instance built
    with `verbosity='debug'` is verbose for its own calls and for nothing else.

    Args:
        level: one of `LEVELS`, or None to leave the current setting alone - which is what
            a `Thresher` built without either option passes.

    Yields:
        None, with the level in force.
    """
    if level is None:
        yield
        return

    token = _verbosity.set(level.lower())
    try:
        yield
    finally:
        _verbosity.reset(token)


def _emit(level: str, message: str, args: tuple[Any, ...]) -> None:
    """Hand one record to loguru, if the level in force lets it through.

    Args:
        level: the level to log at, one of `LEVELS`.
        message: a loguru-style format string, using `{}` placeholders.
        args: values for those placeholders. Interpolated by loguru, and only if the
            record is actually emitted.

    Returns:
        None.
    """
    if not is_enabled_for(level):
        return
    # depth=2 credits the record to this module's caller rather than to `_emit` and its
    # wrapper, so `{name}` and `{line}` point at the solver that had something to say.
    logger.opt(depth=2).log(level.upper(), message, *args)


def debug(message: str, *args: Any) -> None:
    """Log the detail of a run: one line per iteration, per generation, per step.

    Args:
        message: a loguru-style format string, using `{}` placeholders.
        *args: values for those placeholders.

    Returns:
        None.
    """
    _emit("debug", message, args)


def info(message: str, *args: Any) -> None:
    """Log what the run is doing: which algorithm, over how much data, with what result.

    Args:
        message: a loguru-style format string, using `{}` placeholders.
        *args: values for those placeholders.

    Returns:
        None.
    """
    _emit("info", message, args)


def warning(message: str, *args: Any) -> None:
    """Log something the caller should know about a run that is going ahead anyway.

    Args:
        message: a loguru-style format string, using `{}` placeholders.
        *args: values for those placeholders.

    Returns:
        None.
    """
    _emit("warning", message, args)


class PropagateHandler(logging.Handler):
    """A loguru sink that re-emits records through the standard library.

    Loguru's own recipe for the job. The record arrives here already formatted by the
    `logging` machinery loguru builds it from, so it needs only to be handed to the logger
    of the same name - `thresher.dispatch`, `thresher.algs.exact.compute`, and so on,
    which is the hierarchy `logging.getLogger('thresher')` sits above.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Pass one record to the standard library logger of the same name.

        Args:
            record: the record loguru built.

        Returns:
            None.
        """
        logging.getLogger(record.name).handle(record)


def propagate_to_logging(enable: bool = True) -> None:
    """Send this package's records into the standard library's `logging` as well.

    Before 0.8.0 the two messages this package logged went through `logging`, so
    `logging.getLogger('thresher').setLevel(logging.ERROR)` silenced them and an
    application's file handler picked them up. Both now go through loguru, and this is
    what restores that: with it on, every record is handed to the `logging` logger named
    after the module that emitted it.

    Off by default, and deliberately. An application with loguru *and* `logging` both
    writing to a console would print each record twice, and the duplicate is the harder
    problem to work out from the outside.

    Args:
        enable: True to install the bridge, False to remove it. Installing twice is a
            no-op rather than a second copy of every record.

    Returns:
        None.
    """
    global _propagating

    if not enable:
        if _propagating is not None:
            logger.remove(_propagating)
            _propagating = None
        return

    if _propagating is not None:
        return

    # Filtered to this package. Loguru's handlers are global, so an unfiltered one here
    # would also drag every other library's loguru records into `logging` - which is not
    # this package's decision to make.
    _propagating = logger.add(
        PropagateHandler(),
        format="{message}",
        filter=lambda record: record["name"] is not None and record["name"].startswith("thresher"),
        level=0,
    )


class _StandardError:
    """A stand-in for `sys.stderr` that resolves it at write time, not at add time.

    Loguru holds the object it was given, which is the wrong thing to hold when something
    swaps the stream afterwards - `click.testing` does exactly that for the duration of an
    invocation, and pytest does it for the duration of a test. A handler bound to the real
    stderr would write past both of them; a handler bound to *theirs* would outlive it and
    end up writing to a closed buffer. Resolving the attribute on every write is neither.
    """

    def write(self, message: str) -> None:
        """Write one formatted record to whatever `sys.stderr` is right now.

        Args:
            message: the record, already formatted by loguru.

        Returns:
            None.
        """
        sys.stderr.write(message)

    def flush(self) -> None:
        """Flush whatever `sys.stderr` is right now.

        Returns:
            None.
        """
        sys.stderr.flush()


@contextmanager
def cli_logging(level: str) -> Generator[None, None, None]:
    """Give the command line a console handler of its own, for one invocation.

    A library must not configure loguru; a program may, and the `thresher` command is a
    program. Loguru's default handler prints a timestamp, the module path and the line
    number, which is right for an application's log file and too much for a one-line
    answer in a terminal - so it is replaced, for the duration, with one printing the
    level and the message to stderr. stdout is left for the result.

    Scoped rather than done once at start-up, because the command is importable as
    `thresher.cli.main` - the test suite calls it that way, and so may anyone else. A
    program that returns to its caller having removed that caller's log handler is a
    program that has broken something it did not own. Only loguru's own default handler is
    touched, and an equivalent one is put back on the way out.

    It also sets the level process-wide, with `set_verbosity`. The two have to agree:
    handing a record to loguru is what the handler is for, and deciding whether to hand it
    over at all is what the level does.

    Args:
        level: the verbosity the invocation asked for, one of `LEVELS`.

    Yields:
        None, with the handler installed and the level in force.
    """
    previous = current_verbosity()
    set_verbosity(level)

    # 0 is loguru's default stderr handler. Removing it raises if an application has
    # already done so itself, which is a state to accept rather than to report.
    had_default = True
    try:
        logger.remove(0)
    except ValueError:
        had_default = False

    handler = logger.add(_StandardError(), format=CLI_FORMAT, level=level.upper(), colorize=False)
    try:
        yield
    finally:
        logger.remove(handler)
        if had_default:
            # Not the same handler - loguru numbers them from a counter that only goes up -
            # but the same behaviour: its default format, at its default level, on stderr.
            logger.add(_StandardError())
        set_verbosity(previous)

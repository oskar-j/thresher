"""Progress bars: tqdm where it is installed, the built-in bar where it is not.

`tqdm` is an optional extra - `pip install 'thresher-py[progress]'`. When it imports, it
draws; when it does not, the bar this package has always carried draws instead, and
nothing else changes. That is the whole contract, and it is worth stating the two halves
of it explicitly:

* **The answer never depends on which one drew.** A progress bar is output, not
  computation, so the two backends differ in appearance and in nothing else.
* **They are formatted to match.** tqdm is given a `bar_format` shaped like the built-in
  bar - a percentage to one decimal place, then the counts - so installing the extra does
  not change what a script scraping stderr sees, and the tests can assert one thing about
  both.

## Where a bar is drawn, and where it is not

A bar is a thing you watch. It is worth drawing when a person is waiting at a terminal,
and is noise or corruption anywhere else, so:

* **It goes to stderr**, never stdout. The command line prints the threshold on stdout so
  it can be piped onward; a bar redrawing itself with a carriage return in the middle of
  that would corrupt it. Before 0.8.0 the built-in bar wrote to stdout, which is the same
  stream - it went unnoticed only because the command line had no way to ask for a bar.
* **Nothing is drawn from a worker.** The `mp` and `ray` backends run the counting in
  other processes, where several bars would interleave into nonsense on one terminal. The
  functions those backends ship to workers do no reporting at all, which is what makes
  them safe to ship.
* **Nothing is drawn from Spark.** `SparkThresher` has no `progress_bar` option and calls
  the sweeps without one. The work there happens in the cluster, where there is nobody to
  watch it, and the driver's part is a few thousand bins - over before a bar could be
  read.
* **Not while the log is at DEBUG.** Both write to stderr, so together they produce a bar
  interrupted by log lines and log lines interrupted by a bar. The level wins: someone who
  asked for the detail asked for the detail. Until 0.8.0 the genetic solver was alone in
  applying that rule, and announced it with a printed warning.
"""

import sys
from types import TracebackType
from typing import Any, Protocol, TextIO

from thresher import log

#: tqdm's class if the extra is installed, and None if it is not. Declared `Any` because
#: the two branches below give it two types, and the type checker has no way to know which
#: machine it is running on - which is the same reason the Ray backend imports lazily.
_tqdm: Any

try:  # pragma: no cover - the branch taken depends on whether the extra is installed
    # Deliberately `tqdm.tqdm` rather than `tqdm.auto`, which substitutes a notebook
    # widget where one is available. A widget cannot honour `file=` or the `bar_format`
    # below, and the promise here is that the two backends report the same thing.
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - exercised by hiding the module, see the tests
    _tqdm = None


def tqdm_available() -> bool:
    """Say whether the optional `tqdm` extra is installed.

    Returns:
        True if bars will be drawn by tqdm, False if the built-in bar will draw them.
    """
    return _tqdm is not None


class ProgressBar(Protocol):
    """What the solvers need from a progress bar, and all they are given.

    Two methods and a context manager. Small enough that the built-in bar, tqdm and the
    do-nothing bar can each satisfy it without pretending to be one another.
    """

    def update(self, completed: int) -> None:
        """Report the total number of steps finished so far.

        Args:
            completed: steps done, counted from the start rather than since the last call.
                An absolute count, because two of the call sites bracket one batched
                computation and know only "none of it" and "all of it".

        Returns:
            None.
        """
        ...

    def close(self) -> None:
        """Finish the bar off, leaving the line tidy.

        Returns:
            None.
        """
        ...

    def __enter__(self) -> "ProgressBar":
        """Enter the block the bar covers.

        Returns:
            The bar itself.
        """
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the bar on the way out, whether or not the block succeeded.

        Args:
            exc_type: the exception class, if the block raised.
            exc: the exception, if the block raised.
            traceback: its traceback, if the block raised.

        Returns:
            None. Exceptions are not suppressed.
        """
        ...


class _NoProgress:
    """The bar drawn when nobody asked for one, or when nobody could watch it.

    A real object rather than a `None` the call sites test for, so that
    `with make_progress(...) as bar: bar.update(n)` reads the same either way.
    """

    def update(self, completed: int) -> None:
        """Do nothing with a step count.

        Args:
            completed: ignored.

        Returns:
            None.
        """

    def close(self) -> None:
        """Do nothing.

        Returns:
            None.
        """

    def __enter__(self) -> "_NoProgress":
        """Enter the block.

        Returns:
            This same object.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Leave the block.

        Args:
            exc_type: ignored.
            exc: ignored.
            traceback: ignored.

        Returns:
            None.
        """


class _BuiltInBar:
    """The bar this package has always drawn, wrapped in the shared interface.

    `print_progress_bar` renders one frame; this holds the total and the description
    between frames, and makes sure the line is ended exactly once.
    """

    def __init__(self, total: int, description: str, stream: TextIO) -> None:
        """Set up a bar.

        Args:
            total: how many steps the whole job is.
            description: what the job is, printed before the bar.
            stream: where to draw.
        """
        self.total = total
        self.description = description
        self.stream = stream
        self.closed = False

    def update(self, completed: int) -> None:
        """Redraw the bar at `completed` of `total` steps.

        Args:
            completed: steps done in total, not since the last call.

        Returns:
            None.
        """
        if self.closed:
            return
        # The counts in the suffix are what tqdm's `bar_format` puts there, so the two
        # backends report the same things in the same order.
        print_progress_bar(
            completed,
            self.total,
            prefix=self.description,
            suffix=f"[{completed}/{self.total}]",
            stream=self.stream,
        )
        # `print_progress_bar` ends the line when it draws a full bar, so reaching the end
        # this way is a close - anything after it would draw on the line below.
        self.closed = completed >= self.total

    def close(self) -> None:
        """Draw the finished bar, unless it has already been drawn.

        Returns:
            None.
        """
        if not self.closed:
            self.update(self.total)

    def __enter__(self) -> "_BuiltInBar":
        """Enter the block the bar covers.

        Returns:
            This same object.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the bar on the way out.

        Args:
            exc_type: the exception class, if the block raised.
            exc: the exception, if the block raised.
            traceback: its traceback, if the block raised.

        Returns:
            None. Exceptions are not suppressed.
        """
        self.close()


class _TqdmBar:
    """tqdm, adapted from "steps since last time" to "steps so far".

    tqdm's `update` takes an increment. The call sites here have an absolute count -
    `sweep_bins` knows it is on bin 500 of 1,024, and the batched call sites know only
    "none" and "all" - so the difference is worked out here rather than at each of them.
    """

    def __init__(self, total: int, description: str, stream: TextIO) -> None:
        """Set up a tqdm bar.

        Args:
            total: how many steps the whole job is.
            description: what the job is, printed before the bar.
            stream: where to draw.
        """
        self.completed = 0
        # Shaped like the built-in bar rather than left at tqdm's default, so that
        # installing the extra changes how the bar looks and not what it reports.
        self.bar: Any = _tqdm(
            total=total,
            desc=description,
            file=stream,
            leave=True,
            bar_format="{desc} |{bar}| {percentage:.1f}% [{n_fmt}/{total_fmt}]",
        )

    def update(self, completed: int) -> None:
        """Advance the bar to `completed` steps.

        Args:
            completed: steps done in total, not since the last call. Going backwards is
                ignored rather than rewinding the bar.

        Returns:
            None.
        """
        step = completed - self.completed
        if step > 0:
            self.completed = completed
            self.bar.update(step)

    def close(self) -> None:
        """Fill the bar to its total and close it.

        Returns:
            None.
        """
        self.update(self.bar.total)
        self.bar.close()

    def __enter__(self) -> "_TqdmBar":
        """Enter the block the bar covers.

        Returns:
            This same object.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the bar on the way out.

        Args:
            exc_type: the exception class, if the block raised.
            exc: the exception, if the block raised.
            traceback: its traceback, if the block raised.

        Returns:
            None. Exceptions are not suppressed.
        """
        self.close()


def make_progress(
    total: int, description: str = "", *, enabled: bool = False, stream: TextIO | None = None
) -> ProgressBar:
    """Build the progress bar for one job, or a bar that draws nothing.

    Args:
        total: how many steps the whole job is. A total of zero or less has no proportion
            to show, so nothing is drawn.
        description: what the job is, printed before the bar.
        enabled: whether the caller asked for a bar at all - the `progress_bar` option.
        stream: where to draw. Defaults to `sys.stderr`, read now rather than held from
            import, so that a redirected stream is honoured.

    Returns:
        A tqdm-backed bar, the built-in bar, or a bar that does nothing - see the module
        docstring for which and why. All three satisfy `ProgressBar`, so the caller does
        not branch.
    """
    # DEBUG puts a log line on stderr for every step of the very loops that draw bars, so
    # the two would overwrite each other. The level wins.
    if not enabled or total <= 0 or log.is_enabled_for("debug"):
        return _NoProgress()

    target = stream if stream is not None else sys.stderr

    if _tqdm is not None:
        return _TqdmBar(total, description, target)
    return _BuiltInBar(total, description, target)


def print_progress_bar(
    iteration: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    length: int = 100,
    fill: str = "#",
    stream: TextIO | None = None,
) -> None:
    """Draw one frame of the built-in terminal progress bar, in place.

    The fallback for when `tqdm` is not installed, and the bar this package drew for
    everyone before 0.8.0. Call it in a loop; each call overwrites the last.

    Args:
        iteration: current iteration.
        total: total number of iterations. The line is ended once the two match.
        prefix: string printed before the bar.
        suffix: string printed after the bar.
        decimals: number of decimals in the percentage.
        length: character length of the bar.
        fill: bar fill character.
        stream: where to draw. Defaults to `sys.stderr`. It was stdout until 0.8.0, which
            is the stream the command line prints the threshold on - see the module
            docstring.

    Returns:
        None. The bar is written to `stream`.
    """
    target = stream if stream is not None else sys.stderr
    percent = f"{100 * (iteration / float(total)):.{decimals}f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    # Written rather than printed, and to the stream it was handed. A bar is the one thing
    # here that goes to a console directly rather than through the log: it is a picture of
    # how far along a run is, not a message about it, so it has no level and belongs in no
    # log file.
    target.write(f"\r{prefix} |{bar}| {percent}% {suffix}\r")
    # End the line once the job is done, so whatever writes next starts on a fresh one.
    if iteration == total:
        target.write("\n")
    target.flush()

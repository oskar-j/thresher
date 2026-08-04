"""The multiprocessing backend: the same counting, spread over local CPU cores.

Shaped exactly like the Ray backend, and for the same reason - the data is sharded once,
each worker counts its own shard with the plain functions from `base`, and the driver adds
the partial results together. Addition is order-independent, so the answer is identical to
a local run whatever order the shards finish in.

Where Ray spreads work over a cluster, this spreads it over the machine already running.
It needs nothing installed: `multiprocessing` is in the standard library, so unlike `ray`
this backend is available everywhere, including the macOS x86_64 machines Ray has no wheel
for.

`concurrent.futures.ProcessPoolExecutor` rather than `multiprocessing.Pool`, deliberately.
On any start method that re-imports `__main__` - `spawn` on macOS and Windows, `forkserver`
elsewhere on newer Pythons - a caller who builds a pool at module level makes every worker
re-run their own script, which builds another pool. `Pool.map` waits for workers that will
never report, so the process hangs with no error at all; the executor raises
`BrokenProcessPool` instead, which this module turns into an explanation of what to do
about it.

Small inputs are counted in-process. Sharding costs a fork, a pickle and a round trip per
shard, which is more than the counting is worth until there is real data to divide.
"""

import multiprocessing
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, TypeVar

from thresher.backends.base import (
    ClassCounts,
    count_chunk,
    merge_counts,
    merge_tallies,
    plan_shards,
    tally_chunk,
)
from thresher.exceptions import PARALLEL_BOOTSTRAP_FAILED, ConfigurationError, ParallelBootstrapError

#: A shard smaller than this costs more to ship than to compute, so shards are widened
#: rather than multiplied below it. Lower than Ray's, because handing work to a local
#: process is cheaper than scheduling it onto a cluster.
DEFAULT_MIN_ROWS_PER_SHARD = 2_000

#: What one shard's map step returns - a tally list or a counts mapping.
ResultT = TypeVar("ResultT")

#: How a pool that never came up reports itself, which differs by start method. `spawn`
#: reaches the executor's own `BrokenProcessPool`; `forkserver` can instead fail while the
#: driver is still talking to the fork server, which surfaces as a dead connection -
#: `ConnectionResetError` on Linux, and `BrokenPipeError` or `EOFError` depending on where
#: in the handshake it died. All of them mean the same thing: no worker ever ran, so the
#: cause is the environment rather than the data, and the caller needs the same advice.
#: The map steps here are pure arithmetic over lists and open nothing, so none of these can
#: reach this from inside a worker's own work.
BOOTSTRAP_FAILURES = (BrokenProcessPool, ConnectionResetError, BrokenPipeError, EOFError)

INVALID_WORKERS = (
    "num_workers must be -1, meaning every processor bar one, or at least 1 - got {got}. "
    "The 'n_jobs' algorithm parameter of linear search means the same thing."
)


def resolve_worker_count(num_workers: int | None) -> int:
    """Turn a requested worker count into a usable number of processes.

    Shared with linear search's `n_jobs`, which names the same quantity, so the two cannot
    disagree about what `-1` means or about which values are refusable.

    Args:
        num_workers: how many processes to ask for. `None` means one per processor, `-1`
            every processor bar one - the historical meaning of `n_jobs=-1`.

    Returns:
        A process count of at least 1, never more than the machine has processors.
        Over-asking is clamped rather than refused: how many cores are available is a
        property of the machine, not a mistake in the caller's code.

    Raises:
        ConfigurationError: for 0, or anything below -1, which name no sensible number of
            processes. It is a `ValueError`.
    """
    available = multiprocessing.cpu_count()

    if num_workers is None:
        return available
    if (
        not isinstance(num_workers, int)
        or isinstance(num_workers, bool)
        or num_workers == 0
        or num_workers < -1
    ):
        raise ConfigurationError(INVALID_WORKERS.format(got=num_workers))
    if num_workers == -1:
        # Leave a processor for the caller's own machine, as this has always meant.
        return max(1, available - 1)
    return min(num_workers, available)


class MultiprocessingBackend:
    """Count in parallel across this machine's CPU cores.

    Example:
        >>> from thresher import Thresher
        >>> Thresher(backend="mp").optimize_threshold(scores, actual_classes)  # doctest: +SKIP

    Note:
        Because the workers are separate processes, any script that builds one of these at
        module level must sit behind an `if __name__ == "__main__":` guard. Without it the
        workers re-import the script and build their own pools; see the module docstring.
        That mistake is reported rather than left to hang.
    """

    name = "mp"

    def __init__(
        self, num_workers: int | None = None, min_rows_per_shard: int = DEFAULT_MIN_ROWS_PER_SHARD
    ) -> None:
        """Configure how the data is divided, and over how many processes.

        Args:
            num_workers: how many worker processes to use. Defaults to one per processor;
                `-1` means every processor bar one.
            min_rows_per_shard: do not produce shards smaller than this. Below it the work
                is done in this process instead, since a fork would cost more than it saves.

        Raises:
            ConfigurationError: if `num_workers` is 0 or below -1. Checked here rather than
                on first use, so the failure lands when the backend is asked for. It is a
                `ValueError`.
        """
        self._num_workers = resolve_worker_count(num_workers)
        self._min_rows_per_shard = min_rows_per_shard

    def _shards(self, total: int) -> list[tuple[int, int]]:
        """Plan the shard boundaries for this machine.

        Args:
            total: number of samples.

        Returns:
            `(start, stop)` index pairs covering the data exactly once.
        """
        return plan_shards(total, self._num_workers, self._min_rows_per_shard)

    def tally_candidates(
        self,
        candidates: Sequence[float],
        scores: Sequence[float],
        actual_classes: Sequence[int],
    ) -> list[int]:
        """Count correct predictions per candidate, sharded across processes.

        Args:
            candidates: the thresholds to score.
            scores: the values being split.
            actual_classes: the matching classes, as -1 and 1.

        Returns:
            One count per candidate, identical to what the local backend returns.

        Raises:
            ParallelBootstrapError: if the worker processes could not start, which on a
                re-importing start method means a missing `__main__` guard.
        """
        boundaries = self._shards(len(scores))
        if not candidates:
            return [0] * len(candidates)
        if len(boundaries) <= 1:
            # One shard is just the local backend with extra steps.
            return tally_chunk(candidates, scores, actual_classes)

        shared = list(candidates)
        return merge_tallies(
            self._map(
                tally_chunk,
                [
                    (shared, list(scores[start:stop]), list(actual_classes[start:stop]))
                    for start, stop in boundaries
                ],
            )
        )

    def class_counts_by_score(
        self, scores: Sequence[float], actual_classes: Sequence[int]
    ) -> dict[float, ClassCounts]:
        """Count classes per distinct score, sharded across processes.

        This is the step that makes the exact sweep parallel: each worker returns one count
        per distinct score it saw, never the samples.

        Args:
            scores: the values being split.
            actual_classes: the matching classes, as -1 and 1.

        Returns:
            A mapping of score to `(negatives, positives)`, identical to the local result.

        Raises:
            ParallelBootstrapError: if the worker processes could not start.
        """
        boundaries = self._shards(len(scores))
        if not boundaries:
            return {}
        if len(boundaries) == 1:
            return count_chunk(scores, actual_classes)

        return merge_counts(
            self._map(
                count_chunk,
                [(list(scores[start:stop]), list(actual_classes[start:stop])) for start, stop in boundaries],
            )
        )

    def _map(self, function: Callable[..., ResultT], arguments: list[tuple[Any, ...]]) -> list[ResultT]:
        """Run one call per shard across the pool, and collect the results in order.

        Args:
            function: the map step, a module-level function so it pickles by reference.
            arguments: the positional arguments for each shard's call.

        Returns:
            One result per shard, in shard order. The reduce steps do not depend on the
            order, but keeping it makes a failure easier to attribute.

        Raises:
            ParallelBootstrapError: if the workers died before doing any work. It is a
                `RuntimeError`, and the message names the usual cause.
        """
        try:
            with ProcessPoolExecutor(max_workers=self._num_workers) as pool:
                futures: list[Future[ResultT]] = [pool.submit(function, *call) for call in arguments]
                return [future.result() for future in futures]
        except BOOTSTRAP_FAILURES as exc:
            raise ParallelBootstrapError(PARALLEL_BOOTSTRAP_FAILED) from exc

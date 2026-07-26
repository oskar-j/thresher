"""The Ray backend: the same counting, spread over a cluster.

The shape is map-reduce. The data is sharded once and placed in Ray's object store, each
worker counts its own shard, and the driver adds the partial results together. Both map
steps are the plain functions in `base`, shipped to the workers unchanged, so a Ray run
and a local run execute identical counting code and must agree exactly.

Ray is an optional dependency: `pip install thresher-py[ray]`. Note that Ray publishes no
macOS x86_64 wheel, so it cannot be installed on an Intel Mac.
"""

from collections.abc import Sequence
from typing import Any

from thresher.backends.base import (
    ClassCounts,
    count_chunk,
    merge_counts,
    merge_tallies,
    plan_shards,
    tally_chunk,
)
from thresher.exceptions import BackendDependencyError

# A shard smaller than this costs more to schedule than to compute, so shards are widened
# rather than multiplied below it.
DEFAULT_MIN_ROWS_PER_SHARD = 5_000

RAY_MISSING = (
    "The 'ray' backend needs Ray installed: pip install 'thresher-py[ray]'. "
    "Note that Ray publishes no macOS x86_64 wheel, so it cannot be installed on an "
    "Intel Mac; use backend='local' there."
)


def _require_ray() -> Any:
    """Import Ray, or explain how to get it.

    Returns:
        The imported `ray` module.

    Raises:
        BackendDependencyError: if Ray is not installed, carrying the install
            instructions. It is an `ImportError`.
    """
    try:
        import ray
    except ImportError as exc:  # pragma: no cover - exercised only without Ray installed
        raise BackendDependencyError(RAY_MISSING) from exc
    return ray


class RayBackend:
    """Count in parallel across a Ray cluster.

    Connects to whatever cluster Ray is already attached to. If Ray has not been
    initialised, it is started locally with default settings - so a caller who has already
    called `ray.init(address=...)` keeps their cluster, and one who has not gets a working
    local cluster without ceremony.
    """

    name = "ray"

    def __init__(
        self, num_shards: int | None = None, min_rows_per_shard: int = DEFAULT_MIN_ROWS_PER_SHARD
    ) -> None:
        """Configure how the data is divided.

        Args:
            num_shards: how many shards to split into. Defaults to the cluster's CPU
                count, which is the useful maximum since each shard occupies one worker.
            min_rows_per_shard: do not produce shards smaller than this. Sharding a small
                dataset costs more in scheduling than it saves in computation.

        Raises:
            BackendDependencyError: if Ray is not installed, an `ImportError`. Checked
                here rather than on first use,
                so the failure lands when the backend is asked for instead of partway
                through a long run.
        """
        _require_ray()
        self._num_shards = num_shards
        self._min_rows_per_shard = min_rows_per_shard

    def _ray(self) -> Any:
        """Return an initialised Ray module, starting a local cluster if needed."""
        ray = _require_ray()
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        return ray

    def _shards(self, ray: Any, total: int) -> list[tuple[int, int]]:
        """Plan the shard boundaries for this cluster.

        Args:
            ray: the initialised Ray module.
            total: number of samples.

        Returns:
            `(start, stop)` index pairs covering the data exactly once.
        """
        if self._num_shards is not None:
            workers = self._num_shards
        else:
            workers = int(ray.cluster_resources().get("CPU", 1)) or 1
        return plan_shards(total, workers, self._min_rows_per_shard)

    def tally_candidates(
        self,
        candidates: Sequence[float],
        scores: Sequence[float],
        actual_classes: Sequence[int],
    ) -> list[int]:
        """Count correct predictions per candidate, sharded across the cluster.

        The candidate list is put into the object store once and shared by reference, so
        it is not re-serialised per shard.

        Args:
            candidates: the thresholds to score.
            scores: the values being split.
            actual_classes: the matching classes, as -1 and 1.

        Returns:
            One count per candidate, identical to what the local backend returns.
        """
        ray = self._ray()
        boundaries = self._shards(ray, len(scores))
        if not boundaries or not candidates:
            return [0] * len(candidates)

        remote_tally = ray.remote(tally_chunk)
        candidate_ref = ray.put(list(candidates))

        futures = [
            remote_tally.remote(candidate_ref, list(scores[start:stop]), list(actual_classes[start:stop]))
            for start, stop in boundaries
        ]
        return merge_tallies(ray.get(futures))

    def class_counts_by_score(
        self, scores: Sequence[float], actual_classes: Sequence[int]
    ) -> dict[float, ClassCounts]:
        """Count classes per distinct score, sharded across the cluster.

        This is the step that makes the exact sweep distributable: the driver never sees
        the samples, only one count per distinct score.

        Args:
            scores: the values being split.
            actual_classes: the matching classes, as -1 and 1.

        Returns:
            A mapping of score to `(negatives, positives)`, identical to the local result.
        """
        ray = self._ray()
        boundaries = self._shards(ray, len(scores))
        if not boundaries:
            return {}

        remote_count = ray.remote(count_chunk)

        futures = [
            remote_count.remote(list(scores[start:stop]), list(actual_classes[start:stop]))
            for start, stop in boundaries
        ]
        return merge_counts(ray.get(futures))

"""Execution backends.

Split deliberately in two. The map and reduce steps are plain functions and are tested
everywhere, including on machines where Ray cannot be installed - Ray publishes no macOS
x86_64 wheel. The tests that need a live cluster skip themselves when Ray is missing, and
run for real in CI, which is Linux.

The property that matters throughout: a backend changes where the counting happens, never
the answer.
"""

import multiprocessing
import random
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

import thresher
from thresher.backends import (
    AVAILABLE_BACKENDS,
    LocalBackend,
    MultiprocessingBackend,
    get_backend,
)
from thresher.backends.base import (
    count_chunk,
    merge_counts,
    merge_tallies,
    plan_shards,
    tally_chunk,
)
from thresher.backends.mp_backend import resolve_worker_count

Dataset = tuple[list[float], list[int]]
DatasetFactory = Callable[..., Dataset]

# Only these three can be distributed without changing what they compute.
DISTRIBUTABLE = ["exact", "ls", "grid"]


class TestShardPlanning:
    """`plan_shards` decides how the data is cut up, and is pure arithmetic."""

    @pytest.mark.parametrize("total", [1, 2, 7, 100, 10_000, 999_983])
    @pytest.mark.parametrize("workers", [1, 2, 3, 8, 64])
    def test_shards_cover_the_data_exactly_once(self, total: int, workers: int) -> None:
        shards = plan_shards(total, workers, min_rows=1)

        covered = [index for start, stop in shards for index in range(start, stop)]
        assert covered == list(range(total)), "shards must tile the range with no gaps or overlap"

    def test_no_empty_shards(self) -> None:
        assert all(start < stop for start, stop in plan_shards(5, 64, min_rows=1))

    def test_sizes_differ_by_at_most_one(self) -> None:
        sizes = [stop - start for start, stop in plan_shards(100, 7, min_rows=1)]
        assert max(sizes) - min(sizes) <= 1

    def test_small_inputs_are_not_split_pointlessly(self) -> None:
        # Scheduling a shard costs more than counting a handful of rows.
        assert len(plan_shards(1_000, workers=32, min_rows=5_000)) == 1

    def test_large_inputs_use_the_workers(self) -> None:
        assert len(plan_shards(1_000_000, workers=8, min_rows=5_000)) == 8

    def test_empty_input(self) -> None:
        assert plan_shards(0, 4, min_rows=1) == []


class TestMapReduceSteps:
    """The map and reduce steps, which both backends share verbatim."""

    def test_tally_chunk_counts_correct_predictions(self) -> None:
        scores, actual_classes = [0.1, 0.4, 0.6, 0.9], [-1, -1, 1, 1]

        assert tally_chunk([0.5], scores, actual_classes) == [4]
        assert tally_chunk([0.0], scores, actual_classes) == [2]
        assert tally_chunk([0.5, 0.0], scores, actual_classes) == [4, 2]

    def test_sharding_then_merging_equals_counting_it_all_at_once(self) -> None:
        rng = random.Random(0)
        scores = [rng.random() for _ in range(500)]
        actual_classes = [rng.choice([-1, 1]) for _ in range(500)]
        candidates = [i / 20 for i in range(21)]

        whole = tally_chunk(candidates, scores, actual_classes)
        sharded = merge_tallies(
            tally_chunk(candidates, scores[start:stop], actual_classes[start:stop])
            for start, stop in plan_shards(500, workers=7, min_rows=1)
        )

        assert sharded == whole

    def test_merge_tallies_is_order_independent(self) -> None:
        partials = [[1, 2, 3], [10, 20, 30], [100, 200, 300]]

        assert merge_tallies(partials) == merge_tallies(reversed(partials))

    def test_merge_tallies_rejects_ragged_input(self) -> None:
        with pytest.raises(ValueError, match="disagree on length"):
            merge_tallies([[1, 2], [1, 2, 3]])

    def test_merge_tallies_rejects_nothing_to_merge(self) -> None:
        with pytest.raises(ValueError, match="no shard tallies"):
            merge_tallies([])

    def test_count_chunk_groups_by_score(self) -> None:
        counts = count_chunk([0.5, 0.5, 0.9], [-1, 1, 1])

        assert counts == {0.5: (1, 1), 0.9: (0, 1)}

    def test_counting_shards_then_merging_equals_counting_it_all(self) -> None:
        rng = random.Random(1)
        # Low precision, so scores repeat across shard boundaries.
        scores = [round(rng.random(), 2) for _ in range(500)]
        actual_classes = [rng.choice([-1, 1]) for _ in range(500)]

        whole = count_chunk(scores, actual_classes)
        sharded = merge_counts(
            count_chunk(scores[start:stop], actual_classes[start:stop])
            for start, stop in plan_shards(500, workers=7, min_rows=1)
        )

        assert sharded == whole


class TestBackendResolution:
    def test_local_is_the_default(self) -> None:
        assert thresher.Thresher().get_current_options()["backend"] == "local"

    def test_named_lookup(self) -> None:
        assert get_backend("local").name == "local"
        assert get_backend("LOCAL").name == "local"

    def test_an_instance_passes_straight_through(self) -> None:
        backend = LocalBackend()
        assert get_backend(backend) is backend

    def test_unknown_backend_lists_the_valid_ones(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            get_backend("does-not-exist")

        for name in AVAILABLE_BACKENDS:
            assert name in str(excinfo.value)

    def test_a_bad_name_fails_when_the_object_is_built(self) -> None:
        # Not several seconds into a long run.
        with pytest.raises(ValueError):
            thresher.Thresher(backend="does-not-exist")


@pytest.fixture(scope="module")
def ray_cluster() -> object:
    """A small local Ray cluster, started once for this module.

    Module scope rather than class scope because a class-scoped fixture has to be a
    classmethod under pytest 10, and starting Ray twice would be wasteful anyway.
    """
    ray = pytest.importorskip("ray", reason="Ray is not installed on this platform")
    if not ray.is_initialized():
        ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False, ignore_reinit_error=True)
    return ray


class TestRay:
    """Needs a live cluster. Skipped where Ray cannot be installed; runs in CI."""

    @pytest.mark.parametrize("algorithm_name", DISTRIBUTABLE)
    def test_ray_gives_the_same_answer_as_local(
        self, ray_cluster: object, separable: DatasetFactory, algorithm_name: str
    ) -> None:
        """The property the whole design rests on: identical results, not merely close."""
        scores, actual_classes = separable(4000, seed=2)

        local = thresher.Thresher(algorithm=algorithm_name, backend="local").optimize_threshold(
            scores, actual_classes
        )
        distributed = thresher.Thresher(algorithm=algorithm_name, backend="ray").optimize_threshold(
            scores, actual_classes
        )

        assert distributed == local

    @pytest.mark.parametrize("algorithm_name", DISTRIBUTABLE)
    def test_same_answer_on_awkward_data(self, ray_cluster: object, algorithm_name: str) -> None:
        """Duplicates and ties must land identically however the data was cut up."""
        rng = random.Random(3)
        scores = [round(rng.random(), 2) for _ in range(3000)]
        actual_classes = [rng.choice([-1, 1]) for _ in range(3000)]

        local = thresher.Thresher(algorithm=algorithm_name, backend="local").optimize_threshold(
            scores, actual_classes
        )
        distributed = thresher.Thresher(algorithm=algorithm_name, backend="ray").optimize_threshold(
            scores, actual_classes
        )

        assert distributed == local

    def test_ray_handles_the_below_minimum_edge_split(self, ray_cluster: object) -> None:
        scores, actual_classes = [0.1, 0.2, 0.3], [1, 1, -1]

        distributed = thresher.Thresher(algorithm="exact", backend="ray").optimize_threshold(
            scores, actual_classes
        )

        assert distributed < min(scores)

    def test_stochastic_algorithms_still_run_under_ray(self, ray_cluster: object) -> None:
        """They are not distributed, but asking for Ray must not break them."""
        scores, actual_classes = [0.1, 0.15, 0.2, 0.4, 0.7, 0.8], [-1, -1, -1, 1, 1, 1]

        for algorithm_name in ("sgrid", "gen", "sgd"):
            result = thresher.Thresher(algorithm=algorithm_name, backend="ray").optimize_threshold(
                scores, actual_classes
            )
            assert isinstance(result, float)

    def test_explicit_shard_count(self, ray_cluster: object, separable: DatasetFactory) -> None:
        from thresher.backends.ray_backend import RayBackend

        scores, actual_classes = separable(2000, seed=4)
        expected = thresher.Thresher(algorithm="exact").optimize_threshold(scores, actual_classes)

        # A backend instance, rather than a name, is how sharding gets configured.
        result = thresher.Thresher(
            algorithm="exact", backend=RayBackend(num_shards=4, min_rows_per_shard=1)
        ).optimize_threshold(scores, actual_classes)

        assert result == expected


class TestMultiprocessing:
    """The `mp` backend, added in 0.7.0.

    Unlike Ray this needs nothing installed, so these run everywhere - including the
    macOS x86_64 machines Ray has no wheel for, which is where the Ray class above skips.
    A small `min_rows_per_shard` is used throughout: the default deliberately keeps small
    inputs in-process, which would leave the sharding untested.
    """

    @pytest.mark.parametrize("algorithm_name", DISTRIBUTABLE)
    def test_mp_gives_the_same_answer_as_local(self, separable: DatasetFactory, algorithm_name: str) -> None:
        """The property the whole design rests on: identical results, not merely close."""
        scores, actual_classes = separable(4000, seed=2)

        local = thresher.Thresher(algorithm=algorithm_name, backend="local").optimize_threshold(
            scores, actual_classes
        )
        parallel = thresher.Thresher(
            algorithm=algorithm_name, backend=MultiprocessingBackend(num_workers=3, min_rows_per_shard=1)
        ).optimize_threshold(scores, actual_classes)

        assert parallel == local

    @pytest.mark.parametrize("algorithm_name", DISTRIBUTABLE)
    def test_same_answer_on_awkward_data(self, algorithm_name: str) -> None:
        """Duplicates and ties must land identically however the data was cut up."""
        rng = random.Random(3)
        scores = [round(rng.random(), 2) for _ in range(2000)]
        actual_classes = [rng.choice([-1, 1]) for _ in range(2000)]

        local = thresher.Thresher(algorithm=algorithm_name, backend="local").optimize_threshold(
            scores, actual_classes
        )
        parallel = thresher.Thresher(
            algorithm=algorithm_name, backend=MultiprocessingBackend(num_workers=3, min_rows_per_shard=1)
        ).optimize_threshold(scores, actual_classes)

        assert parallel == local

    @pytest.mark.parametrize("workers", [1, 2, 3, 7])
    def test_the_worker_count_cannot_change_the_answer(self, separable: DatasetFactory, workers: int) -> None:
        scores, actual_classes = separable(2000, seed=4)
        expected = thresher.Thresher(algorithm="exact").optimize_threshold(scores, actual_classes)

        result = thresher.Thresher(
            algorithm="exact",
            backend=MultiprocessingBackend(num_workers=workers, min_rows_per_shard=1),
        ).optimize_threshold(scores, actual_classes)

        assert result == expected

    def test_mp_handles_the_below_minimum_edge_split(self) -> None:
        scores, actual_classes = [0.1, 0.2, 0.3], [1, 1, -1]

        result = thresher.Thresher(algorithm="exact", backend="mp").optimize_threshold(scores, actual_classes)

        assert result < min(scores)

    def test_stochastic_algorithms_still_run_under_mp(self) -> None:
        """They are not distributed, but asking for mp must not break them."""
        scores, actual_classes = [0.1, 0.15, 0.2, 0.4, 0.7, 0.8], [-1, -1, -1, 1, 1, 1]

        for algorithm_name in ("sgrid", "gen", "sgd"):
            result = thresher.Thresher(algorithm=algorithm_name, backend="mp").optimize_threshold(
                scores, actual_classes
            )
            assert isinstance(result, float)

    def test_small_inputs_stay_in_this_process(self) -> None:
        """Below `min_rows_per_shard` a fork costs more than the counting it saves.

        Asserted through the shard plan rather than by timing, which would be flaky.
        """
        backend = MultiprocessingBackend(num_workers=4)

        assert len(backend._shards(100)) == 1
        assert len(backend._shards(1_000_000)) > 1

    def test_it_is_registered_under_its_name(self) -> None:
        assert "mp" in AVAILABLE_BACKENDS
        assert get_backend("mp").name == "mp"


class TestWorkerCounts:
    """`n_jobs` and `num_workers` name the same thing and resolve through one function."""

    def test_none_means_one_per_processor(self) -> None:
        assert resolve_worker_count(None) == multiprocessing.cpu_count()

    def test_minus_one_leaves_a_processor_free(self) -> None:
        assert resolve_worker_count(-1) == max(1, multiprocessing.cpu_count() - 1)

    def test_over_asking_is_clamped_rather_than_refused(self) -> None:
        """How many cores exist is a property of the machine, not a mistake in the code.

        `n_jobs=9999` used to open 9,999 processes.
        """
        assert resolve_worker_count(9999) == multiprocessing.cpu_count()

    @pytest.mark.parametrize("value", [0, -2, -100])
    def test_meaningless_counts_are_refused(self, value: int) -> None:
        """These used to print a message and carry on with one process regardless."""
        with pytest.raises(ValueError, match="num_workers"):
            resolve_worker_count(value)

    @pytest.mark.parametrize("value", [0, -2])
    def test_a_bad_n_jobs_is_refused_through_the_public_api(self, value: int) -> None:
        with pytest.raises(ValueError):
            thresher.Thresher(algorithm="ls", algorithm_params={"n_jobs": value}).optimize_threshold(
                [0.1, 0.4, 0.6, 0.9], [-1, -1, 1, 1]
            )

    def test_n_jobs_agrees_with_the_sequential_search(self, separable: DatasetFactory) -> None:
        """Parallelising must not change the answer.

        Before 0.7.0 the parallel path scored the raw scores while the sequential path
        scored the midpoints between them, so the two disagreed on the same data.
        """
        scores, actual_classes = separable(600, seed=6)

        sequential = thresher.Thresher(algorithm="ls").optimize_threshold(scores, actual_classes)
        parallel = thresher.Thresher(algorithm="ls", algorithm_params={"n_jobs": 2}).optimize_threshold(
            scores, actual_classes
        )

        assert parallel == sequential

    def test_an_explicit_backend_beats_n_jobs(self, separable: DatasetFactory) -> None:
        """Documented precedence: the two would otherwise contend for the same cores."""
        scores, actual_classes = separable(600, seed=7)

        expected = thresher.Thresher(algorithm="ls").optimize_threshold(scores, actual_classes)
        result = thresher.Thresher(
            algorithm="ls",
            backend=MultiprocessingBackend(num_workers=2, min_rows_per_shard=1),
            algorithm_params={"n_jobs": 4},
        ).optimize_threshold(scores, actual_classes)

        assert result == expected


class TestMissingMainGuard:
    """A script that parallelises at module level, fixed in 0.7.0 (#22).

    On a start method that re-imports `__main__` - spawn on macOS and Windows - the
    workers re-run the script and start workers of their own. `multiprocessing.Pool` then
    waited on children that would never report, so the process hung with no error at all;
    the run had to be killed. It now fails, quickly, saying what to do.

    Necessarily a subprocess test: the failure only exists for a script whose module-level
    code starts a pool, which is not something pytest's own `__main__` can reproduce.
    """

    SCRIPT = """
import thresher
scores = [0.1, 0.4, 0.6, 0.9] * 2000
labels = [-1, -1, 1, 1] * 2000
t = thresher.Thresher(algorithm='ls', algorithm_params={'n_jobs': 3})
print(t.optimize_threshold(scores, labels))
"""

    GUARDED_SCRIPT = """
import thresher

def main():
    scores = [0.1, 0.4, 0.6, 0.9] * 2000
    labels = [-1, -1, 1, 1] * 2000
    t = thresher.Thresher(algorithm='ls', algorithm_params={'n_jobs': 3})
    print(t.optimize_threshold(scores, labels))

if __name__ == "__main__":
    main()
"""

    def run_script(self, tmp_path: Path, source: str) -> "subprocess.CompletedProcess[str]":
        script = tmp_path / "run_me.py"
        script.write_text(source)
        # Generous but finite: the point is that this used to never return at all.
        return subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, timeout=120, check=False
        )

    @pytest.mark.skipif(
        multiprocessing.get_start_method() == "fork",
        reason="fork does not re-import __main__, so there is nothing to guard against",
    )
    def test_it_fails_with_an_explanation_rather_than_hanging(self, tmp_path: Path) -> None:
        """Terminating is half of it; saying what to do is the half that helps.

        How a pool that never came up reports itself depends on the start method - `spawn`
        raises `BrokenProcessPool`, `forkserver` can die at the connection instead - so
        this asserts on the translated error rather than on any one of those.
        """
        completed = self.run_script(tmp_path, self.SCRIPT)

        assert completed.returncode != 0, "the unguarded script should not have succeeded"
        assert "ParallelBootstrapError" in completed.stderr, (
            f"the underlying failure reached the user untranslated:\n{completed.stderr[-1500:]}"
        )
        # The message has to carry the fix, not just name the failure.
        assert '__name__ == "__main__"' in completed.stderr

    def test_the_guarded_form_works(self, tmp_path: Path) -> None:
        completed = self.run_script(tmp_path, self.GUARDED_SCRIPT)

        assert completed.returncode == 0, completed.stderr[-2000:]
        # Any threshold from 0.4 up to but excluding 0.6 classifies this data perfectly;
        # which one comes back is the tie-breaking, not the point of this test.
        assert 0.4 <= float(completed.stdout.strip()) < 0.6

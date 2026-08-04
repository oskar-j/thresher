"""Counting across several processes, with the `mp` backend.

The `__main__` guard is required, not stylistic. This example asks for worker processes,
and on platforms whose start method is 'spawn' (macOS and Windows) each worker re-imports
this file. Without the guard every worker would re-run the example and start workers of
its own. Before 0.7.0 that hung the machine; it now raises `ParallelBootstrapError` and
says what to do - but the guard is still the thing to do.

Two spellings of the same request are shown. `backend='mp'` is the general one and works
for `exact` and `grid` as well; linear search's `n_jobs` predates it, means the same
thing, and since 0.7.0 runs on the same backend.
"""

import random

import thresher
from thresher.backends import MultiprocessingBackend


def sample_data(size: int = 40_000) -> tuple[list[float], list[int]]:
    """Build enough data to be worth dividing between processes.

    Below a couple of thousand rows the backend counts in this process instead: handing
    work to another one would cost more than the counting saves.

    Args:
        size: how many samples to generate.

    Returns:
        A `(scores, actual_classes)` pair, separable at around 0.4.
    """
    rng = random.Random(7)
    scores = [rng.random() for _ in range(size)]
    return scores, [1 if score > 0.4 else -1 for score in scores]


def main() -> None:
    scores, actual_classes = sample_data()

    # The general form: any of exact, ls and grid can use it.
    parallel = thresher.Thresher(backend="mp", verbose=True)
    print(f"Exact sweep on the mp backend: {parallel.optimize_threshold(scores, actual_classes)}")

    # The same thing with the process count chosen explicitly. -1 would mean every
    # processor bar one.
    configured = thresher.Thresher(backend=MultiprocessingBackend(num_workers=2))
    print(f"...over two workers:           {configured.optimize_threshold(scores, actual_classes)}")

    # And in this process, for comparison. A backend changes where the counting happens,
    # never the answer, so these three agree exactly.
    local = thresher.Thresher()
    print(f"...and in this process:        {local.optimize_threshold(scores, actual_classes)}")

    # Linear search's older spelling of the same request. Kept smaller: ls is O(n^2).
    legacy = thresher.Thresher(algorithm="ls", algorithm_params={"n_jobs": 3})
    print(f"Linear search with n_jobs=3:   {legacy.optimize_threshold(scores[:3000], actual_classes[:3000])}")

    print("Done")


if __name__ == "__main__":
    main()

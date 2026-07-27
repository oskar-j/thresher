"""The registry of selectable algorithms, and lookup by name."""

from typing import NamedTuple

from thresher.exceptions import UnknownAlgorithmError


class Algorithm(NamedTuple):
    """A selectable algorithm.

    Attributes:
        id: the canonical short name, and the key in `available_algorithms`.
        full_name: human-readable name, used in verbose output.
        synonyms: alternative names accepted by `retrieve_by_alias`.
        data_vol_thresh: input size beyond which this algorithm is slow enough to be worth
            warning about. `run_computations` logs a warning above it, so nobody starts a
            run that will take far longer than they expect.

            Each value is roughly where a run passes ten seconds, extrapolated from the
            timings in `examples/benchmark.py` on one laptop. They are order-of-magnitude
            guidance rather than promises - a faster machine moves them all up - which is
            why crossing one is a warning rather than a refusal.
    """

    id: str
    full_name: str
    synonyms: list[str]
    data_vol_thresh: int


available_algorithms: dict[str, Algorithm] = {
    "exact": Algorithm(
        id="exact",
        # 'auto', 'default' and 'default_heuristics' named the oracle until 0.5.0 removed
        # it. They are kept as aliases of the default so existing calls keep working.
        synonyms=["sweep", "exact_sweep", "sorted_sweep", "auto", "default", "default_heuristics"],
        full_name="Exact sweep",
        # O(n log n); 12 ms at 16,000 rows, so it is the input size rather than the
        # algorithm that runs out first
        data_vol_thresh=10_000_000,
    ),
    "ls": Algorithm(
        id="ls",
        synonyms=["linear", "linear_search"],
        full_name="Linear search",
        # O(n^2), and it bites early: 0.9 s at 4,000 rows becomes 18 s at 16,000
        data_vol_thresh=10_000,
    ),
    "sgd": Algorithm(
        id="sgd",
        synonyms=["curve_fitting"],
        full_name="Stochastic gradient descent",
        # linear in the sampled fraction of the data
        data_vol_thresh=2_000_000,
    ),
    "gen": Algorithm(
        id="gen",
        synonyms=["genetic", "sim"],
        full_name="Genetic algorithm",
        # linear, but with thousands of evaluations per run it is the slowest of the
        # approximations
        data_vol_thresh=100_000,
    ),
    "grid": Algorithm(
        id="grid",
        synonyms=["grid-search", "gs"],
        full_name="Grid search",
        # linear, with a fixed candidate count set by no_of_decimal_places
        data_vol_thresh=1_000_000,
    ),
    "sgrid": Algorithm(
        id="sgrid",
        synonyms=["random-grid-search", "rn-grid", "s-grid"],
        full_name="Stochastic grid search",
        # like grid, but each candidate reads only stoch_ratio of the data
        data_vol_thresh=10_000_000,
    ),
}

# What `Thresher()` uses when no algorithm is named.
DEFAULT = available_algorithms["exact"]


def retrieve_by_alias(name: str) -> Algorithm:
    """Resolve an algorithm by its id or by one of its synonyms.

    Args:
        name: an algorithm id such as `'grid'`, or a synonym such as `'sim'`. Matched
            case-insensitively, ids first and synonyms second.

    Returns:
        The matching `Algorithm` from `available_algorithms`.

    Raises:
        UnknownAlgorithmError: if the name matches nothing. It is a `ValueError`, and
            carries `.name` and `.available` alongside the message.
    """
    name = name.lower()
    try:
        return available_algorithms[name]
    except KeyError:
        # try to match by the 'alternate name'
        try:
            return next(_ for _ in available_algorithms.values() if name in _.synonyms)
        except StopIteration:
            # 'next' on an exhausted generator raises StopIteration, which says nothing
            # about what went wrong and reads as a bug rather than a bad argument.
            raise UnknownAlgorithmError(name, available_algorithms) from None

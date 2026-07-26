"""The registry of selectable algorithms, and lookup by name."""

from typing import NamedTuple

from thresher.exceptions import UNKNOWN_ALGORITHM_NAME


class Algorithm(NamedTuple):
    """A selectable algorithm.

    Attributes:
        id: the canonical short name, and the key in `available_algorithms`.
        full_name: human-readable name, used in verbose output.
        synonyms: alternative names accepted by `retrieve_by_alias`.
        data_vol_thresh: advisory upper bound on the input size for which this algorithm
            is a sensible manual choice. Since 0.4.0 the oracle no longer routes on it -
            `exact` is both exact and cheaper than every alternative, so there is nothing
            left to trade off - but it still records where each of the others stops being
            practical. `None` means no particular ceiling.
    """

    id: str
    full_name: str
    synonyms: list[str]
    data_vol_thresh: int | None


available_algorithms: dict[str, Algorithm] = {
    "auto": Algorithm(
        id="auto",
        synonyms=["default", "default_heuristics"],
        data_vol_thresh=None,
        full_name="Default heuristics",
    ),
    "exact": Algorithm(
        id="exact",
        synonyms=["sweep", "exact_sweep", "sorted_sweep"],
        data_vol_thresh=None,
        full_name="Exact sweep",
    ),
    "ls": Algorithm(
        id="ls",
        synonyms=["linear", "linear_search"],
        data_vol_thresh=1000,
        full_name="Linear search",
    ),
    "sgd": Algorithm(
        id="sgd",
        synonyms=["curve_fitting"],
        data_vol_thresh=None,
        full_name="Stochastic gradient descent",
    ),
    "gen": Algorithm(
        id="gen",
        synonyms=["genetic", "sim"],
        data_vol_thresh=None,
        full_name="Genetic algorithm",
    ),
    "grid": Algorithm(
        id="grid",
        synonyms=["grid-search", "gs"],
        data_vol_thresh=50 * 1000,
        full_name="Grid search",
    ),
    "sgrid": Algorithm(
        id="sgrid",
        synonyms=["random-grid-search", "rn-grid", "s-grid"],
        data_vol_thresh=None,
        full_name="Stochastic grid search",
    ),
}

DEFAULT = available_algorithms["auto"]


def retrieve_by_alias(name: str) -> Algorithm:
    """Resolve an algorithm by its id or by one of its synonyms.

    Args:
        name: an algorithm id such as `'grid'`, or a synonym such as `'sim'`. Matched
            case-insensitively, ids first and synonyms second.

    Returns:
        The matching `Algorithm` from `available_algorithms`.

    Raises:
        ValueError: if the name matches nothing. The message names the offending value
            and lists the ids that would have worked.
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
            raise ValueError(
                UNKNOWN_ALGORITHM_NAME.format(name=name, available=", ".join(sorted(available_algorithms)))
            ) from None

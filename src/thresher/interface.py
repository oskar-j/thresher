"""The user-facing interface: the `Thresher` class."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from thresher import algorithm
from thresher.algorithm import DEFAULT
from thresher.backends import get_backend
from thresher.dispatch import run_computations, validate_algorithm_params
from thresher.exceptions import (
    ALGORITHM_PARAMS_TYPE,
    UNKNOWN_OPTIONS,
    ConfigurationError,
    NotIterableError,
)
from thresher.utils import map_labels, validate_label_mapping

#: Every option `Thresher.__init__` accepts. Anything else is a mistyped name, and a
#: mistyped name is never harmless here: it leaves the default silently in place, so the
#: caller believes they configured a run they did not.
KNOWN_OPTIONS = frozenset(
    {"algorithm", "allow_parallel", "verbose", "progress_bar", "algorithm_params", "labels", "backend"}
)


class ThresherBase:
    """Internal plumbing behind `Thresher`.

    Split out so the public class holds only the documented API. Instances carry their
    configuration in `options`, which `Thresher.__init__` populates.
    """

    options: dict[str, Any]

    def _compute(
        self,
        chosen_algorithm: algorithm.Algorithm,
        scores: Sequence[float],
        actual_classes: Sequence[int],
    ) -> float:
        """Hand the work to the dispatcher, applying this instance's options.

        Args:
            chosen_algorithm: the algorithm to run.
            scores: the values being split.
            actual_classes: the matching classes, already normalized to -1 and 1.

        Returns:
            The threshold chosen by the algorithm.
        """
        return run_computations(
            chosen_algorithm,
            scores,
            actual_classes,
            self.options["verbose"],
            self.options["progress_bar"],
            self.options["allow_parallel"],
            self.options["algorithm_params"],
            get_backend(self.options["backend"]),
        )


class Thresher(ThresherBase):
    """Find the threshold that best separates two classes of scores.

    Example:
        >>> t = Thresher()
        >>> t.optimize_threshold([0.1, 0.3, 0.4, 0.7], [-1, -1, 1, 1])
        0.35

    An instance is reusable: the options are fixed at construction (or through
    `set_algorithm`), and `optimize_threshold` may be called repeatedly.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Create a new Thresher object, an interface to the Thresher evaluator.

        Note:
            No need to pass any extra arguments if you don't understand what you're doing.

        Args:
            **kwargs: any of the documented options - 'algorithm', 'allow_parallel',
                'verbose', 'progress_bar', 'algorithm_params', 'labels' and 'backend'.
                `labels=None` is accepted and means no mapping, the same as leaving the
                option out.

        Raises:
            ConfigurationError: if an option is passed whose name is not one of the
                documented seven, if 'algorithm' names no known algorithm, if 'backend'
                names no known backend, or if 'algorithm_params' is not a mapping or
                holds a key the chosen algorithm does not read. It is a `ValueError`.
            LabelMappingError: if 'labels' is given but is not a two-item list or tuple.
                It is a `TypeError`.
            BackendDependencyError: if 'backend' is 'ray' and Ray is not installed. It is
                an `ImportError`.
        """
        super().__init__()

        self.options: dict[str, Any] = {
            "algorithm": DEFAULT.id,
            "allow_parallel": True,
            "verbose": False,
            "progress_bar": False,
            "algorithm_params": {},
            "backend": "local",
        }

        unknown_options = set(kwargs) - KNOWN_OPTIONS
        if unknown_options:
            raise ConfigurationError(
                UNKNOWN_OPTIONS.format(
                    unknown=", ".join(sorted(repr(name) for name in unknown_options)),
                    valid=", ".join(sorted(KNOWN_OPTIONS)),
                )
            )

        self.options.update(kwargs)

        self.options["algorithm"] = algorithm.retrieve_by_alias(self.options["algorithm"])
        # Resolve and validate now rather than at optimize_threshold time, so a bad
        # backend name - or a missing Ray, or an unusable labels pair, or a mistyped
        # algorithm parameter - is reported when the object is built, not several seconds
        # into a long run.
        if self.options.get("labels") is not None:
            validate_label_mapping(self.options["labels"])
        if not isinstance(self.options["algorithm_params"], Mapping):
            raise ConfigurationError(
                ALGORITHM_PARAMS_TYPE.format(got=type(self.options["algorithm_params"]).__name__)
            )
        validate_algorithm_params(self.options["algorithm"], self.options["algorithm_params"])
        get_backend(self.options["backend"])

    def get_current_algorithm(self) -> dict[str, Any]:
        """Get the algorithm this instance is currently set to use.

        Returns:
            A dict with `name`, the algorithm's short id, and `object`, the `Algorithm`
            itself. Since 0.5.0 this is always the algorithm that will actually run: there
            is no longer a per-call choice for it to disagree with.
        """
        current_algorithm: algorithm.Algorithm = self.options["algorithm"]
        return {"name": current_algorithm.id, "object": current_algorithm}

    def get_current_options(self) -> dict[str, Any]:
        """Get this instance's configuration.

        Returns:
            The live options dict, not a copy - mutating it changes the instance.
        """
        return self.options

    def set_algorithm(self, algorithm_name: str) -> "Thresher":
        """Select the algorithm to use, by id or by one of its synonyms.

        Args:
            algorithm_name: an algorithm id such as `'grid'`, or a synonym such as
                `'sim'` or `'linear'`. Case-insensitive.

        Returns:
            This same instance, so calls can be chained.

        Raises:
            UnknownAlgorithmError: if the name matches no known algorithm, a
                `ValueError`. This previously printed a
                message and carried on with the old algorithm still in place, which left
                callers believing a switch had happened when it had not.
            ConfigurationError: if the `algorithm_params` already held by this instance
                are not all read by the new algorithm - `stoch_ratio` means nothing to
                `exact`, and silently ignoring it here would recreate the problem
                validating at construction exists to prevent. Also a `ValueError`.
        """
        chosen = algorithm.retrieve_by_alias(algorithm_name)
        validate_algorithm_params(chosen, self.options["algorithm_params"])
        self.options["algorithm"] = chosen
        return self

    @staticmethod
    def get_supported_algorithms(as_dict: bool = False) -> list[str] | dict[str, str]:
        """Get the algorithms this build supports.

        Args:
            as_dict: return a mapping of id to human-readable name instead of a list.

        Returns:
            A list of algorithm ids, or a dict of id to full name when `as_dict` is set.
            Every entry is a real algorithm; `'auto'` was removed in 0.5.0 along with the
            oracle, and survives only as a synonym of the default.
        """
        if as_dict:
            return {k: v.full_name for k, v in algorithm.available_algorithms.items()}
        return list(algorithm.available_algorithms.keys())

    def optimize_threshold(self, scores: Iterable[float], actual_classes: Iterable[Any]) -> float:
        """Find the threshold that classifies the most samples correctly.

        Args:
            scores: the scores to split, e.g. the output of a `predict_proba`.
            actual_classes: the ground-truth classes, as -1 and 1 unless the 'labels'
                option declares a different pair.

        Returns:
            The threshold yielding the highest fraction of correctly classified samples.
            Where several thresholds tie, any one of them may be returned.

        Raises:
            NotIterableError: if either argument is not iterable, naming the offending
                one. It is an `AttributeError`.
            InvalidInputError: if the labels are empty, single-class, outside (-1, 1), or
                a different length from the scores. It is a `ValueError`.
        """
        if not isinstance(scores, Iterable):
            raise NotIterableError("scores")
        if not isinstance(actual_classes, Iterable):
            raise NotIterableError("actual_classes")

        # Materialise only what has to be. Copying a caller's list costs memory
        # proportional to the input, which defeats the algorithms whose own allocation
        # does not grow with it - `hist` holds a few kilobytes of counters however large
        # the data is, and would still have paid for two full copies to get here. A
        # Sequence can already be measured and iterated more than once, which is all the
        # solvers need; anything else is consumed into a list as before.
        score_values: Sequence[float] = scores if isinstance(scores, Sequence) else list(scores)
        class_values: Sequence[int]
        if self.options.get("labels") is not None:
            class_values = list(map_labels(actual_classes, self.options["labels"]))
        elif isinstance(actual_classes, Sequence):
            class_values = actual_classes
        else:
            class_values = list(actual_classes)

        chosen_algorithm: algorithm.Algorithm = self.options["algorithm"]

        if self.options["verbose"]:
            print(f"Chosen algorithm: {chosen_algorithm.full_name}")

        return self._compute(chosen_algorithm, score_values, class_values)

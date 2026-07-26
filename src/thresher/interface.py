"""The user-facing interface: the `Thresher` class."""

from collections.abc import Iterable, Mapping
from typing import Any

from thresher import algorithm
from thresher.backends import get_backend
from thresher.exceptions import NotIterableError
from thresher.oracle import run_computations, run_oracle
from thresher.utils import map_labels


class ThresherBase:
    """Internal plumbing behind `Thresher`.

    Split out so the public class holds only the documented API. Instances carry their
    configuration in `options`, which `Thresher.__init__` populates.
    """

    options: dict[str, Any]

    def _run_oracle(self, data_traits: Mapping[str, Any]) -> algorithm.Algorithm:
        """Resolve which algorithm to use for this call.

        Args:
            data_traits: measurements of the input, passed to the oracle when the
                algorithm option is left at its default.

        Returns:
            The algorithm the oracle selected, or the one explicitly configured.
        """
        if self.options["algorithm"] == algorithm.DEFAULT:
            if self.options["verbose"]:
                print("Running heuristics on choosing a proper algorithm")
            chosen_algorithm = run_oracle(data_traits)

        else:
            chosen_algorithm = self.options["algorithm"]

        if self.options["verbose"]:
            print(f"Chosen algorithm: {chosen_algorithm.full_name}")

        return chosen_algorithm

    def _compute(
        self,
        chosen_algorithm: algorithm.Algorithm,
        scores: list[float],
        actual_classes: list[int],
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

        Raises:
            ConfigurationError: if 'algorithm' names no known algorithm, or 'backend' no
                known backend. It is a `ValueError`.
            BackendDependencyError: if 'backend' is 'ray' and Ray is not installed. It is
                an `ImportError`.
        """
        super().__init__()

        self.options: dict[str, Any] = {
            "algorithm": "auto",
            "allow_parallel": True,
            "verbose": False,
            "progress_bar": False,
            "algorithm_params": {},
            "backend": "local",
        }

        self.options.update(kwargs)

        self.options["algorithm"] = algorithm.retrieve_by_alias(self.options["algorithm"])
        # Resolve now rather than at optimize_threshold time, so a bad backend name - or a
        # missing Ray - is reported when the object is built, not several seconds into a
        # long run.
        get_backend(self.options["backend"])

    def get_current_algorithm(self) -> dict[str, Any]:
        """Get the algorithm this instance is currently set to use.

        Returns:
            A dict with `name`, the algorithm's short id, and `object`, the `Algorithm`
            itself. Note this reports what was configured: with the default `'auto'` the
            name is `'auto'`, not whatever the oracle will go on to choose per call.
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
        """
        self.options["algorithm"] = algorithm.retrieve_by_alias(algorithm_name)
        return self

    @staticmethod
    def get_supported_algorithms(as_dict: bool = False) -> list[str] | dict[str, str]:
        """Get the algorithms this build supports.

        Args:
            as_dict: return a mapping of id to human-readable name instead of a list.

        Returns:
            A list of algorithm ids, or a dict of id to full name when `as_dict` is set.
            Includes `'auto'`, which is the oracle rather than an algorithm as such.
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
            NotIterableError: if either argument is not iterable. It is an
                `AttributeError`.
            InvalidInputError: if the labels are empty, single-class, outside (-1, 1), or
                a different length from the scores. It is a `ValueError`.
        """
        if not isinstance(scores, Iterable):
            raise NotIterableError
        if not isinstance(actual_classes, Iterable):
            raise NotIterableError

        score_values = list(scores)
        if ("labels" in self.options) and (isinstance(self.options["labels"], Iterable)):
            class_values = list(map_labels(actual_classes, self.options["labels"]))
        else:
            class_values = list(actual_classes)

        data_traits = {"data_length": len(score_values)}

        chosen_algorithm = self._run_oracle(data_traits)

        return self._compute(chosen_algorithm, score_values, class_values)

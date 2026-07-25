from collections.abc import Iterable, Mapping
from typing import Any

from thresher import algorithm
from thresher.exceptions import NOT_IMPLEMENTED_ERROR
from thresher.oracle import run_computations, run_oracle
from thresher.utils import map_labels


class ThresherBase:
    options: dict[str, Any]

    def _run_oracle(self, data_traits: Mapping[str, Any]) -> algorithm.Algorithm:
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
        return run_computations(
            chosen_algorithm,
            scores,
            actual_classes,
            self.options["verbose"],
            self.options["progress_bar"],
            self.options["allow_parallel"],
            self.options["algorithm_params"],
        )


class Thresher(ThresherBase):
    def __init__(self, **kwargs: Any) -> None:
        """Create a new Thresher object, an interface to the Thresher evaluator.

        Note:
            No need to pass any extra arguments if you don't understand what you're doing.

        Args:
            **kwargs: any of the documented options - 'algorithm', 'allow_parallel',
                'verbose', 'progress_bar', 'algorithm_params' and 'labels'.

        Raises:
            ValueError: if 'algorithm' names no known algorithm.
        """
        super().__init__()

        self.options: dict[str, Any] = {
            "algorithm": "auto",
            "allow_parallel": True,
            "verbose": False,
            "progress_bar": False,
            "algorithm_params": {},
        }

        self.options.update(kwargs)

        self.options["algorithm"] = algorithm.retrieve_by_alias(self.options["algorithm"])

    def get_current_algorithm(self) -> dict[str, Any]:
        """Get the algorithm this instance is currently set to use."""
        current_algorithm: algorithm.Algorithm = self.options["algorithm"]
        return {"name": current_algorithm.id, "object": current_algorithm}

    def get_current_options(self) -> dict[str, Any]:
        return self.options

    def set_algorithm(self, algorithm_name: str) -> "Thresher":
        """Select the algorithm to use, by id or by one of its synonyms.

        Raises:
            ValueError: if the name matches no known algorithm. This previously printed a
                message and carried on with the old algorithm still in place, which left
                callers believing a switch had happened when it had not.
        """
        self.options["algorithm"] = algorithm.retrieve_by_alias(algorithm_name)
        return self

    @staticmethod
    def get_supported_algorithms(as_dict: bool = False) -> list[str] | dict[str, str]:
        """Get the algorithms this build supports, as ids or as id -> full name."""
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
            AttributeError: if either argument is not iterable.
            ValueError: if the labels are empty, single-class, or outside (-1, 1).
        """
        if not isinstance(scores, Iterable):
            raise AttributeError(NOT_IMPLEMENTED_ERROR)
        if not isinstance(actual_classes, Iterable):
            raise AttributeError(NOT_IMPLEMENTED_ERROR)

        score_values = list(scores)
        if ("labels" in self.options) and (isinstance(self.options["labels"], Iterable)):
            class_values = list(map_labels(actual_classes, self.options["labels"]))
        else:
            class_values = list(actual_classes)

        data_traits = {"data_length": len(score_values)}

        chosen_algorithm = self._run_oracle(data_traits)

        return self._compute(chosen_algorithm, score_values, class_values)

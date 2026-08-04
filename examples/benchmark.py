"""Benchmark every algorithm against the exact optimum.

Produces the table in the README. Run it with:

    uv run python examples/benchmark.py

Accuracy is measured on the whole dataset, against the best accuracy any single threshold
could achieve on that dataset - computed exactly here by sweeping the sorted scores, not
by asking one of the algorithms under test.
"""

import random
import statistics
import time
from collections.abc import Callable

import thresher

Dataset = tuple[list[float], list[int]]

SEEDS = 5
SIZE = 2000


def exact_best_accuracy(scores: list[float], actual_classes: list[int]) -> float:
    """The highest accuracy any threshold can reach on this dataset.

    Sweeps the sorted scores keeping running class counts, which is O(n log n) - the
    algorithms under test are what we are measuring, so the reference cannot be one of
    them.

    Args:
        scores: the values being split.
        actual_classes: the matching ground-truth classes, as -1 and 1.

    Returns:
        The best achievable fraction of correctly classified samples.
    """
    paired = sorted(zip(scores, actual_classes, strict=True))
    total_positive = sum(1 for _, c in paired if c == 1)
    total = len(paired)

    # Threshold above everything: all predicted negative.
    correct = total - total_positive
    best = correct
    # Walk the threshold down, highest score first, so samples flip to positive in the
    # order they actually would. Sweeping upwards instead flips them in the wrong order
    # and reports an impossible ceiling.
    for _, actual in reversed(paired):
        correct += 1 if actual == 1 else -1
        best = max(best, correct)
    return best / total


def accuracy_of(threshold: float, scores: list[float], actual_classes: list[int]) -> float:
    """Fraction of samples the given threshold classifies correctly."""
    right = sum(
        1
        for score, actual in zip(scores, actual_classes, strict=True)
        if (1 if score > threshold else -1) == actual
    )
    return right / len(scores)


def separable(n: int, seed: int) -> Dataset:
    """Two classes that a single threshold can split perfectly."""
    random.seed(seed)
    scores = sorted(random.random() for _ in range(n))
    return scores, [-1] * (n // 2) + [1] * (n - n // 2)


def overlapping(n: int, seed: int, flip: float = 0.15) -> Dataset:
    """Classes that overlap, so even the best threshold gets some wrong."""
    random.seed(seed)
    scores = [random.random() for _ in range(n)]
    labels = [1 if s > 0.5 else -1 for s in scores]
    return scores, [(-lab if random.random() < flip else lab) for lab in labels]


def imbalanced(n: int, seed: int, boundary: float = 0.9) -> Dataset:
    """One class is rare, so the optimum sits far from the mean of the scores."""
    random.seed(seed)
    scores = sorted(random.random() for _ in range(n))
    return scores, [1 if s > boundary else -1 for s in scores]


DATASETS: dict[str, Callable[[int, int], Dataset]] = {
    "Separable": separable,
    "Overlapping": overlapping,
    "Imbalanced": imbalanced,
}

ALGORITHMS = ["exact", "hist", "ls", "grid", "sgrid", "gen", "sgd"]

# Cost of a run, in the number of (score, class) pairs examined.
#
#   n  the number of scores
#   c  grid candidates, 10**no_of_decimal_places + 1, so 101 by default
#   r  stoch_ratio, the fraction of the data each evaluation reads
#   e  genetic evaluations, population_size * number_of_generations * number_of_iterations
#   i  sgd steps, at most num_of_iters
#
# 'exact' sorts once and sweeps, carrying running class counts, so it never rescores a
# candidate - the sort is all it costs. 'ls' scores each of the n-1 candidate midpoints
# against all n samples, which is the same search done the expensive way. For the
# approximate algorithms the candidate count comes from their own parameters rather than
# from n, which leaves them linear but inexact.
COMPLEXITY = {
    "exact": "O(n log n)",
    "hist": "O(n + k)",
    "ls": "O(n²)",
    "grid": "O(c·n)",
    "sgrid": "O(c·r·n)",
    "gen": "O(e·r·n)",
    "sgd": "O(i·r·n)",
}

# Peak extra allocation, read off the implementations rather than measured:
#
#   exact  a dict keyed by *distinct* score, so d rather than n - which is why rounded
#          probabilities cost it far less than the row count suggests
#   ls     every midpoint between adjacent sorted scores, plus the sorted copy
#   grid   only the grid itself; the data is streamed through a lazy zip, so this is the
#          one algorithm whose memory does not grow with the input at all
#   sgrid  the sampled pairs only. Until 0.6.4 it materialised the full paired list
#          before sampling from it, paying O(n) per candidate despite reading a fraction
#   gen    the sampled indices, plus a fitness sample per agent per iteration, which is a
#          constant with respect to n
#   sgd    the sampled indices only
#   hist   k pairs of counters and nothing else - it reads each row once and keeps none,
#          so its allocation is set by the resolution rather than by the input
MEMORY = {
    "exact": "O(d)",
    "hist": "O(k)",
    "ls": "O(n)",
    "grid": "O(c)",
    "sgrid": "O(r·n)",
    "gen": "O(r·n)",
    "sgd": "O(r·n)",
}


def main() -> None:
    """Run the benchmark and print the results as a markdown table."""
    results: dict[str, dict[str, tuple[float, float]]] = {}

    for data_name, builder in DATASETS.items():
        results[data_name] = {}
        for algorithm_name in ALGORITHMS:
            accuracies, elapsed = [], []
            for seed in range(SEEDS):
                scores, actual_classes = builder(SIZE, seed)
                ceiling = exact_best_accuracy(scores, actual_classes)

                started = time.perf_counter()
                threshold = thresher.Thresher(algorithm=algorithm_name).optimize_threshold(
                    scores, actual_classes
                )
                elapsed.append(time.perf_counter() - started)

                # How much of the achievable accuracy this algorithm actually captured.
                accuracies.append(accuracy_of(threshold, scores, actual_classes) / ceiling)
            results[data_name][algorithm_name] = (
                statistics.mean(accuracies),
                statistics.mean(elapsed),
            )

    print(f"\n{SIZE} rows, mean of {SEEDS} seeds. Accuracy is relative to the exact optimum.\n")
    header = "| Algorithm | " + " | ".join(DATASETS) + " | Time | Complexity | Memory |"
    print(header)
    print("|" + "---|" * (len(DATASETS) + 4))
    for algorithm_name in ALGORITHMS:
        cells = [f"{results[d][algorithm_name][0] * 100:.2f}%" for d in DATASETS]
        mean_time = statistics.mean(results[d][algorithm_name][1] for d in DATASETS)
        print(
            f"| `{algorithm_name}` | "
            + " | ".join(cells)
            + f" | {mean_time * 1000:.0f} ms | {COMPLEXITY[algorithm_name]}"
            + f" | {MEMORY[algorithm_name]} |"
        )


if __name__ == "__main__":
    main()

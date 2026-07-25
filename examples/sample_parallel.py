"""Linear search across several processes.

The `__main__` guard is required, not stylistic: this example asks for multiprocessing,
and on platforms whose start method is 'spawn' (macOS and Windows) each worker re-imports
this file. Without the guard every worker would re-run the example and spawn its own
workers, which hangs the machine rather than failing.
"""

import thresher


def main() -> None:
    t = thresher.Thresher(verbose=True, algorithm_params={"n_jobs": 3})

    print("Currently supported algorithms:")
    print(t.get_supported_algorithms())

    case_small_scores = [0.1, 0.15, 0.2, 0.22, 0.27, 0.29, 0.3, 0.4, 0.7]
    case_small_labels = [-1, -1, -1, -1, -1, -1, -1, 1, 1]

    print(f"Optimization result: {t.optimize_threshold(case_small_scores, case_small_labels)}")

    print("Done")


if __name__ == "__main__":
    main()

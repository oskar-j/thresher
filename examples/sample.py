"""Basic usage, on a tiny input and on the real-world sample dataset."""

from sample_data import get_sample_data

import thresher


def main() -> None:
    t = thresher.Thresher(progress_bar=True)

    print("Currently supported algorithms:")
    print(t.get_supported_algorithms())

    case_small_scores = [0.1, 0.3, 0.4, 0.7]
    case_small_labels = [-1, -1, 1, 1]

    print(f"Optimization result: {t.optimize_threshold(case_small_scores, case_small_labels)}")

    # The columns go in as pandas holds them. Since 0.7.2 a Series is read where it lies
    # rather than copied into a list, so `list(...)` here would allocate the dataset a
    # second time for nothing - which matters at rather more than three thousand rows.
    medium_data = get_sample_data()
    case_medium_scores = medium_data["pred"]
    case_medium_labels = medium_data["actual"]

    t = thresher.Thresher(progress_bar=True, verbose=True)
    print(f"Optimization result: {t.optimize_threshold(case_medium_scores, case_medium_labels)}")

    t = thresher.Thresher(algorithm="gen", progress_bar=True, verbose=True)
    print(f"Alternative optimization result: {t.optimize_threshold(case_medium_scores, case_medium_labels)}")

    print("Done")


if __name__ == "__main__":
    main()

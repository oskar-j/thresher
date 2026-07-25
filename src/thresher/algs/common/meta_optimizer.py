"""Summary statistics used to seed a search before it starts."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def calculate_range_mean(scores: Sequence[float], actual_classes: Sequence[int], label: int) -> float:
    """Average the scores belonging to one class.

    The genetic solver takes the negative-class and positive-class means as the bounds of
    its initial population, so the search starts around where the boundary should lie.

    Args:
        scores: the values being split.
        actual_classes: the matching ground-truth classes, as -1 and 1.
        label: the class to average over, -1 or 1.

    Returns:
        The mean of the scores whose class equals `label`. Returns NaN if that class is
        absent, since numpy averages an empty selection.
    """
    return float(np.mean([_[0] for _ in zip(scores, actual_classes, strict=False) if _[1] == label]))


def get_mean_value_for_class_pd(label: Any, label_column: str, data: pd.DataFrame, data_column: str) -> float:
    """Average one column of a DataFrame over the rows belonging to one class.

    The pandas equivalent of `calculate_range_mean`, for callers holding a frame rather
    than parallel sequences. Nothing in the package calls this today.

    Args:
        label: the class to select on.
        label_column: name of the column holding the class labels.
        data: the frame to read.
        data_column: name of the column to average.

    Returns:
        The mean of `data_column` across the rows where `label_column` equals `label`.
    """
    return float(np.mean(data[data[label_column] == label][data_column]))

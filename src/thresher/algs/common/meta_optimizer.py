from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def calculate_range_mean(scores: Sequence[float], actual_classes: Sequence[int], label: int) -> float:
    return float(np.mean([_[0] for _ in zip(scores, actual_classes, strict=False) if _[1] == label]))


def get_mean_value_for_class_pd(label: Any, label_column: str, data: pd.DataFrame, data_column: str) -> float:
    return float(np.mean(data[data[label_column] == label][data_column]))

"""Shared fixtures.

The fixture files are located relative to this file rather than the working directory.
Before 0.3.0 the suite loaded them through a bare `'./'` prefix, so it only passed when
run from inside the tests directory - and `examples/sample.py` needed the opposite
working directory, making the two mutually exclusive.
"""

import random
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

DATA_DIR = Path(__file__).parent / "data"

Dataset = tuple[list[float], list[int]]
DatasetFactory = Callable[..., Dataset]


@pytest.fixture(scope="session")
def medium_dataset() -> Dataset:
    """The real-world anonymised sample, ~3k rows, whose optimum sits around 0.5."""
    frame = pd.concat(
        [
            pd.read_excel(DATA_DIR / "positives.xlsx", header=None, names=["pred", "actual"]),
            pd.read_excel(DATA_DIR / "negatives.xlsx", header=None, names=["pred", "actual"]),
        ]
    )
    return list(frame["pred"].values), list(frame["actual"].values)


@pytest.fixture
def separable() -> DatasetFactory:
    """Build a cleanly separable dataset of a given size.

    Separable data is the awkward case for the stochastic solvers: their evaluations can
    return a perfect score, which used to divide by zero and to send the sgd walk out of
    the data range.
    """

    def _make(n: int, seed: int = 0) -> Dataset:
        random.seed(seed)
        scores = sorted(random.random() for _ in range(n))
        return scores, [-1] * (n // 2) + [1] * (n - n // 2)

    return _make


@pytest.fixture
def tiny_dataset() -> Dataset:
    """The smallest input the README documents."""
    return [0.1, 0.3, 0.4, 0.7], [-1, -1, 1, 1]

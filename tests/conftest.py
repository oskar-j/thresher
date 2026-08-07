"""Shared fixtures.

The fixture files are located relative to this file rather than the working directory.
Before 0.3.0 the suite loaded them through a bare `'./'` prefix, so it only passed when
run from inside the tests directory - and `examples/sample.py` needed the opposite
working directory, making the two mutually exclusive.
"""

import random
from collections.abc import Callable, Iterator
from pathlib import Path

import pandas as pd
import pytest
from loguru import logger

from thresher import log

DATA_DIR = Path(__file__).parent / "data"

Dataset = tuple[list[float], list[int]]
DatasetFactory = Callable[..., Dataset]


@pytest.fixture(autouse=True)
def _fixed_random_state() -> None:
    """Start every test from the same global random state.

    Four solvers sample through the `random` module, so a test that asserts an outcome
    from one of them was really asserting something about wherever the module-level
    generator happened to be by the time that test ran - which is set by every test
    before it. Tests passed or failed on their position in the run.

    That is not theoretical. `TestScoresOutsideTheUnitInterval` demands exact accuracy
    from `sgrid`, which scores each candidate against 5 of the 100 rows by default: 43 of
    200 starting states fail it. It has been passing on the alignment the suite happened
    to have, and 0.7.3 - which changed how many values the genetic solver draws, and
    nothing else about `sgrid` - moved that alignment and turned it red.

    Seeding here does not make a stochastic solver deterministic in production. It makes
    each test reproducible and independent of the ones before it, so a failure means the
    code changed rather than that the order did.
    """
    random.seed(20260805)


@pytest.fixture(autouse=True)
def _default_verbosity() -> Iterator[None]:
    """Start every test at the package's default level, and leave it there.

    `set_verbosity` is process-wide by design - it is how a program sets the library's
    default, and `thresher.cli` calls it on every invocation. In a test session that means
    one test invoking the command line at `-vv` would leave the next one running at DEBUG,
    which is the same order-dependence `_fixed_random_state` exists to remove.

    Yields:
        None, with the level at its default for the duration of the test.
    """
    log.set_verbosity(log.DEFAULT_VERBOSITY)
    yield
    log.set_verbosity(log.DEFAULT_VERBOSITY)


@pytest.fixture
def logs() -> Iterator[list[str]]:
    """Collect everything this package logs during a test.

    A loguru sink appending to a list, which is what `caplog` was doing before 0.8.0 moved
    the reporting off the standard library. It takes every level and leaves the filtering
    to the code under test, because what is being asserted is usually that a level *was*
    applied.

    Yields:
        A list of `LEVEL:module:message` lines, in the order they were logged. It fills as
        the test runs, so it can be read repeatedly.
    """
    collected: list[str] = []
    sink_id = logger.add(collected.append, level=0, format="{level}:{name}:{message}")
    try:
        yield collected
    finally:
        logger.remove(sink_id)


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
def skewed() -> DatasetFactory:
    """Build a dataset whose optimal threshold sits far from the mean score.

    The classes are imbalanced, so the boundary is nowhere near the middle. This is the
    case that exposed the stalling `sgd` walk in 0.3.0: it starts at the mean of the
    scores and has to travel to reach the answer.
    """

    def _make(n: int, boundary: float, seed: int = 0) -> Dataset:
        random.seed(seed)
        scores = sorted(random.random() for _ in range(n))
        return scores, [1 if score > boundary else -1 for score in scores]

    return _make


@pytest.fixture
def tiny_dataset() -> Dataset:
    """The smallest input the README documents."""
    return [0.1, 0.3, 0.4, 0.7], [-1, -1, 1, 1]

"""Loader for the sample dataset shared by the examples.

The path is resolved relative to this file, so the examples run from any working
directory.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "data"


def get_sample_data() -> pd.DataFrame:
    """Load the anonymised ~3k-row sample used across the examples."""
    return pd.concat(
        [
            pd.read_excel(DATA_DIR / "positives.xlsx", header=None, names=["pred", "actual"]),
            pd.read_excel(DATA_DIR / "negatives.xlsx", header=None, names=["pred", "actual"]),
        ]
    )

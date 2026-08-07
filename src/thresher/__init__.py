"""Thresher - find the threshold that maximizes classification accuracy."""

from importlib.metadata import version

from thresher.interface import Thresher
from thresher.log import propagate_to_logging, set_verbosity, verbosity

__version__ = version("thresher-py")
"""The installed package version, read from the distribution metadata.

`pyproject.toml` remains the single source of truth; this is the same record the
`thresher --version` command reports.
"""

__all__ = ["Thresher", "__version__", "propagate_to_logging", "set_verbosity", "verbosity"]

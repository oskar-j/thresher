"""Execution backends: where the counting happens.

`local` runs in this process and is the default. `ray` spreads the same counting over a
Ray cluster. A backend never changes the answer - see `base` for why that is enforceable.
"""

from typing import Any

from thresher.backends.base import Backend
from thresher.backends.local import LocalBackend

AVAILABLE_BACKENDS = ("local", "ray")


def get_backend(backend: Any) -> Backend:
    """Resolve the `backend` option to something that can do the counting.

    Args:
        backend: the name of a backend, or an object already implementing the protocol -
            which is how a caller passes a pre-configured `RayBackend(num_shards=...)`.

    Returns:
        The backend to use.

    Raises:
        ValueError: if the name is not recognised.
        ImportError: if `'ray'` was asked for and Ray is not installed. The message says
            how to install it.
    """
    if not isinstance(backend, str):
        # Already a backend instance; trust the protocol.
        return backend  # type: ignore[no-any-return]

    name = backend.lower()
    if name == "local":
        return LocalBackend()
    if name == "ray":
        from thresher.backends.ray_backend import RayBackend

        return RayBackend()

    raise ValueError(f"Unknown backend {backend!r}. Available backends are: {', '.join(AVAILABLE_BACKENDS)}.")


__all__ = ["AVAILABLE_BACKENDS", "Backend", "LocalBackend", "get_backend"]

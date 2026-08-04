"""Execution backends: where the counting happens.

`local` runs in this process and is the default. `mp` spreads the same counting over this
machine's CPU cores, and `ray` over a Ray cluster. A backend never changes the answer -
see `base` for why that is enforceable.
"""

from typing import Any

from thresher.backends.base import Backend
from thresher.backends.local import LocalBackend
from thresher.backends.mp_backend import MultiprocessingBackend
from thresher.exceptions import UnknownBackendError

AVAILABLE_BACKENDS = ("local", "mp", "ray")


def get_backend(backend: Any) -> Backend:
    """Resolve the `backend` option to something that can do the counting.

    Args:
        backend: the name of a backend, or an object already implementing the protocol -
            which is how a caller passes a pre-configured `RayBackend(num_shards=...)`.

    Returns:
        The backend to use.

    Raises:
        UnknownBackendError: if the name is not recognised. It is a `ValueError`.
        BackendDependencyError: if `'ray'` was asked for and Ray is not installed. It is
            an `ImportError`, and the message says how to install it.

    Note:
        The names build default instances. To configure one - `MultiprocessingBackend(
        num_workers=4)`, `RayBackend(num_shards=...)` - construct it and pass the object
        as the `backend` option instead of the name.
    """
    if not isinstance(backend, str):
        # Already a backend instance; trust the protocol.
        return backend  # type: ignore[no-any-return]

    name = backend.lower()
    if name == "local":
        return LocalBackend()
    if name == "mp":
        return MultiprocessingBackend()
    if name == "ray":
        from thresher.backends.ray_backend import RayBackend

        return RayBackend()

    raise UnknownBackendError(backend, AVAILABLE_BACKENDS)


__all__ = [
    "AVAILABLE_BACKENDS",
    "Backend",
    "LocalBackend",
    "MultiprocessingBackend",
    "get_backend",
]

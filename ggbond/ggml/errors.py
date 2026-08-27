"""Error and null-check helpers shared across the wrapper.

Modules import the backing bindings directly as ``import ggml as _ggml`` (and
``from ggml import utils as _utils``); the alias keeps the external package
visually distinct from this ``ggbond.ggml`` subpackage. This module only holds
the small helpers built on top of those bindings.
"""

from __future__ import annotations

import ggml as _ggml

__all__ = ["GGMLError", "check_status", "nonnull", "status_name"]


class GGMLError(RuntimeError):
    """Raised when an underlying ggml call fails."""


def status_name(status) -> str:
    """Return ggml's name for a status code (``ggml_status_to_string``)."""
    try:
        name = _ggml.ggml_status_to_string(int(status))
    except Exception:  # pragma: no cover - older bindings without the helper
        return str(int(status))
    return name.decode() if isinstance(name, bytes) else str(name)


def check_status(status, what: str) -> None:
    """Raise :class:`GGMLError` unless ``status`` is ``GGML_STATUS_SUCCESS``."""
    if int(status) != _ggml.GGML_STATUS_SUCCESS:
        raise GGMLError(
            f"{what} failed with status {int(status)} ({status_name(status)})"
        )


def nonnull(ptr, what: str):
    """Return ``ptr`` if it is a non-null pointer, otherwise raise.

    Works for both ``c_void_p`` results (``None`` when NULL) and typed pointer
    results (falsy when NULL).
    """
    if not ptr:
        raise GGMLError(f"{what} returned a null pointer")
    return ptr

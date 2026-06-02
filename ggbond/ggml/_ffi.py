"""Isolation layer for the underlying ggml-python bindings.

This is the *only* module that imports the top-level ``ggml`` (ggml-python)
package. Every other module in :mod:`ggbond.ggml` imports the bindings from
here as ``_ggml`` / ``_utils``. Keeping the import in one place avoids confusion
with the subpackage name ``ggbond.ggml`` and makes it easy to swap the backing
bindings later.
"""

from __future__ import annotations

# Absolute import: resolves to the site-packages top-level ggml-python package,
# never to this subpackage (``ggbond.ggml``).
import ggml as _ggml
from ggml import utils as _utils

__all__ = ["_ggml", "_utils", "GGMLError", "check_status", "nonnull"]


class GGMLError(RuntimeError):
    """Raised when an underlying ggml call fails."""


def check_status(status, what: str) -> None:
    """Raise :class:`GGMLError` unless ``status`` is ``GGML_STATUS_SUCCESS``."""
    if int(status) != _ggml.GGML_STATUS_SUCCESS:
        raise GGMLError(f"{what} failed with status {int(status)}")


def nonnull(ptr, what: str):
    """Return ``ptr`` if it is a non-null pointer, otherwise raise.

    Works for both ``c_void_p`` results (``None`` when NULL) and typed pointer
    results (falsy when NULL).
    """
    if not ptr:
        raise GGMLError(f"{what} returned a null pointer")
    return ptr

"""GGML data types and numpy dtype conversion.

Reuses :class:`ggml.utils.GGML_TYPE` directly instead of defining a second
enum, so the values never drift from the underlying bindings.
"""

from __future__ import annotations

import numpy as np

from ggml import utils as _utils

# The canonical ggml type enum (F32 = 0, F16 = 1, I32 = 26, ...).
DType = _utils.GGML_TYPE

# Convenience aliases for the most common types.
F32 = DType.F32
F16 = DType.F16
I8 = DType.I8
I16 = DType.I16
I32 = DType.I32
I64 = DType.I64
F64 = DType.F64
BF16 = DType.BF16

__all__ = ["DType", "F32", "F16", "I8", "I16", "I32", "I64", "F64", "BF16",
           "dtype_from_numpy", "dtype_to_numpy"]


def dtype_from_numpy(np_dtype) -> "DType":
    """Map a numpy dtype to the corresponding :class:`DType`."""
    key = np.dtype(np_dtype).type
    try:
        return _utils.NUMPY_DTYPE_TO_GGML_TYPE[key]
    except KeyError:
        raise ValueError(f"no ggml type for numpy dtype {np_dtype!r}")


def dtype_to_numpy(dt: "DType"):
    """Map a :class:`DType` to the corresponding numpy dtype."""
    try:
        return np.dtype(_utils.GGML_TYPE_TO_NUMPY_DTYPE[DType(dt)])
    except KeyError:
        raise ValueError(f"no numpy dtype for ggml type {dt!r}")

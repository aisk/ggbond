"""GGBond: ggml from Python.

Two layers:

* this package -- the high-level API: a :class:`Session`, variables and graphs
  (see :mod:`ggbond.session`);
* :mod:`ggbond.ggml` -- a one-to-one object-oriented mapping of ggml itself,
  for when you need its concepts directly.
"""

__version__ = "0.1.0"

from .ggml import (
    BF16, F16, F32, F64, I8, I16, I32, I64,
    DType, GGMLError, PoolOp, RopeType, ScaleMode, SortOrder, Tensor,
)
from .session import GraphScope, Session, Variable, default_n_threads

__all__ = [
    "__version__",
    "Session",
    "GraphScope",
    "Variable",
    "default_n_threads",
    "Tensor",
    "DType",
    "GGMLError",
    "PoolOp", "RopeType", "ScaleMode", "SortOrder",
    "F32", "F16", "I8", "I16", "I32", "I64", "F64", "BF16",
]

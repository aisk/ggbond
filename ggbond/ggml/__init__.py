"""Object-oriented thin wrapper over ggml, backed by ggml-python.

Stays close to ggml: classes map to ggml concepts (``Context``, ``Tensor``,
``Backend``, ``Graph``, ``GAllocr``) and methods map ~1:1 to ggml functions.
Computation is explicit and two-phase -- nothing is lazy.

Typical CPU flow::

    import numpy as np
    from ggbond.ggml import Context, F32

    with Context(mem_size=16 * 1024 * 1024, no_alloc=False) as ctx:
        a = ctx.tensor_from_numpy(np.random.rand(4, 3).astype(np.float32))
        b = ctx.tensor_from_numpy(np.random.rand(2, 3).astype(np.float32))
        c = a.mul_mat(b)
        ctx.new_graph().build_forward_expand(c).compute_with_ctx(n_threads=4)
        print(c.numpy())
"""

from __future__ import annotations

from ._ffi import GGMLError
from .types import DType, dtype_from_numpy, dtype_to_numpy, F32, F16, I8, I16, I32, I64, F64, BF16
from .tensor import Tensor
from .context import Context
from .graph import Graph
from .gallocr import GAllocr
from .backend import Backend
from .gguf import GGUF, load_gguf

__all__ = [
    "GGMLError",
    "DType",
    "dtype_from_numpy",
    "dtype_to_numpy",
    "F32", "F16", "I8", "I16", "I32", "I64", "F64", "BF16",
    "Tensor",
    "Context",
    "Graph",
    "GAllocr",
    "Backend",
    "GGUF",
    "load_gguf",
]

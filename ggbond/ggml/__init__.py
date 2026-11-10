"""Object-oriented thin wrapper over ggml, backed by ggml-python.

Stays close to ggml: classes map to ggml concepts (``Context``, ``Tensor``,
``Backend``, ``Graph``, ``GAllocr``) and methods map ~1:1 to ggml functions.
Computation is explicit and two-phase -- nothing is lazy.

Typical CPU flow::

    import numpy as np
    from ggbond.ggml import Context, Backend, GAllocr, F32

    with Backend.cpu_init() as be, \\
         Context(mem_size=1 << 20, no_alloc=True) as ctx:
        be.set_n_threads(4)
        a = ctx.new_tensor_2d(F32, 3, 4, name="a")  # ggml order = reversed numpy shape
        b = ctx.new_tensor_2d(F32, 3, 2, name="b")
        c = b.mul_mat(a)                             # ggml_mul_mat(ctx, b, a) == a @ b.T
        be.alloc_ctx_tensors(ctx)
        a.write(np.random.rand(4, 3).astype(np.float32))  # upload via the tensor
        b.write(np.random.rand(2, 3).astype(np.float32))
        graph = ctx.new_graph().build_forward_expand(c)
        with GAllocr.from_backend(be) as alloc:
            alloc.alloc_graph(graph)
            be.compute(graph)
        out = c.to_numpy()
        print(out)
"""

from __future__ import annotations

from .errors import GGMLError
from .log import LogLevel, log_set
from .types import DType, dtype_from_numpy, dtype_to_numpy, F32, F16, I8, I16, I32, I64, F64, BF16
from .tensor import Tensor, Op, PoolOp, RopeType, ScaleMode, SortOrder, TensorFlag
from .context import Context, tensor_overhead
from .graph import Graph, graph_overhead_custom
from .gallocr import GAllocr
from .buffer import Buffer, BufferType
from .device import Device, DeviceType
from .backend import Backend
from .scheduler import Scheduler
from .gguf import GGUF, GGUFType

__all__ = [
    "GGMLError",
    "LogLevel",
    "log_set",
    "DType",
    "dtype_from_numpy",
    "dtype_to_numpy",
    "F32", "F16", "I8", "I16", "I32", "I64", "F64", "BF16",
    "Tensor",
    "Op",
    "PoolOp",
    "RopeType",
    "ScaleMode",
    "SortOrder",
    "TensorFlag",
    "Context",
    "tensor_overhead",
    "Graph",
    "graph_overhead_custom",
    "GAllocr",
    "Buffer",
    "BufferType",
    "Device",
    "DeviceType",
    "Backend",
    "Scheduler",
    "GGUF",
    "GGUFType",
]

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
        c = b.mul_mat(a)                             # numpy a @ b == b.mul_mat(a)
        be.alloc_ctx_tensors(ctx)
        a.set(np.random.rand(4, 3).astype(np.float32))  # upload via the tensor
        b.set(np.random.rand(2, 3).astype(np.float32))
        graph = ctx.new_graph().build_forward_expand(c)
        with GAllocr.from_backend(be) as alloc:
            alloc.alloc_graph(graph)
            be.compute(graph); be.synchronize()
        out = c.get(np.empty(tuple(reversed(c.ne)), dtype=np.float32))
        print(out)
"""

from __future__ import annotations

from .errors import GGMLError
from .types import DType, dtype_from_numpy, dtype_to_numpy, F32, F16, I8, I16, I32, I64, F64, BF16
from .tensor import (
    Tensor,
    POOL_MAX,
    POOL_AVG,
    ROPE_TYPE_NORMAL,
    ROPE_TYPE_NEOX,
    ROPE_TYPE_MROPE,
    ROPE_TYPE_VISION,
)
from .context import Context, tensor_overhead
from .graph import Graph, graph_overhead_custom
from .gallocr import GAllocr
from .buffer import Buffer, BufferType
from .device import Device
from .backend import Backend, DEVICE_CPU, DEVICE_GPU, DEVICE_IGPU, DEVICE_ACCEL
from .scheduler import Scheduler
from .gguf import GGUF

__all__ = [
    "GGMLError",
    "DType",
    "dtype_from_numpy",
    "dtype_to_numpy",
    "F32", "F16", "I8", "I16", "I32", "I64", "F64", "BF16",
    "Tensor",
    "POOL_MAX", "POOL_AVG",
    "ROPE_TYPE_NORMAL", "ROPE_TYPE_NEOX", "ROPE_TYPE_MROPE", "ROPE_TYPE_VISION",
    "Context",
    "tensor_overhead",
    "Graph",
    "graph_overhead_custom",
    "GAllocr",
    "Buffer",
    "BufferType",
    "Device",
    "Backend",
    "DEVICE_CPU", "DEVICE_GPU", "DEVICE_IGPU", "DEVICE_ACCEL",
    "Scheduler",
    "GGUF",
]

"""Graph abstraction, allocator wrapper, and GGUF model loading helper."""

from __future__ import annotations

import numpy as np

from ggbond import ggml
from ggbond.backend import Backend
from ggbond.context import Context


class Graph:
    """Computation graph that internally manages its own context.

    Provides two ways to call ggml ops:

    1. Convenience methods: ``g.add(a, b)`` — no need to pass ctx.
    2. Direct access: ``ggml.some_op(g.ctx, ...)`` via the :attr:`ctx` property.
    """

    def __init__(self, max_nodes: int | None = None, *, grads: bool = False):
        self._max_nodes = max_nodes or ggml.DEFAULT_GRAPH_SIZE()
        self._context = Context._for_graph(self._max_nodes, grads=grads)
        self._graph = self._context._new_graph(self._max_nodes, grads=grads)

    # -- properties -----------------------------------------------------------

    @property
    def ctx(self) -> ggml.Context:
        """Underlying ``ggml.Context`` pointer for direct ``ggml.xxx(g.ctx, ...)`` calls."""
        return self._context.raw

    @property
    def raw(self) -> ggml.Graph:
        """Underlying ``ggml.Graph`` pointer."""
        return self._graph

    # -- structural helpers ---------------------------------------------------

    def new_tensor(self, dtype, *shape: int, name: str | None = None) -> ggml.Tensor:
        """Create a tensor inside this graph's context."""
        return self._context.new_tensor(dtype, *shape, name=name)

    def build_forward(self, tensor: ggml.Tensor) -> None:
        """Expand the forward computation graph from *tensor*."""
        ggml.build_forward_expand(self._graph, tensor)

    # -- binary ops -----------------------------------------------------------

    def add(self, a: ggml.Tensor, b: ggml.Tensor) -> ggml.Tensor:
        return ggml.add(self.ctx, a, b)

    def sub(self, a: ggml.Tensor, b: ggml.Tensor) -> ggml.Tensor:
        return ggml.sub(self.ctx, a, b)

    def mul(self, a: ggml.Tensor, b: ggml.Tensor) -> ggml.Tensor:
        return ggml.mul(self.ctx, a, b)

    def div(self, a: ggml.Tensor, b: ggml.Tensor) -> ggml.Tensor:
        return ggml.div(self.ctx, a, b)

    def mul_mat(self, a: ggml.Tensor, b: ggml.Tensor) -> ggml.Tensor:
        return ggml.mul_mat(self.ctx, a, b)

    def concat(self, a: ggml.Tensor, b: ggml.Tensor, dim: int = 0) -> ggml.Tensor:
        return ggml.concat(self.ctx, a, b, dim)

    # -- unary / activation ops -----------------------------------------------

    def relu(self, a: ggml.Tensor) -> ggml.Tensor:
        return ggml.relu(self.ctx, a)

    def gelu(self, a: ggml.Tensor) -> ggml.Tensor:
        return ggml.gelu(self.ctx, a)

    def silu(self, a: ggml.Tensor) -> ggml.Tensor:
        return ggml.silu(self.ctx, a)

    def sigmoid(self, a: ggml.Tensor) -> ggml.Tensor:
        return ggml.sigmoid(self.ctx, a)

    def tanh(self, a: ggml.Tensor) -> ggml.Tensor:
        return ggml.tanh(self.ctx, a)

    def sqrt(self, a: ggml.Tensor) -> ggml.Tensor:
        return ggml.sqrt(self.ctx, a)

    def sqr(self, a: ggml.Tensor) -> ggml.Tensor:
        return ggml.sqr(self.ctx, a)

    def log(self, a: ggml.Tensor) -> ggml.Tensor:
        return ggml.log(self.ctx, a)

    def exp(self, a: ggml.Tensor) -> ggml.Tensor:
        return ggml.exp(self.ctx, a)

    def neg(self, a: ggml.Tensor) -> ggml.Tensor:
        return ggml.neg(self.ctx, a)

    def abs(self, a: ggml.Tensor) -> ggml.Tensor:
        return ggml.abs(self.ctx, a)

    def soft_max(self, a: ggml.Tensor) -> ggml.Tensor:
        return ggml.soft_max(self.ctx, a)

    def transpose(self, a: ggml.Tensor) -> ggml.Tensor:
        return ggml.transpose(self.ctx, a)

    def cont(self, a: ggml.Tensor) -> ggml.Tensor:
        return ggml.cont(self.ctx, a)

    def leaky_relu(self, a: ggml.Tensor, negative_slope: float = 0.01) -> ggml.Tensor:
        return ggml.leaky_relu(self.ctx, a, negative_slope, False)

    # -- parameterized ops ----------------------------------------------------

    def norm(self, a: ggml.Tensor, eps: float = 1e-5) -> ggml.Tensor:
        return ggml.norm(self.ctx, a, eps)

    def rms_norm(self, a: ggml.Tensor, eps: float = 1e-5) -> ggml.Tensor:
        return ggml.rms_norm(self.ctx, a, eps)

    def scale(self, a: ggml.Tensor, s: float) -> ggml.Tensor:
        return ggml.scale(self.ctx, a, s)

    def clamp(self, a: ggml.Tensor, min: float, max: float) -> ggml.Tensor:
        return ggml.clamp(self.ctx, a, min, max)

    # -- reduction ops --------------------------------------------------------

    def sum(self, a: ggml.Tensor) -> ggml.Tensor:
        return ggml.sum(self.ctx, a)

    def sum_rows(self, a: ggml.Tensor) -> ggml.Tensor:
        return ggml.sum_rows(self.ctx, a)

    def mean(self, a: ggml.Tensor) -> ggml.Tensor:
        return ggml.mean(self.ctx, a)

    def argmax(self, a: ggml.Tensor) -> ggml.Tensor:
        return ggml.argmax(self.ctx, a)

    # -- reshape ops ----------------------------------------------------------

    def reshape(self, a: ggml.Tensor, *shape: int) -> ggml.Tensor:
        """Reshape tensor, dispatching to ``reshape_{1..4}d`` by *shape* length."""
        ndim = len(shape)
        if ndim < 1 or ndim > 4:
            raise ValueError(f"reshape supports 1-4 dims, got {ndim}")
        fn = getattr(ggml, f"reshape_{ndim}d")
        return fn(self.ctx, a, *shape)

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        """Free the underlying context. Safe to call multiple times."""
        if self._context:
            self._context.close()
            self._context = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> Graph:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class GAllocr:
    """Wraps ``ggml_gallocr`` for graph memory allocation and lifecycle management."""

    def __init__(self, backend: Backend):
        self._allocr = ggml.gallocr_new(backend.buffer_type)

    # -- core operations ------------------------------------------------------

    def reserve(self, graph: ggml.Graph) -> int:
        """Reserve memory for *graph*. Returns buffer size in bytes."""
        ggml.gallocr_reserve(self._allocr, graph)
        return ggml.gallocr_get_buffer_size(self._allocr, 0)

    def alloc(self, graph: ggml.Graph) -> None:
        """Allocate graph memory (must call :meth:`reserve` first)."""
        ggml.gallocr_alloc_graph(self._allocr, graph)

    # -- properties -----------------------------------------------------------

    @property
    def buffer_size(self) -> int:
        """Current buffer size (call after :meth:`reserve`)."""
        return ggml.gallocr_get_buffer_size(self._allocr, 0)

    # -- lifecycle ------------------------------------------------------------

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Free the graph allocator.  Safe to call multiple times."""
        if self._allocr:
            ggml.gallocr_free(self._allocr)
            self._allocr = None

    def __enter__(self) -> GAllocr:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_gguf(
    fname: str, backend: Backend
) -> tuple[ggml.Context, ggml.Buffer, dict[str, ggml.Tensor]]:
    """Load a GGUF model file and return ``(ctx_w, buf_w, tensors)``.

    The caller is responsible for freeing *ctx_w* and *buf_w* when done.
    """
    gguf_ctx, ctx_w = ggml.gguf_init_from_file(fname, no_alloc=True)
    buf_w = ggml.backend_alloc_ctx_tensors(ctx_w, backend.raw)

    n_tensors = ggml.gguf_get_n_tensors(gguf_ctx)
    data_offset = ggml.gguf_get_data_offset(gguf_ctx)

    tensors: dict[str, ggml.Tensor] = {}
    with open(fname, "rb") as f:
        for i in range(n_tensors):
            name = ggml.gguf_get_tensor_name(gguf_ctx, i)
            tensor = ggml.get_tensor(ctx_w, name)
            tensors[name] = tensor
            offset = data_offset + ggml.gguf_get_tensor_offset(gguf_ctx, i)
            nbytes = ggml.nbytes(tensor)
            f.seek(offset)
            data = np.frombuffer(f.read(nbytes), dtype=np.uint8)
            ggml.backend_tensor_set(tensor, data, 0, nbytes)

    ggml.gguf_free(gguf_ctx)
    return ctx_w, buf_w, tensors

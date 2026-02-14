"""Unified session that wraps Backend, Context, and GraphRunner with automatic resource tracking."""

from __future__ import annotations

import numpy as np

from ggbond import ggml
from ggbond.backend import Backend
from ggbond.context import Context
from ggbond.graph import GraphRunner, load_gguf as _load_gguf


class Session:
    """Single entry-point that owns a backend, tracks all resources, and cleans up on exit.

    Usage::

        with Session("cpu") as s:
            ctx = s.context(n_tensors=2)
            t = ctx.new_tensor(ggml.Type.F32, 4, 4)
            s.alloc(ctx)
            s.set(t, data)
            ...
            s.run(graph)
            out = s.get(graph, t)
    """

    def __init__(self, backend: str = "cpu", *, n_threads: int = 4):
        self._backend = Backend(backend, n_threads=n_threads)
        self._runner = GraphRunner(self._backend)
        self._contexts: list[Context] = []
        self._buffers: list[ggml.Buffer] = []
        self._reserved = False

    # -- properties (escape hatches) ------------------------------------------

    @property
    def backend(self) -> Backend:
        """Underlying :class:`Backend` for direct access."""
        return self._backend

    @property
    def runner(self) -> GraphRunner:
        """Underlying :class:`GraphRunner` for direct access."""
        return self._runner

    # -- context creation -----------------------------------------------------

    def context(self, n_tensors: int, *, no_alloc: bool = True, track: bool = True) -> Context:
        """Create a tensor context. Tracked by default for automatic cleanup."""
        ctx = Context(n_tensors=n_tensors, no_alloc=no_alloc)
        if track:
            self._contexts.append(ctx)
        return ctx

    def graph_context(
        self, max_nodes: int | None = None, *, grads: bool = False, track: bool = True
    ) -> Context:
        """Create a graph-sized context. Tracked by default for automatic cleanup."""
        ctx = Context.for_graph(max_nodes=max_nodes, grads=grads)
        if track:
            self._contexts.append(ctx)
        return ctx

    # -- allocation -----------------------------------------------------------

    def alloc(self, ctx: Context) -> ggml.Buffer:
        """Allocate tensors in *ctx* on the backend. The buffer is tracked."""
        buf = self._backend.alloc_ctx(ctx)
        self._buffers.append(buf)
        return buf

    # -- data transfer --------------------------------------------------------

    def set(self, tensor: ggml.Tensor, data: np.ndarray) -> None:
        """Copy *data* into *tensor*."""
        self._backend.tensor_set(tensor, data)

    def get(self, graph: ggml.Graph, tensor_or_name, dtype=np.float32) -> np.ndarray:
        """Read full tensor data from the graph."""
        return self._runner.get(graph, tensor_or_name, dtype=dtype)

    def get_slice(
        self, graph: ggml.Graph, name: str, offset: int, count: int, dtype=np.float32
    ) -> np.ndarray:
        """Read *count* elements from tensor *name* starting at byte *offset*."""
        return self._runner.get_slice(graph, name, offset, count, dtype=dtype)

    # -- graph execution ------------------------------------------------------

    def reserve(self, graph: ggml.Graph) -> int:
        """Explicitly reserve memory for *graph*. Returns buffer size in bytes."""
        size = self._runner.reserve(graph)
        self._reserved = True
        return size

    def run(self, graph: ggml.Graph, inputs: dict[str, np.ndarray] | None = None) -> None:
        """Reserve (if needed) and compute *graph*.

        On first call, automatically reserves memory.  Subsequent calls only compute.
        """
        if not self._reserved:
            self._runner.reserve(graph)
            self._reserved = True
        self._runner.compute(graph, inputs=inputs)

    # -- GGUF -----------------------------------------------------------------

    def load_gguf(self, fname: str) -> tuple[Context, dict[str, ggml.Tensor]]:
        """Load a GGUF model. The context and buffer are tracked automatically.

        Returns ``(ctx_w, tensors)`` — the buffer is managed internally.
        """
        ctx_w, buf_w, tensors = _load_gguf(fname, self._backend)
        # Wrap the raw ctx_w in a Context so we can track it
        wrapper = object.__new__(Context)
        wrapper._ctx = ctx_w
        self._contexts.append(wrapper)
        self._buffers.append(buf_w)
        return wrapper, tensors

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        """Release all resources in reverse order. Safe to call multiple times."""
        # 1. GraphRunner
        if self._runner:
            self._runner.close()
            self._runner = None

        # 2. Contexts (reverse order)
        for ctx in reversed(self._contexts):
            ctx.close()
        self._contexts.clear()

        # 3. Buffers (reverse order)
        for buf in reversed(self._buffers):
            ggml.backend_buffer_free(buf)
        self._buffers.clear()

        # 4. Backend
        if self._backend:
            self._backend.close()
            self._backend = None

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

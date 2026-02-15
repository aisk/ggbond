"""Unified session that wraps Backend, Context, and GraphRunner with automatic resource tracking."""

from __future__ import annotations

import numpy as np

from ggbond import ggml
from ggbond.backend import Backend
from ggbond.context import Context
from ggbond.graph import GAllocr, Graph
from ggbond.gguf import load_gguf as _load_gguf


class Session:
    """Single entry-point that owns a backend, tracks all resources, and cleans up on exit."""

    def __init__(self, backend: str = "cpu", *, n_threads: int = 4):
        self._backend = Backend(backend, n_threads=n_threads)
        self._allocr = GAllocr(self._backend)
        self._contexts: list[Context] = []
        self._buffers: list[ggml.Buffer] = []

    # -- properties (escape hatches) ------------------------------------------

    @property
    def backend(self) -> Backend:
        """Underlying :class:`Backend` for direct access."""
        return self._backend

    @property
    def buffer_size(self) -> int:
        """Current graph allocator buffer size (call after :meth:`reserve`)."""
        return self._allocr.buffer_size

    # -- context creation -----------------------------------------------------

    def context(self, n_tensors: int, *, no_alloc: bool = True, track: bool = True) -> Context:
        """Create a tensor context. Tracked by default for automatic cleanup."""
        ctx = Context(n_tensors=n_tensors, no_alloc=no_alloc)
        if track:
            self._contexts.append(ctx)
        return ctx

    def graph(
        self, max_nodes: int | None = None, *, grads: bool = False, track: bool = True
    ) -> Graph:
        """Create a :class:`Graph` (with its own internal context).

        Tracked by default for automatic cleanup.
        """
        g = Graph(max_nodes=max_nodes, grads=grads)
        if track:
            self._contexts.append(g._context)
        return g

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

    def get(self, graph: ggml.Graph | Graph, tensor_or_name, dtype=np.float32) -> np.ndarray:
        """Read full tensor data from the graph.

        *tensor_or_name* can be a ``ggml.Tensor`` object or a string name.
        """
        raw = self._raw(graph)
        tensor = (
            ggml.graph_get_tensor(raw, tensor_or_name)
            if isinstance(tensor_or_name, str)
            else tensor_or_name
        )
        return self._backend.tensor_get(tensor, dtype)

    def get_slice(
        self, graph: ggml.Graph | Graph, name: str, offset: int, count: int, dtype=np.float32
    ) -> np.ndarray:
        """Read *count* elements from tensor *name* starting at byte *offset*."""
        raw = self._raw(graph)
        tensor = ggml.graph_get_tensor(raw, name)
        return self._backend.tensor_get_slice(tensor, offset, count, dtype)

    # -- graph execution ------------------------------------------------------

    def reserve(self, graph: ggml.Graph | Graph) -> int:
        """Explicitly reserve memory for *graph*. Returns buffer size in bytes."""
        return self._allocr.reserve(self._raw(graph))

    def run(self, graph: ggml.Graph | Graph, inputs: dict[str, np.ndarray] | None = None) -> None:
        """Allocate, set inputs, and compute *graph*.

        Caller must call :meth:`reserve` before the first :meth:`run`.
        """
        raw = self._raw(graph)
        self._allocr.alloc(raw)
        if inputs:
            for name, data in inputs.items():
                tensor = ggml.graph_get_tensor(raw, name)
                self._backend.tensor_set(tensor, data)
        self._backend.compute(raw)

    @staticmethod
    def _raw(graph: ggml.Graph | Graph) -> ggml.Graph:
        """Resolve a Graph or ggml.Graph to the underlying ggml.Graph."""
        return graph.raw if isinstance(graph, Graph) else graph

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

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Release all resources in reverse order. Safe to call multiple times."""
        # 1. GAllocr
        if self._allocr:
            self._allocr.close()
            self._allocr = None

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

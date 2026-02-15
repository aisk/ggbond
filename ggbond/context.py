"""GGML context wrapper with automatic lifecycle management."""

from __future__ import annotations

from ggbond import ggml


class Context:
    """Wraps a ggml_context with context-manager support."""

    def __init__(self, n_tensors: int, *, no_alloc: bool = True):
        size = ggml.tensor_overhead() * n_tensors
        self._ctx = ggml.context_init(size, no_alloc=no_alloc)

    @classmethod
    def _for_graph(
        cls, max_nodes: int, *, grads: bool = False
    ) -> Context:
        """Create a context sized for holding a computation graph (internal)."""
        size = (
            ggml.tensor_overhead() * max_nodes
            + ggml.graph_overhead_custom(max_nodes, grads)
        )
        obj = object.__new__(cls)
        obj._ctx = ggml.context_init(size, no_alloc=True)
        return obj

    # -- properties ----------------------------------------------------------

    @property
    def raw(self) -> ggml.Context:
        """Underlying opaque ``ggml.Context`` pointer."""
        return self._ctx

    # -- tensor helpers -------------------------------------------------------

    def new_tensor(
        self, dtype, *shape: int, name: str | None = None
    ) -> ggml.Tensor:
        """Create a tensor, dispatching to ``new_tensor_{1..4}d`` by *shape* length."""
        ndim = len(shape)
        if ndim == 1:
            t = ggml.new_tensor_1d(self._ctx, dtype, shape[0])
        elif ndim == 2:
            t = ggml.new_tensor_2d(self._ctx, dtype, shape[0], shape[1])
        elif ndim == 3:
            t = ggml.new_tensor_3d(self._ctx, dtype, shape[0], shape[1], shape[2])
        elif ndim == 4:
            t = ggml.new_tensor_4d(self._ctx, dtype, shape[0], shape[1], shape[2], shape[3])
        else:
            raise ValueError(f"unsupported number of dimensions: {ndim}")
        if name is not None:
            ggml.set_name(t, name)
        return t

    def _new_graph(
        self, max_nodes: int, *, grads: bool = False
    ) -> ggml.Graph:
        """Create a computation graph inside this context (internal)."""
        return ggml.new_graph_custom(self._ctx, max_nodes, grads)

    # -- lifecycle ------------------------------------------------------------

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Free the underlying context.  Safe to call multiple times."""
        if self._ctx:
            ggml.context_free(self._ctx)
            self._ctx = None

    def __enter__(self) -> Context:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

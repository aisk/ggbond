"""Thin OO wrapper around ``ggml_cgraph``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import ggml as _ggml

from .errors import check_status, nonnull
from .tensor import Tensor

if TYPE_CHECKING:
    from .context import Context


def graph_overhead_custom(size: int, grads: bool = False) -> int:
    """Return bytes needed for a custom graph (``ggml_graph_overhead_custom``)."""
    return _ggml.ggml_graph_overhead_custom(size, grads)


class Graph:
    """A computation graph backed by an external :class:`Context`.

    Following ggml, ``ggml_new_graph`` needs a context to allocate the graph
    structure; the graph holds (and keeps alive) the context that also owns its
    nodes. A graph has no independent ``close()`` -- its memory is released when
    the backing context is freed.
    """

    def __init__(self, ctx: "Context", *, size: Optional[int] = None, grads: bool = False):
        self.ctx = ctx
        if size is None and not grads:
            ptr = _ggml.ggml_new_graph(ctx.ptr)
        else:
            ptr = _ggml.ggml_new_graph_custom(
                ctx.ptr, size or _ggml.GGML_DEFAULT_GRAPH_SIZE, grads
            )
        self.ptr = nonnull(ptr, "ggml_new_graph")

    def build_forward_expand(self, tensor: Tensor) -> "Graph":
        """Add ``tensor`` (and its dependencies) to the forward graph."""
        _ggml.ggml_build_forward_expand(self.ptr, tensor.ptr)
        return self

    def compute_with_ctx(self, n_threads: int = 1) -> None:
        """Compute on the CPU using the backing context's memory.

        Convenience path for ``no_alloc=False`` contexts; for backend-resident
        tensors use :meth:`Backend.compute` instead.
        """
        status = _ggml.ggml_graph_compute_with_ctx(self.ctx.ptr, self.ptr, n_threads)
        check_status(status, "ggml_graph_compute_with_ctx")

    # -- node access (no dedicated bindings; read the struct fields) --------

    @property
    def n_nodes(self) -> int:
        return self.ptr.contents.n_nodes

    def node(self, i: int) -> Tensor:
        if not (0 <= i < self.n_nodes):
            raise IndexError(i)
        return Tensor(self.ctx, self.ptr.contents.nodes[i])

    def get_tensor(self, name: str) -> Optional[Tensor]:
        ptr = _ggml.ggml_graph_get_tensor(self.ptr, name.encode())
        return Tensor(self.ctx, ptr) if ptr else None

    def __len__(self) -> int:
        return self.n_nodes

    def __iter__(self):
        return (self.node(i) for i in range(self.n_nodes))

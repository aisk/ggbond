"""Thin OO wrapper around ``ggml_cgraph``."""

from __future__ import annotations

import ctypes
import struct
from typing import TYPE_CHECKING, Iterator, Optional

import ggml as _ggml

from .errors import GGMLError, check_status, nonnull
from .tensor import Tensor

if TYPE_CHECKING:
    from .context import Context


_POINTER_SIZE = ctypes.sizeof(ctypes.c_void_p)


def graph_overhead_custom(size: int, grads: bool = False) -> int:
    """Return bytes needed for a custom graph (``ggml_graph_overhead_custom``)."""
    return _ggml.ggml_graph_overhead_custom(size, grads)


class Graph:
    """A computation graph backed by an external :class:`Context`.

    Following ggml, ``ggml_new_graph`` needs a context to allocate the graph
    structure; the graph holds (and keeps alive) that context. A graph has no
    independent ``close()`` -- its memory is released when the backing context
    is freed.

    As in ggml, the graph keeps two lists: :meth:`nodes` are the tensors to
    compute (ops and parameters), :meth:`leafs` the constants and inputs they
    read. Iterating the graph or calling ``len`` on it walks the nodes.
    """

    def __init__(self, ctx: "Context", *, size: Optional[int] = None, grads: bool = False):
        self.ctx = ctx
        # ggml_new_graph_custom aborts when the graph does not fit the arena.
        ctx._reserve(
            graph_overhead_custom(size or _ggml.GGML_DEFAULT_GRAPH_SIZE, grads),
            "new_graph",
        )
        if size is None and not grads:
            ptr = _ggml.ggml_new_graph(ctx.ptr)
        else:
            ptr = _ggml.ggml_new_graph_custom(
                ctx.ptr, size or _ggml.GGML_DEFAULT_GRAPH_SIZE, grads
            )
        self._ptr = nonnull(ptr, "ggml_new_graph")
        self._epoch = ctx._epoch
        self.grads = grads

    @property
    def ptr(self):
        """The ``ggml_cgraph *``, or :class:`GGMLError` once its context is closed or reset."""
        epoch = self._epoch
        if epoch is None or epoch is not self.ctx._epoch:
            raise GGMLError(
                "graph used after its context was "
                + ("closed" if self.ctx._ptr is None else "reset")
            )
        return self._ptr

    # -- building -----------------------------------------------------------

    def build_forward_expand(self, tensor: Tensor) -> "Graph":
        """Add ``tensor`` (and its dependencies) to the forward graph."""
        if tensor.ctx is not self.ctx:
            self.ctx._deps.add(tensor.ctx)
        _ggml.ggml_build_forward_expand(self.ptr, tensor.ptr)
        return self

    def build_backward_expand(self) -> "Graph":
        """Add the backward pass to this graph (``ggml_build_backward_expand``).

        The graph must have been created with ``grads=True`` and already hold
        the forward pass of a tensor marked with :meth:`Tensor.set_loss`, built
        from leaves marked with :meth:`Tensor.set_param`. Gradient tensors are
        created in the backing context, so size it for them; call :meth:`reset`
        after allocating and before each compute, then read gradients with
        :meth:`get_grad`.
        """
        if not self.grads:
            raise GGMLError("build_backward_expand: create the graph with grads=True")
        _ggml.ggml_build_backward_expand(self.ctx.ptr, self.ptr, None)
        return self

    def get_grad(self, tensor: Tensor) -> Optional[Tensor]:
        """Return the gradient of ``tensor`` in this graph, or ``None``."""
        if not self.grads:
            return None
        ptr = _ggml.ggml_graph_get_grad(self.ptr, tensor.ptr)
        return self.ctx._adopt(ptr) if ptr else None

    def reset(self) -> None:
        """Zero the gradients and seed the loss gradient (``ggml_graph_reset``).

        Needs the graph's tensors to be allocated already.
        """
        if self.grads:
            _ggml.ggml_graph_reset(self.ptr)

    def clear(self) -> None:
        """Remove every node and leaf so the graph can be rebuilt (``ggml_graph_clear``)."""
        _ggml.ggml_graph_clear(self.ptr)

    # -- compute ------------------------------------------------------------

    def _check_allocated(self, what: str) -> None:
        """Raise unless every node has memory to be computed into.

        ggml dereferences the data pointers without looking, so computing a
        graph that was never allocated, or whose allocator has since moved on
        to another graph or closed, would crash the process.
        """
        cgraph = self.ptr.contents
        n = cgraph.n_nodes
        if n <= 0:
            return
        base = ctypes.cast(cgraph.nodes, ctypes.c_void_p).value
        tensor_at = _ggml.ggml_tensor.from_address
        for address in struct.unpack(f"{n}P", ctypes.string_at(base, n * _POINTER_SIZE)):
            tensor = tensor_at(address)
            if not tensor.data:
                name = bytes(tensor.name).split(b"\0", 1)[0].decode(errors="replace")
                raise GGMLError(
                    f"{what}: graph tensor {name!r} is not allocated. Allocate "
                    "the graph first (GAllocr.alloc_graph); a graph loses its "
                    "memory when its allocator allocates another graph or closes"
                )

    def compute_with_ctx(self, n_threads: int = 1) -> None:
        """Compute on the CPU using the backing context's memory.

        Convenience path for ``no_alloc=False`` contexts; for backend-resident
        tensors use :meth:`Backend.compute` instead.
        """
        status = _ggml.ggml_graph_compute_with_ctx(self.ctx.ptr, self.ptr, n_threads)
        check_status(status, "ggml_graph_compute_with_ctx")

    # -- node access (read from the struct fields) --------------------------
    #
    # A graph may reference tensors living in other contexts, so wrappers are
    # resolved to the context that really owns each tensor (Context._adopt).

    @property
    def size(self) -> int:
        """Node capacity the graph was created with (``ggml_graph_size``)."""
        return self.ptr.contents.size

    @property
    def n_nodes(self) -> int:
        return self.ptr.contents.n_nodes

    @property
    def n_leafs(self) -> int:
        return self.ptr.contents.n_leafs

    def node(self, i: int) -> Tensor:
        """Return node ``i``; negative indices count from the end, as in ggml."""
        n = self.n_nodes
        if not (-n <= i < n):
            raise IndexError(i)
        return self.ctx._adopt(self.ptr.contents.nodes[i % n])

    def leaf(self, i: int) -> Tensor:
        n = self.n_leafs
        if not (0 <= i < n):
            raise IndexError(i)
        return self.ctx._adopt(self.ptr.contents.leafs[i])

    def nodes(self) -> Iterator[Tensor]:
        """Iterate over the nodes, in evaluation order."""
        return (self.node(i) for i in range(self.n_nodes))

    def leafs(self) -> Iterator[Tensor]:
        """Iterate over the leafs (constants and inputs)."""
        return (self.leaf(i) for i in range(self.n_leafs))

    def get_tensor(self, name: str) -> Optional[Tensor]:
        """Return the node or leaf named ``name``, or ``None`` (``ggml_graph_get_tensor``)."""
        cgraph = self.ptr
        # ggml compares names by reading every tensor of the graph, including
        # those of other contexts, which must therefore still exist.
        seen, stack = set(), list(self.ctx._deps)
        while stack:
            ctx = stack.pop()
            if ctx in seen:
                continue
            seen.add(ctx)
            if ctx._ptr is None:
                raise GGMLError(
                    "get_tensor: the graph reads tensors of a context that is "
                    "closed or unknown to this one"
                )
            stack.extend(ctx._deps)
        ptr = _ggml.ggml_graph_get_tensor(cgraph, name.encode())
        return self.ctx._adopt(ptr) if ptr else None

    # -- debugging ----------------------------------------------------------

    def print(self) -> None:
        """Print the graph's nodes to stderr (``ggml_graph_print``)."""
        _ggml.ggml_graph_print(self.ptr)

    def dump_dot(self, path: str, *, forward: Optional["Graph"] = None) -> None:
        """Write this graph to ``path`` as Graphviz dot (``ggml_graph_dump_dot``).

        ggml renders the graph passed as its first argument, which is this one.
        ``forward`` is the corresponding forward graph, used only to highlight
        nodes shared with it when dumping a gradient graph.
        """
        _ggml.ggml_graph_dump_dot(
            self.ptr,
            forward.ptr if forward is not None else None,
            str(path).encode(),
        )

    def __len__(self) -> int:
        return self.n_nodes

    def __iter__(self) -> Iterator[Tensor]:
        return self.nodes()

    def __repr__(self) -> str:
        if self._epoch is not self.ctx._epoch:
            return "<ggml.Graph closed>"
        return f"<ggml.Graph n_nodes={self.n_nodes} n_leafs={self.n_leafs}>"

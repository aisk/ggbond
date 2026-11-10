"""Thin OO wrapper around ``ggml_gallocr`` (graph memory allocator)."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import ggml as _ggml

from .buffer import LIVE_BUFFERS, BufferType, detach, mark_freed, mark_live
from .errors import GGMLError, nonnull
from .handle import Handle

if TYPE_CHECKING:
    from .backend import Backend
    from .graph import Graph


# Every open allocator, so one allocator can tell its own buffers from those
# of the others when several allocate tensors of the same graph.
_ALLOCATORS = weakref.WeakSet()


class GraphAllocator(Handle):
    """Shared bookkeeping of :class:`GAllocr` and :class:`Scheduler`.

    Both own internal backend buffers that hold the graph tensors they
    allocate, and both free or reallocate those buffers without ggml telling
    the tensors. This base remembers which graphs were allocated and, when the
    buffers go away (on ``close()``, or when a larger graph forces a
    reallocation), clears the ``buffer`` / ``data`` fields of the tensors that
    lived in them. Such tensors are then unallocated again: reading them raises
    :class:`GGMLError`, and ggml allocates them afresh if the graph is
    allocated again.

    ggml exposes the buffer sizes but not the buffers, so the addresses are
    learned from the first graph allocated after the sizes change (ggml only
    reallocates when a buffer has to grow).
    """

    def _init_tracking(self) -> None:
        # id(graph) -> (weak graph, [(context, epoch), ...] it spans)
        self._graphs = {}
        self._sizes = None
        self._addresses = set()
        _ALLOCATORS.add(self)

    def _buffer_sizes(self) -> tuple:
        raise NotImplementedError

    @staticmethod
    def _spanned(graph: "Graph") -> list:
        """Every context ``graph`` may hold tensors of, with its current epoch."""
        seen, stack, spanned = set(), [graph.ctx], []
        while stack:
            ctx = stack.pop()
            if ctx in seen:
                continue
            seen.add(ctx)
            spanned.append((ctx, ctx._epoch))
            stack.extend(ctx._deps)
        return spanned

    def _live_graphs(self, exclude=None):
        """Yield tracked graphs whose tensors all still exist; forget the rest."""
        for key, (ref, spanned) in list(self._graphs.items()):
            graph = ref()
            if graph is None or any(
                epoch is None or ctx._epoch is not epoch for ctx, epoch in spanned
            ):
                del self._graphs[key]
            elif graph is not exclude:
                yield graph

    @staticmethod
    def _tensors(graph: "Graph"):
        cgraph = graph._ptr.contents
        for i in range(cgraph.n_nodes):
            yield cgraph.nodes[i]
        for i in range(cgraph.n_leafs):
            yield cgraph.leafs[i]

    def _release(self, exclude=None) -> None:
        """Detach every tracked tensor living in this allocator's buffers."""
        addresses = self._addresses
        if addresses:
            for graph in self._live_graphs(exclude):
                for tensor in self._tensors(graph):
                    if tensor.contents.buffer in addresses:
                        detach(tensor)
            for address in addresses:
                mark_freed(address)
        self._addresses = set()
        self._sizes = None

    def _reserved(self) -> None:
        """Call after a reserve: a grown buffer invalidates what lived in the old one."""
        if self._sizes is not None and self._buffer_sizes() != self._sizes:
            self._release()

    def _allocated(self, graph: "Graph") -> None:
        """Call after ``graph`` was allocated by this allocator."""
        sizes = self._buffer_sizes()
        if sizes != self._sizes:
            # The buffers were (re)allocated: whatever earlier graphs kept in
            # the old ones is gone, and the new addresses must be learned.
            self._release(exclude=graph)
            foreign = set()
            for other in _ALLOCATORS:
                if other is not self:
                    foreign |= other._addresses
            addresses = set()
            for tensor in self._tensors(graph):
                address = tensor.contents.buffer
                if address and address not in foreign and address not in LIVE_BUFFERS:
                    addresses.add(address)
            for address in addresses:
                mark_live(address)
            self._addresses = addresses
            self._sizes = sizes
        self._graphs[id(graph)] = (weakref.ref(graph), self._spanned(graph))
        if len(self._graphs) > 16:
            for _ in self._live_graphs():
                pass

    def _free(self, ptr) -> None:
        self._release()
        self._graphs = {}
        _ALLOCATORS.discard(self)


class GAllocr(GraphAllocator):
    """Allocates the working memory for a computation graph from a backend
    buffer type. Use :meth:`reserve` to pre-compute the requirement, then
    :meth:`alloc_graph` before computing.

    The allocator owns the memory holding every tensor it allocated, which is
    every graph node that had no buffer of its own. Download results before the
    allocator goes away: once it is closed (or leaves its ``with`` block) those
    tensors are unallocated again and reading them raises :class:`GGMLError`.
    The same happens to the tensors of earlier graphs when a larger graph makes
    the allocator grow its buffer.
    """

    def __init__(self, buffer_type):
        """``buffer_type`` is a :class:`BufferType` or a raw
        ``ggml_backend_buffer_type_t``.
        """
        if isinstance(buffer_type, BufferType):
            buffer_type = buffer_type.ptr
        self._ptr = nonnull(_ggml.ggml_gallocr_new(buffer_type), "ggml_gallocr_new")
        self._backend = None
        self._init_tracking()

    @classmethod
    def from_backend(cls, backend: "Backend") -> "GAllocr":
        """Create an allocator on ``backend``'s default buffer type.

        The allocator keeps the backend alive, and closing the backend closes
        the allocator first.
        """
        self = cls(backend.buffer_type)
        self._backend = backend
        backend._track(self)
        return self

    def _buffer_sizes(self) -> tuple:
        return (_ggml.ggml_gallocr_get_buffer_size(self._ptr, 0),)

    def reserve(self, graph: "Graph") -> None:
        if not _ggml.ggml_gallocr_reserve(self.ptr, graph.ptr):
            raise GGMLError("ggml_gallocr_reserve failed")
        self._reserved()

    def alloc_graph(self, graph: "Graph") -> None:
        if not _ggml.ggml_gallocr_alloc_graph(self.ptr, graph.ptr):
            raise GGMLError("ggml_gallocr_alloc_graph failed")
        self._allocated(graph)

    def buffer_size(self, buffer_id: int = 0) -> int:
        return _ggml.ggml_gallocr_get_buffer_size(self.ptr, buffer_id)

    def _free(self, ptr) -> None:
        super()._free(ptr)
        _ggml.ggml_gallocr_free(ptr)
        self._backend = None

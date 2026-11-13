"""Thin OO wrapper around ``ggml_gallocr`` (graph memory allocator)."""

from __future__ import annotations

import ctypes
import struct
import weakref
from typing import TYPE_CHECKING

import ggml as _ggml

from .buffer import LIVE_BUFFERS, BufferType, address_of, mark_freed, mark_live
from .errors import GGMLError, nonnull
from .handle import Handle

if TYPE_CHECKING:
    from .backend import Backend
    from .graph import Graph


# Every open allocator, so one allocator can tell its own buffers from those
# of the others when several allocate tensors of the same graph.
_ALLOCATORS = weakref.WeakSet()

_TENSOR = _ggml.ggml_tensor
_POINTER_SIZE = ctypes.sizeof(ctypes.c_void_p)


def _pointers(array, n: int) -> bytes:
    """The raw bytes of a C array of ``n`` pointers (``b""`` when empty)."""
    address = address_of(array) if n > 0 else None
    return ctypes.string_at(address, n * _POINTER_SIZE) if address else b""


class GraphAllocator(Handle):
    """Shared bookkeeping of :class:`GAllocr` and :class:`Scheduler`.

    Both own internal backend buffers that hold the graph tensors they
    allocate, and both reuse, free or reallocate those buffers without ggml
    telling the tensors. An allocator therefore holds **one graph at a time**:
    this base remembers the tensors it allocated for that graph and clears
    their ``buffer`` / ``data`` fields (``buffer.detach``) as soon as the
    memory stops being theirs, which is

    * right before ggml is asked to allocate a *different* graph (another
      graph, or the same one after nodes were added to it), since the new graph
      is laid out over the same memory;
    * when a larger ``reserve()`` regrows the buffers;
    * on ``close()``.

    Such tensors are then unallocated again: reading them raises
    :class:`GGMLError`, and ggml allocates them afresh when a graph containing
    them is allocated, so a tensor may be shared by successive graphs (upload
    its data again). Detaching *before* the allocation matters: ggml takes a
    tensor that still has data for one allocated elsewhere and leaves it where
    it was. Allocating the unchanged current graph again detaches nothing, and
    tensors living in any other buffer (``Backend.alloc_ctx_tensors``) are
    never touched.

    The tensors are remembered by address, grouped by the context owning them,
    not through the :class:`Graph`: the graph wrapper may be dropped, and its
    context reset or closed, while tensors of other contexts that were
    allocated for it live on.

    ggml exposes the buffer sizes but not the buffers, so the addresses are
    learned from the first graph allocated after the sizes change (ggml only
    reallocates when a buffer has to grow).

    Subclasses call :meth:`_allocating` before and :meth:`_allocated` after
    every ggml call that allocates a graph, and :meth:`_reserved` after a
    reserve.
    """

    def _init_tracking(self) -> None:
        # Identity of the graph whose tensors currently occupy the buffers
        # (see _key), and the tensors allocated for it as a list of
        # (weak context, epoch, [tensor address, ...]).
        self._key = None
        self._held = None
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

    @staticmethod
    def _key_of(graph: "Graph") -> tuple:
        """What tells one allocation request from another: the ``ggml_cgraph``
        and the exact node and leaf pointers it lists.
        """
        cgraph = graph.ptr.contents
        return (
            address_of(graph.ptr),
            _pointers(cgraph.nodes, cgraph.n_nodes),
            _pointers(cgraph.leafs, cgraph.n_leafs),
        )

    def _snapshot(self, key: tuple, graph: "Graph") -> list:
        """Group the tensors of ``graph`` living in this allocator's buffers
        by the context that owns them.
        """
        held, arenas = [], []
        for ctx, epoch in self._spanned(graph):
            if ctx._ptr is None:
                continue
            tensors = []
            held.append((weakref.ref(ctx), epoch, tensors))
            base = _ggml.ggml_get_mem_buffer(ctx._ptr)
            if base is not None:
                arenas.append((base, base + ctx.mem_size, tensors))
        own = self._addresses
        if own:
            for raw in key[1:]:
                for address in struct.unpack(f"{len(raw) // _POINTER_SIZE}P", raw):
                    if _TENSOR.from_address(address).buffer in own:
                        for low, high, tensors in arenas:
                            if low <= address < high:
                                tensors.append(address)
                                break
        return held

    def _intact(self) -> bool:
        """Whether every context of the held graph still has those tensors."""
        for ref, epoch, _ in self._held:
            ctx = ref()
            if ctx is None or epoch is None or ctx._epoch is not epoch:
                return False
        return True

    def _retire(self) -> None:
        """Detach the tensors of the held graph that still exist and still
        live in this allocator's buffers, and hold no graph.
        """
        held, self._held, self._key = self._held, None, None
        own = self._addresses
        if not held or not own:
            return
        for ref, epoch, tensors in held:
            ctx = ref()
            # A context that was reset or closed took its tensors with it.
            if ctx is None or epoch is None or ctx._epoch is not epoch:
                continue
            for address in tensors:
                tensor = _TENSOR.from_address(address)
                if tensor.buffer in own:
                    # buffer.detach, without building a pointer per tensor.
                    tensor.buffer = None
                    tensor.data = None

    def _release(self) -> None:
        """The buffers are gone: detach what lived in them and forget them."""
        self._retire()
        for address in self._addresses:
            mark_freed(address)
        self._addresses = set()
        self._sizes = None

    def _reserved(self) -> None:
        """Call after a reserve: a grown buffer invalidates what lived in the old one."""
        if self._sizes is not None and self._buffer_sizes() != self._sizes:
            self._release()

    def _allocating(self, graph: "Graph") -> None:
        """Call before asking ggml to allocate ``graph``.

        Unless ``graph`` is exactly what the buffers already hold, the tensors
        held so far lose their memory to it.
        """
        if self._held is not None and not (
            self._key_of(graph) == self._key and self._intact()
        ):
            self._retire()

    def _allocated(self, graph: "Graph") -> None:
        """Call after ``graph`` was allocated by this allocator."""
        key = self._key_of(graph)
        sizes = self._buffer_sizes()
        if sizes != self._sizes:
            # The buffers were (re)allocated and the new addresses must be learned.
            regrown_under_graph = self._held is not None
            self._release()
            if regrown_under_graph:
                # ggml left the tensors it took for allocated in the old buffers.
                raise GGMLError(
                    "the allocator reallocated its buffers under the graph it "
                    "already held; its tensors were unallocated, allocate the "
                    "graph again"
                )
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
        elif key == self._key:
            # The held graph again, in the same buffers: nothing moved.
            return
        else:
            self._retire()
        self._key = key
        self._held = self._snapshot(key, graph)

    @staticmethod
    def _tensors(graph: "Graph"):
        cgraph = graph._ptr.contents
        for i in range(cgraph.n_nodes):
            yield cgraph.nodes[i]
        for i in range(cgraph.n_leafs):
            yield cgraph.leafs[i]

    def _free(self, ptr) -> None:
        self._release()
        _ALLOCATORS.discard(self)


class GAllocr(GraphAllocator):
    """Allocates the working memory for a computation graph from a backend
    buffer type. Use :meth:`reserve` to pre-compute the requirement, then
    :meth:`alloc_graph` before computing.

    The allocator owns the memory holding every tensor it allocated, which is
    every graph node that had no buffer of its own. Download results before the
    allocator goes away: once it is closed (or leaves its ``with`` block) those
    tensors are unallocated again and reading them raises :class:`GGMLError`.

    The allocator holds one graph at a time. :meth:`alloc_graph` on a different
    graph (or on the same one after nodes were added) lays it out over the same
    memory, so the tensors allocated for the previous graph are unallocated
    first: download its results before that, and upload again any input the
    two graphs share. Calling :meth:`alloc_graph` on the unchanged graph again
    keeps its tensors and their data. A :meth:`reserve` that grows the buffer
    also unallocates the tensors of the current graph.
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
        """Allocate the tensors of ``graph``, which replaces the graph allocated before."""
        ptr = self.ptr
        self._allocating(graph)
        if not _ggml.ggml_gallocr_alloc_graph(ptr, graph.ptr):
            raise GGMLError("ggml_gallocr_alloc_graph failed")
        self._allocated(graph)

    def buffer_size(self, buffer_id: int = 0) -> int:
        return _ggml.ggml_gallocr_get_buffer_size(self.ptr, buffer_id)

    def _free(self, ptr) -> None:
        super()._free(ptr)
        _ggml.ggml_gallocr_free(ptr)
        self._backend = None

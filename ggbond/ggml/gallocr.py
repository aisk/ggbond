"""Thin OO wrapper around ``ggml_gallocr`` (graph memory allocator)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ggml as _ggml

from .buffer import BufferType
from .errors import GGMLError, nonnull
from .handle import Handle

if TYPE_CHECKING:
    from .backend import Backend
    from .graph import Graph


class GAllocr(Handle):
    """Allocates the working memory for a computation graph from a backend
    buffer type. Use :meth:`reserve` to pre-compute the requirement, then
    :meth:`alloc_graph` before computing.

    The allocator owns the memory holding every tensor it allocated, which is
    every graph node that had no buffer of its own. Reading such a tensor after
    :meth:`close` (or after leaving its ``with`` block) reads freed memory and
    silently yields garbage, so download results before the allocator goes away.
    """

    def __init__(self, buffer_type):
        """``buffer_type`` is a :class:`BufferType` or a raw
        ``ggml_backend_buffer_type_t``.
        """
        if isinstance(buffer_type, BufferType):
            buffer_type = buffer_type.ptr
        self._ptr = nonnull(_ggml.ggml_gallocr_new(buffer_type), "ggml_gallocr_new")
        self._backend = None

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

    def reserve(self, graph: "Graph") -> None:
        if not _ggml.ggml_gallocr_reserve(self.ptr, graph.ptr):
            raise GGMLError("ggml_gallocr_reserve failed")

    def alloc_graph(self, graph: "Graph") -> None:
        if not _ggml.ggml_gallocr_alloc_graph(self.ptr, graph.ptr):
            raise GGMLError("ggml_gallocr_alloc_graph failed")

    def buffer_size(self, buffer_id: int = 0) -> int:
        return _ggml.ggml_gallocr_get_buffer_size(self.ptr, buffer_id)

    def _free(self, ptr) -> None:
        _ggml.ggml_gallocr_free(ptr)
        self._backend = None

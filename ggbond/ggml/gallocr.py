"""Thin OO wrapper around ``ggml_gallocr`` (graph memory allocator)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ggml as _ggml

from .buffer import BufferType
from .errors import GGMLError, nonnull

if TYPE_CHECKING:
    from .backend import Backend
    from .graph import Graph


class GAllocr:
    """Allocates the working memory for a computation graph from a backend
    buffer type. Use :meth:`reserve` to pre-compute the requirement, then
    :meth:`alloc_graph` before computing.
    """

    def __init__(self, buffer_type):
        """``buffer_type`` is a :class:`BufferType` or a raw
        ``ggml_backend_buffer_type_t``.
        """
        if isinstance(buffer_type, BufferType):
            buffer_type = buffer_type.ptr
        self.ptr = nonnull(_ggml.ggml_gallocr_new(buffer_type), "ggml_gallocr_new")
        self._closed = False

    @classmethod
    def from_backend(cls, backend: "Backend") -> "GAllocr":
        return cls(backend.buffer_type())

    def reserve(self, graph: "Graph") -> None:
        if not _ggml.ggml_gallocr_reserve(self.ptr, graph.ptr):
            raise GGMLError("ggml_gallocr_reserve failed")

    def alloc_graph(self, graph: "Graph") -> None:
        if not _ggml.ggml_gallocr_alloc_graph(self.ptr, graph.ptr):
            raise GGMLError("ggml_gallocr_alloc_graph failed")

    def buffer_size(self, buffer_id: int = 0) -> int:
        return _ggml.ggml_gallocr_get_buffer_size(self.ptr, buffer_id)

    def close(self) -> None:
        if not self._closed and self.ptr is not None:
            _ggml.ggml_gallocr_free(self.ptr)
            self._closed = True
            self.ptr = None

    def __enter__(self) -> "GAllocr":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

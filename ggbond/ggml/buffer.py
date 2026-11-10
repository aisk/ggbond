"""Thin OO wrapper around ``ggml_backend_buffer_t`` and its buffer type."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import ggml as _ggml

from .errors import nonnull
from .handle import Handle

if TYPE_CHECKING:
    from .backend import Backend


class BufferType:
    """A ``ggml_backend_buffer_type_t``: where a backend allocates from.

    Obtained from :attr:`Backend.buffer_type`; passed to :class:`GAllocr` and to
    :class:`Scheduler`. Buffer types are owned by their backend, so this wrapper
    holds no lifetime of its own.
    """

    __slots__ = ("ptr",)

    def __init__(self, ptr):
        self.ptr = nonnull(ptr, "ggml_backend_buffer_type_t")

    @property
    def name(self) -> str:
        return _ggml.ggml_backend_buft_name(self.ptr).decode()

    def __repr__(self) -> str:
        return f"<ggml.BufferType name={self.name!r}>"


class Buffer(Handle):
    """Backend memory holding tensor data (``ggml_backend_buffer_t``).

    Returned by :meth:`Backend.alloc_ctx_tensors`. As in ggml, a buffer is its
    own resource: :meth:`close` (or its ``with`` block, or garbage collection)
    frees it with ``ggml_backend_buffer_free``.

    You rarely need to manage one by hand. The :class:`Context` whose tensors
    it holds keeps the buffer alive, the buffer keeps its :class:`Backend`
    alive, and a backend that is closed explicitly closes its buffers first.
    """

    def __init__(self, ptr, *, backend: "Optional[Backend]" = None):
        self._ptr = nonnull(ptr, "ggml_backend_buffer_t")
        # Kept alive for as long as this buffer is: a buffer must be freed
        # before the backend it was allocated from.
        self._backend = backend
        if backend is not None:
            backend._track(self)

    # -- queries ------------------------------------------------------------

    @property
    def name(self) -> str:
        return _ggml.ggml_backend_buffer_name(self.ptr).decode()

    @property
    def size(self) -> int:
        """Size in bytes (``ggml_backend_buffer_get_size``)."""
        return _ggml.ggml_backend_buffer_get_size(self.ptr)

    @property
    def alignment(self) -> int:
        return _ggml.ggml_backend_buffer_get_alignment(self.ptr)

    @property
    def max_size(self) -> int:
        """Largest single allocation this buffer supports."""
        return _ggml.ggml_backend_buffer_get_max_size(self.ptr)

    @property
    def is_host(self) -> bool:
        """True when the data is directly addressable by the CPU."""
        return bool(_ggml.ggml_backend_buffer_is_host(self.ptr))

    @property
    def buffer_type(self) -> "BufferType":
        return BufferType(_ggml.ggml_backend_buffer_get_type(self.ptr))

    # -- operations ---------------------------------------------------------

    def clear(self, value: int = 0) -> None:
        """Fill the whole buffer with the byte ``value``."""
        _ggml.ggml_backend_buffer_clear(self.ptr, value)

    # -- lifecycle ----------------------------------------------------------

    def _free(self, ptr) -> None:
        _ggml.ggml_backend_buffer_free(ptr)
        self._backend = None

    def __repr__(self) -> str:
        if self.closed:
            return "<ggml.Buffer closed>"
        return f"<ggml.Buffer name={self.name!r} size={self.size}>"

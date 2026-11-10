"""Thin OO wrapper around ``ggml_backend_buffer_t`` and its buffer type."""

from __future__ import annotations

import ctypes
import weakref
from typing import TYPE_CHECKING, Optional

import ggml as _ggml

from .errors import nonnull
from .handle import Handle

if TYPE_CHECKING:
    from .backend import Backend
    from .context import Context


# -- freed-buffer bookkeeping -------------------------------------------------
#
# A ``ggml_tensor`` keeps pointing at its backend buffer after the buffer is
# freed, and ggml has no way to tell. Whoever frees a buffer therefore clears
# the ``buffer`` / ``data`` fields of the tensors it knows it allocated
# (``detach``), which makes them "unallocated" again for both this wrapper and
# ggml. Tensors it cannot reach (views created in another context) are caught
# by address instead: ``_FREED`` holds the addresses of buffers freed through
# this wrapper, until a new buffer is seen at the same address.

_FREED: "set[int]" = set()

# Addresses of the open :class:`Buffer` objects, so the graph allocators can
# tell tensors living in one of these from tensors living in their own buffers.
LIVE_BUFFERS: "set[int]" = set()


def address_of(ptr) -> Optional[int]:
    """Return the integer address of a ctypes pointer (``None`` for NULL)."""
    if not ptr:
        return None
    return ptr if isinstance(ptr, int) else ctypes.cast(ptr, ctypes.c_void_p).value


def mark_live(address: int) -> None:
    _FREED.discard(address)


def mark_freed(address: int) -> None:
    _FREED.add(address)


def is_freed(address: int) -> bool:
    return address in _FREED


def detach(tensor_ptr) -> None:
    """Make a ``ggml_tensor`` unallocated: clear its ``buffer`` and ``data``."""
    contents = tensor_ptr.contents
    contents.buffer = None
    contents.data = None


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

    Closing a buffer turns the tensors it held back into unallocated ones, so
    reading or writing them raises :class:`GGMLError` instead of touching freed
    memory.
    """

    def __init__(self, ptr, *, backend: "Optional[Backend]" = None):
        self._ptr = nonnull(ptr, "ggml_backend_buffer_t")
        # Kept alive for as long as this buffer is: a buffer must be freed
        # before the backend it was allocated from.
        self._backend = backend
        if backend is not None:
            backend._track(self)
        self._address = address_of(self._ptr)
        mark_live(self._address)
        LIVE_BUFFERS.add(self._address)
        # The context whose tensors live here (set by Backend.alloc_ctx_tensors),
        # weakly, with the epoch its tensors were created in.
        self._ctx = None
        self._ctx_epoch = None

    def _bind(self, ctx: "Context") -> None:
        self._ctx = weakref.ref(ctx)
        self._ctx_epoch = ctx._epoch

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
        ctx = self._ctx() if self._ctx is not None else None
        # Only walk the context while the tensors allocated here still exist.
        if ctx is not None and ctx._epoch is not None and ctx._epoch is self._ctx_epoch:
            tensor = _ggml.ggml_get_first_tensor(ctx._ptr)
            while tensor:
                if tensor.contents.buffer == self._address:
                    detach(tensor)
                tensor = _ggml.ggml_get_next_tensor(ctx._ptr, tensor)
        _ggml.ggml_backend_buffer_free(ptr)
        LIVE_BUFFERS.discard(self._address)
        mark_freed(self._address)
        self._backend = None
        self._ctx = None

    def __repr__(self) -> str:
        if self.closed:
            return "<ggml.Buffer closed>"
        return f"<ggml.Buffer name={self.name!r} size={self.size}>"

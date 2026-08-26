"""Thin OO wrapper around ``ggml_backend_buffer_t`` and its buffer type."""

from __future__ import annotations

import ggml as _ggml

from .errors import nonnull


class BufferType:
    """A ``ggml_backend_buffer_type_t``: where a backend allocates from.

    Obtained from :meth:`Backend.buffer_type`; passed to :class:`GAllocr` and to
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


class Buffer:
    """Backend memory holding tensor data (``ggml_backend_buffer_t``).

    Returned by :meth:`Backend.alloc_ctx_tensors`. A buffer's lifetime is
    independent of the backend that allocated it: by default the allocating
    :class:`Backend` frees it on close, but you can take ownership by calling
    :meth:`release`, after which closing the buffer (or its ``with`` block) is
    what frees it.
    """

    def __init__(self, ptr, *, owner=None):
        self.ptr = nonnull(ptr, "ggml_backend_buffer_t")
        # The Backend that will free this buffer, or None once released.
        self._owner = owner
        self._closed = False

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

    def buffer_type(self) -> "BufferType":
        return BufferType(_ggml.ggml_backend_buffer_get_type(self.ptr))

    # -- operations ---------------------------------------------------------

    def clear(self, value: int = 0) -> None:
        """Fill the whole buffer with the byte ``value``."""
        _ggml.ggml_backend_buffer_clear(self.ptr, value)

    # -- lifecycle ----------------------------------------------------------

    def release(self) -> "Buffer":
        """Take ownership from the allocating backend, returning ``self``.

        After this the buffer is freed by :meth:`close` (or its ``with`` block)
        rather than when the backend closes. The buffer must still be freed
        before the backend it came from.
        """
        if self._owner is not None:
            self._owner.disown_buffer(self)
            self._owner = None
        return self

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owner is not None:
            # Still backend-owned; let the backend free it in its own close().
            self.ptr = None
            return
        if self.ptr is not None:
            _ggml.ggml_backend_buffer_free(self.ptr)
            self.ptr = None

    def __enter__(self) -> "Buffer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        if self.ptr is None:
            return "<ggml.Buffer closed>"
        return f"<ggml.Buffer name={self.name!r} size={self.size}>"

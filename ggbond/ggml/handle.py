"""Shared lifecycle for wrappers that own one native ggml handle."""

from __future__ import annotations

from .errors import GGMLError


class Handle:
    """Base class for an owned native pointer.

    Subclasses store the pointer in ``_ptr`` and implement :meth:`_free`.
    :attr:`ptr` raises :class:`GGMLError` once the handle is closed, so a stale
    wrapper can never pass NULL (or a freed pointer) into ggml. ``close()`` is
    idempotent and also runs from ``__exit__`` and ``__del__``.
    """

    # Class-level default so ``close()`` is safe when ``__init__`` raised.
    _ptr = None

    @property
    def ptr(self):
        """The native pointer, or :class:`GGMLError` once closed."""
        if self._ptr is None:
            raise GGMLError(f"{type(self).__name__} is closed")
        return self._ptr

    @property
    def closed(self) -> bool:
        return self._ptr is None

    def _free(self, ptr) -> None:
        raise NotImplementedError

    def close(self) -> None:
        ptr = self._ptr
        if ptr is not None:
            self._ptr = None
            self._free(ptr)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

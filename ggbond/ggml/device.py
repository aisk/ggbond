"""Thin OO wrapper around ``ggml_backend_dev_t`` (the device registry).

Devices are owned by the backend registry, not by this wrapper, so a
:class:`Device` has no lifetime of its own: it is a handle you inspect and then
call :meth:`Device.init_backend` on. Call :meth:`Backend.load_all` once before
enumerating so dynamically loaded backends are registered.
"""

from __future__ import annotations

import ctypes
from typing import Optional

import ggml as _ggml

from .buffer import BufferType
from .errors import nonnull


class Device:
    """One device from ggml's backend registry."""

    __slots__ = ("ptr",)

    def __init__(self, ptr):
        self.ptr = nonnull(ptr, "ggml_backend_dev_t")

    # -- lookup -------------------------------------------------------------

    @staticmethod
    def count() -> int:
        """Number of registered devices (``ggml_backend_dev_count``)."""
        return _ggml.ggml_backend_dev_count()

    @classmethod
    def get(cls, index: int) -> "Device":
        """Return the device at ``index`` (``ggml_backend_dev_get``)."""
        if not (0 <= index < cls.count()):
            raise IndexError(index)
        return cls(_ggml.ggml_backend_dev_get(index))

    @classmethod
    def all(cls) -> "list[Device]":
        """Every registered device, in registry order."""
        return [cls.get(i) for i in range(cls.count())]

    @classmethod
    def by_name(cls, name: str) -> "Optional[Device]":
        ptr = _ggml.ggml_backend_dev_by_name(name.encode())
        return cls(ptr) if ptr else None

    @classmethod
    def by_type(cls, dev_type: int) -> "Optional[Device]":
        """Return the first device of ``dev_type`` (a ``DEVICE_*`` constant)."""
        ptr = _ggml.ggml_backend_dev_by_type(int(dev_type))
        return cls(ptr) if ptr else None

    # -- queries ------------------------------------------------------------

    @property
    def name(self) -> str:
        return _ggml.ggml_backend_dev_name(self.ptr).decode()

    @property
    def description(self) -> str:
        return _ggml.ggml_backend_dev_description(self.ptr).decode()

    @property
    def type(self) -> int:
        """One of the ``DEVICE_CPU`` / ``DEVICE_GPU`` / ... constants."""
        return _ggml.ggml_backend_dev_type(self.ptr)

    def memory(self) -> "tuple[int, int]":
        """Return ``(free, total)`` bytes of device memory."""
        free = ctypes.c_size_t(0)
        total = ctypes.c_size_t(0)
        _ggml.ggml_backend_dev_memory(self.ptr, ctypes.byref(free), ctypes.byref(total))
        return free.value, total.value

    def buffer_type(self) -> "BufferType":
        return BufferType(_ggml.ggml_backend_dev_buffer_type(self.ptr))

    # -- initialization -----------------------------------------------------

    def init_backend(self, params: Optional[bytes] = None):
        """Initialize a :class:`Backend` on this device (``ggml_backend_dev_init``)."""
        from .backend import Backend

        return Backend(
            nonnull(_ggml.ggml_backend_dev_init(self.ptr, params), "ggml_backend_dev_init"),
            kind=self.name,
        )

    def __repr__(self) -> str:
        return f"<ggml.Device name={self.name!r} type={self.type}>"

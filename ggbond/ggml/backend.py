"""Thin OO wrapper around a ggml backend (CPU / CUDA / Metal / HIP)."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Optional

import ggml as _ggml

from .buffer import Buffer, BufferType
from .device import Device, DeviceType
from .errors import GGMLError, check_status, nonnull
from .handle import Handle

if TYPE_CHECKING:
    from .context import Context
    from .graph import Graph
    from .tensor import Tensor


class Backend(Handle):
    """A ``ggml_backend_t`` handle.

    The constructor adopts an already-created handle; most callers use a
    device-specific factory (:meth:`cpu_init`, :meth:`cuda_init`,
    :meth:`metal_init`, :meth:`hip_init`) or the device registry
    (:meth:`init_best`, :meth:`init_by_type`, :meth:`init_by_name`,
    :meth:`Device.init_backend`).

    Buffers, allocators and schedulers created from a backend must be freed
    before it. They keep the backend alive, and :meth:`close` closes whichever
    of them are still open first, so either order of ``with`` blocks is safe.
    """

    def __init__(self, ptr):
        self._ptr = nonnull(ptr, "ggml_backend_t")
        # Open Buffer / GAllocr / Scheduler objects built on this backend.
        self._dependents = weakref.WeakSet()

    def _track(self, dependent) -> None:
        self._dependents.add(dependent)

    # -- device-specific factories (one per ``ggml_backend_*_init``) --------

    @classmethod
    def cpu_init(cls) -> "Backend":
        """Initialize the CPU backend (``ggml_backend_cpu_init``)."""
        return cls(_ggml.ggml_backend_cpu_init())

    @classmethod
    def cuda_init(cls, device: int = 0) -> "Backend":
        """Initialize CUDA device ``device`` (``ggml_backend_cuda_init``)."""
        try:
            ptr = _ggml.ggml_backend_cuda_init(device)
        except RuntimeError as exc:
            raise GGMLError(f"CUDA backend initialization failed: {exc}") from exc
        return cls(ptr)

    @classmethod
    def metal_init(cls) -> "Backend":
        """Initialize the Metal backend (``ggml_backend_metal_init``)."""
        try:
            ptr = _ggml.ggml_backend_metal_init()
        except RuntimeError as exc:
            raise GGMLError(f"Metal backend initialization failed: {exc}") from exc
        return cls(ptr)

    @classmethod
    def hip_init(cls, device: int = 0) -> "Backend":
        """Initialize HIP/ROCm device ``device`` (``ggml_backend_hip_init``)."""
        if not hasattr(_ggml, "ggml_backend_hip_init"):
            raise GGMLError("HIP backend is not available in this ggml build")
        try:
            ptr = _ggml.ggml_backend_hip_init(device)
        except RuntimeError as exc:
            raise GGMLError(f"HIP backend initialization failed: {exc}") from exc
        return cls(ptr)

    # -- device registry ----------------------------------------------------

    @staticmethod
    def load_all() -> None:
        """Load every available backend library (``ggml_backend_load_all``).

        Call once before the registry factories below or :meth:`Device.all` so
        dynamically loaded backends (CUDA, Metal, ...) get registered.
        """
        _ggml.ggml_backend_load_all()

    @classmethod
    def init_best(cls) -> "Backend":
        """Initialize the best available device (``ggml_backend_init_best``)."""
        return cls(nonnull(_ggml.ggml_backend_init_best(), "ggml_backend_init_best"))

    @classmethod
    def init_by_type(cls, dev_type: DeviceType, *, params: Optional[bytes] = None) -> "Backend":
        """Initialize the first device of ``dev_type`` (``ggml_backend_init_by_type``)."""
        dev_type = DeviceType(dev_type)
        ptr = _ggml.ggml_backend_init_by_type(int(dev_type), params)
        if not ptr:
            raise GGMLError(f"no {dev_type.name} device is available")
        return cls(ptr)

    @classmethod
    def init_by_name(cls, name: str, *, params: Optional[bytes] = None) -> "Backend":
        """Initialize the device called ``name`` (``ggml_backend_init_by_name``).

        ``name`` is a registry device name as reported by :attr:`Device.name`,
        e.g. ``"CPU"`` or ``"CUDA0"``.
        """
        ptr = _ggml.ggml_backend_init_by_name(name.encode(), params)
        if not ptr:
            raise GGMLError(f"no device named {name!r} is available")
        return cls(ptr)

    # -- queries ------------------------------------------------------------

    @property
    def name(self) -> str:
        """The backend's own name (``ggml_backend_name``), e.g. ``"CPU"``."""
        return _ggml.ggml_backend_name(self.ptr).decode()

    @property
    def device(self) -> "Optional[Device]":
        """The :class:`Device` this backend runs on, if ggml reports one."""
        ptr = _ggml.ggml_backend_get_device(self.ptr)
        return Device(ptr) if ptr else None

    @property
    def is_cpu(self) -> bool:
        return bool(_ggml.ggml_backend_is_cpu(self.ptr))

    @property
    def buffer_type(self) -> "BufferType":
        """This backend's default buffer type."""
        return BufferType(_ggml.ggml_backend_get_default_buffer_type(self.ptr))

    def set_n_threads(self, n: int) -> None:
        if not self.is_cpu:
            raise GGMLError("set_n_threads is only valid for the CPU backend")
        _ggml.ggml_backend_cpu_set_n_threads(self.ptr, n)

    def supports_op(self, tensor: "Tensor") -> bool:
        """Whether this backend can compute the op that produced ``tensor``."""
        return bool(_ggml.ggml_backend_supports_op(self.ptr, tensor.ptr))

    def offload_op(self, tensor: "Tensor") -> bool:
        """Whether this backend wants to take ``tensor``'s op from another backend."""
        return bool(_ggml.ggml_backend_offload_op(self.ptr, tensor.ptr))

    # -- allocation ---------------------------------------------------------

    def alloc_ctx_tensors(self, ctx: "Context") -> "Buffer":
        """Allocate backend memory for every tensor defined in ``ctx``.

        Wraps ``ggml_backend_alloc_ctx_tensors``. ``ctx`` keeps the returned
        :class:`Buffer` alive, so the return value can be ignored: the memory
        is freed when the buffer is closed explicitly, or once both ``ctx`` and
        your own reference to the buffer are gone.
        """
        buffer = Buffer(
            nonnull(
                _ggml.ggml_backend_alloc_ctx_tensors(ctx.ptr, self.ptr),
                "ggml_backend_alloc_ctx_tensors",
            ),
            backend=self,
        )
        ctx._buffers.append(buffer)
        return buffer

    # -- compute ------------------------------------------------------------

    def compute(self, graph: "Graph") -> None:
        """Compute ``graph`` and wait for it (``ggml_backend_graph_compute``).

        ggml synchronizes before returning, so results can be read right away.
        """
        status = _ggml.ggml_backend_graph_compute(self.ptr, graph.ptr)
        check_status(status, "ggml_backend_graph_compute")

    def compute_async(self, graph: "Graph") -> None:
        """Start computing ``graph``; call :meth:`synchronize` before reading results."""
        status = _ggml.ggml_backend_graph_compute_async(self.ptr, graph.ptr)
        check_status(status, "ggml_backend_graph_compute_async")

    def synchronize(self) -> None:
        _ggml.ggml_backend_synchronize(self.ptr)

    # -- lifecycle ----------------------------------------------------------

    def _free(self, ptr) -> None:
        for dependent in list(self._dependents):
            dependent.close()
        _ggml.ggml_backend_free(ptr)

    def __repr__(self) -> str:
        if self.closed:
            return "<ggml.Backend closed>"
        return f"<ggml.Backend name={self.name!r}>"

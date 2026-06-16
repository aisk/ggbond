"""Thin OO wrapper around a ggml backend (CPU / CUDA / Metal / HIP)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import ggml as _ggml

from .errors import GGMLError, check_status, nonnull

if TYPE_CHECKING:
    from .context import Context
    from .graph import Graph


# Device types for :meth:`Backend.init_by_type` (mirror ggml's
# ``enum ggml_backend_dev_type`` -- values never drift).
DEVICE_CPU = _ggml.GGML_BACKEND_DEVICE_TYPE_CPU
DEVICE_GPU = _ggml.GGML_BACKEND_DEVICE_TYPE_GPU
DEVICE_IGPU = _ggml.GGML_BACKEND_DEVICE_TYPE_IGPU
DEVICE_ACCEL = _ggml.GGML_BACKEND_DEVICE_TYPE_ACCEL


class Backend:
    """A ggml backend handle plus the buffers it allocates.

    Buffers allocated through :meth:`alloc_ctx_tensors` are tracked and freed
    on :meth:`close` (before the backend itself).
    """

    def __init__(self, ptr, *, kind: str = "backend"):
        """Adopt an already-created ``ggml_backend_t`` handle.

        Most callers should use a device-specific factory
        (:meth:`cpu_init`, :meth:`cuda_init`, :meth:`metal_init`,
        :meth:`hip_init`) or :meth:`init_best` / :meth:`init_by_type` rather
        than constructing from a raw pointer directly. ``kind`` is just a label
        for :func:`repr`.
        """
        self.ptr = nonnull(ptr, f"{kind} backend")
        self.kind = kind
        self._buffers = []
        self._closed = False

    # -- device-specific factories (one per ``ggml_backend_*_init``) --------

    @classmethod
    def cpu_init(cls) -> "Backend":
        """Initialize the CPU backend (``ggml_backend_cpu_init``)."""
        return cls(_ggml.ggml_backend_cpu_init(), kind="cpu")

    @classmethod
    def cuda_init(cls, device: int = 0) -> "Backend":
        """Initialize CUDA device ``device`` (``ggml_backend_cuda_init``)."""
        return cls(_ggml.ggml_backend_cuda_init(device), kind="cuda")

    @classmethod
    def metal_init(cls) -> "Backend":
        """Initialize the Metal backend (``ggml_backend_metal_init``)."""
        return cls(_ggml.ggml_backend_metal_init(), kind="metal")

    @classmethod
    def hip_init(cls, device: int = 0) -> "Backend":
        """Initialize HIP/ROCm device ``device`` (``ggml_backend_hip_init``)."""
        if not hasattr(_ggml, "ggml_backend_hip_init"):
            raise GGMLError("HIP backend is not available in this ggml build")
        return cls(_ggml.ggml_backend_hip_init(device), kind="hip")

    @staticmethod
    def load_all() -> None:
        """Load every available backend library (``ggml_backend_load_all``).

        Call once before :meth:`init_best` / :meth:`init_by_type` so dynamically
        loaded backends (CUDA, Metal, ...) get registered.
        """
        _ggml.ggml_backend_load_all()

    @classmethod
    def init_best(cls) -> "Backend":
        """Initialize the best available device (``ggml_backend_init_best``)."""
        return cls(_ggml.ggml_backend_init_best(), kind="best")

    @classmethod
    def init_by_type(cls, dev_type: int, *, params: Optional[bytes] = None) -> "Backend":
        """Initialize the first device of ``dev_type`` (``ggml_backend_init_by_type``).

        ``dev_type`` is one of :data:`DEVICE_CPU`, :data:`DEVICE_GPU`,
        :data:`DEVICE_IGPU`, :data:`DEVICE_ACCEL`.
        """
        return cls(_ggml.ggml_backend_init_by_type(dev_type, params),
                   kind=f"type:{int(dev_type)}")

    # -- queries ------------------------------------------------------------

    @property
    def is_cpu(self) -> bool:
        return bool(_ggml.ggml_backend_is_cpu(self.ptr))

    def set_n_threads(self, n: int) -> None:
        if not self.is_cpu:
            raise GGMLError("set_n_threads is only valid for the CPU backend")
        _ggml.ggml_backend_cpu_set_n_threads(self.ptr, n)

    def buffer_type(self):
        return _ggml.ggml_backend_get_default_buffer_type(self.ptr)

    # -- allocation ---------------------------------------------------------

    def alloc_ctx_tensors(self, ctx: "Context"):
        """Allocate backend memory for every tensor defined in ``ctx``."""
        buf = nonnull(
            _ggml.ggml_backend_alloc_ctx_tensors(ctx.ptr, self.ptr),
            "ggml_backend_alloc_ctx_tensors",
        )
        self._buffers.append(buf)
        return buf

    # -- compute ------------------------------------------------------------

    def compute(self, graph: "Graph") -> None:
        status = _ggml.ggml_backend_graph_compute(self.ptr, graph.ptr)
        check_status(status, "ggml_backend_graph_compute")

    def synchronize(self) -> None:
        _ggml.ggml_backend_synchronize(self.ptr)

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        for buf in self._buffers:
            if buf is not None:
                _ggml.ggml_backend_buffer_free(buf)
        self._buffers.clear()
        if self.ptr is not None:
            _ggml.ggml_backend_free(self.ptr)
            self.ptr = None
        self._closed = True

    def __enter__(self) -> "Backend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

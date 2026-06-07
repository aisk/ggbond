"""Thin OO wrapper around a ggml backend (CPU / CUDA / Metal / HIP)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ._ffi import _ggml, GGMLError, check_status, nonnull

if TYPE_CHECKING:
    from .context import Context
    from .graph import Graph


def _init_backend(kind: str, device: int):
    if kind == "cpu":
        return _ggml.ggml_backend_cpu_init()
    if kind == "cuda":
        return _ggml.ggml_backend_cuda_init(device)
    if kind == "metal":
        return _ggml.ggml_backend_metal_init()
    if kind in ("hip", "rocm"):
        if not hasattr(_ggml, "ggml_backend_hip_init"):
            raise GGMLError("HIP backend is not available in this ggml build")
        return _ggml.ggml_backend_hip_init(device)
    raise ValueError(f"unknown backend kind: {kind!r}")


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

    def __init__(self, kind: str = "cpu", *, device: int = 0, n_threads: Optional[int] = None):
        kind = kind.lower()
        ptr = nonnull(_init_backend(kind, device), f"{kind} backend init")
        self._adopt(ptr, kind, n_threads)

    def _adopt(self, ptr, kind: str, n_threads: Optional[int]) -> None:
        """Shared setup for an already-created ``ggml_backend_t`` handle."""
        self.ptr = ptr
        self.kind = kind
        self._buffers = []
        self._closed = False
        if n_threads is not None and self.is_cpu:
            self.set_n_threads(n_threads)

    @staticmethod
    def load_all() -> None:
        """Load every available backend library (``ggml_backend_load_all``).

        Call once before :meth:`init_best` / :meth:`init_by_type` so dynamically
        loaded backends (CUDA, Metal, ...) get registered.
        """
        _ggml.ggml_backend_load_all()

    @classmethod
    def init_best(cls, *, n_threads: Optional[int] = None) -> "Backend":
        """Initialize the best available device (``ggml_backend_init_best``)."""
        be = cls.__new__(cls)
        ptr = nonnull(_ggml.ggml_backend_init_best(), "ggml_backend_init_best")
        be._adopt(ptr, "best", n_threads)
        return be

    @classmethod
    def init_by_type(cls, dev_type: int, *, params: Optional[bytes] = None,
                     n_threads: Optional[int] = None) -> "Backend":
        """Initialize the first device of ``dev_type`` (``ggml_backend_init_by_type``).

        ``dev_type`` is one of :data:`DEVICE_CPU`, :data:`DEVICE_GPU`,
        :data:`DEVICE_IGPU`, :data:`DEVICE_ACCEL`.
        """
        be = cls.__new__(cls)
        ptr = nonnull(_ggml.ggml_backend_init_by_type(dev_type, params),
                      "ggml_backend_init_by_type")
        be._adopt(ptr, f"type:{int(dev_type)}", n_threads)
        return be

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

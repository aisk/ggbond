"""Thin OO wrapper around a ggml backend (CPU / CUDA / Metal / HIP)."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Optional

import numpy as np

from ._ffi import _ggml, GGMLError, check_status, nonnull
from .types import dtype_to_numpy

if TYPE_CHECKING:
    from .context import Context
    from .graph import Graph
    from .tensor import Tensor


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


class Backend:
    """A ggml backend handle plus the buffers it allocates.

    Buffers allocated through :meth:`alloc_ctx_tensors` are tracked and freed
    on :meth:`close` (before the backend itself).
    """

    def __init__(self, kind: str = "cpu", *, device: int = 0, n_threads: Optional[int] = None):
        kind = kind.lower()
        self.ptr = nonnull(_init_backend(kind, device), f"{kind} backend init")
        self.kind = kind
        self._buffers = []
        self._closed = False
        if n_threads is not None and self.is_cpu:
            self.set_n_threads(n_threads)

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

    # -- data transfer ------------------------------------------------------

    def tensor_set(self, tensor: "Tensor", x) -> None:
        arr = np.ascontiguousarray(x, dtype=dtype_to_numpy(tensor.dtype))
        ptr = arr.ctypes.data_as(ctypes.c_void_p)
        _ggml.ggml_backend_tensor_set(tensor.ptr, ptr, 0, tensor.nbytes)

    def tensor_get(self, tensor: "Tensor", out: np.ndarray) -> np.ndarray:
        """Copy ``tensor`` into the pre-allocated, contiguous numpy array ``out``."""
        ptr = out.ctypes.data_as(ctypes.c_void_p)
        _ggml.ggml_backend_tensor_get(tensor.ptr, ptr, 0, tensor.nbytes)
        return out

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

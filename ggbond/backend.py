"""GGML backend wrapper with automatic lifecycle management."""

from __future__ import annotations

import platform

import numpy as np

from ggbond import ggml
from ggbond.context import Context


class Backend:
    """Wraps a ggml backend (CPU or Metal) with context-manager support."""

    def __init__(self, name: str = "cpu", *, n_threads: int = 4):
        name = name.lower()
        if name == "metal":
            if platform.system() != "Darwin":
                raise RuntimeError("Metal backend is only available on macOS")
            self._backend = ggml.backend_metal_init()
        elif name == "cpu":
            self._backend = ggml.backend_cpu_init()
        else:
            raise ValueError(f"unknown backend: {name!r} (use 'cpu' or 'metal')")

        if not self._backend:
            raise RuntimeError(f"failed to initialize {name} backend")

        self._name = name
        self._n_threads = n_threads
        if name == "cpu":
            ggml.backend_cpu_set_n_threads(self._backend, n_threads)

    # -- properties ----------------------------------------------------------

    @property
    def raw(self) -> ggml.Backend:
        """Underlying opaque ``ggml.Backend`` pointer."""
        return self._backend

    @property
    def buffer_type(self) -> ggml.BufferType:
        """Default buffer type for this backend."""
        return ggml.backend_get_default_buffer_type(self._backend)

    # -- high-level helpers ---------------------------------------------------

    def alloc_ctx(self, ctx: Context) -> ggml.Buffer:
        """Allocate all tensors in *ctx* on this backend. Returns the buffer."""
        return ggml.backend_alloc_ctx_tensors(ctx.raw, self._backend)

    def compute(self, graph: ggml.Graph) -> None:
        """Execute a computation graph on this backend."""
        if self._name == "cpu":
            ggml.backend_cpu_set_n_threads(self._backend, self._n_threads)
        ggml.backend_graph_compute(self._backend, graph)

    def tensor_set(self, tensor: ggml.Tensor, data: np.ndarray) -> None:
        """Copy *data* into *tensor*.  Automatically flattens and computes size."""
        flat = data.flatten()
        ggml.backend_tensor_set(tensor, flat, 0, ggml.nbytes(tensor))

    def tensor_get(self, tensor: ggml.Tensor, dtype=np.float32) -> np.ndarray:
        """Read full tensor data back as a 1-D numpy array."""
        out = np.empty(ggml.nelements(tensor), dtype=dtype)
        ggml.backend_tensor_get(tensor, out, 0, ggml.nbytes(tensor))
        return out

    def tensor_get_slice(self, tensor: ggml.Tensor, offset: int, count: int, dtype=np.float32) -> np.ndarray:
        """Read *count* elements from *tensor* starting at byte *offset*."""
        out = np.empty(count, dtype=dtype)
        ggml.backend_tensor_get(tensor, out, offset, count * out.itemsize)
        return out

    # -- lifecycle ------------------------------------------------------------

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Free the backend.  Safe to call multiple times."""
        if self._backend:
            ggml.backend_free(self._backend)
            self._backend = None

    def __enter__(self) -> Backend:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

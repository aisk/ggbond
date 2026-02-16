"""Lightweight session that binds a Backend and tracks resources for Tensor creation."""

from __future__ import annotations

import numpy as np

from ggbond import ggml
from ggbond.backend import Backend
from ggbond.context import Context
from ggbond.gguf import load_gguf as _load_gguf
from ggbond.tensor import Tensor


def _tensor_shape(t: ggml.Tensor) -> tuple[int, ...]:
    """Extract GGML shape (ne0, ne1, ...) from a raw ggml tensor."""
    ndim = ggml.n_dims(t)
    return tuple(ggml.tensor_ne(t, i) for i in range(ndim))


class Session:
    """Single entry-point that owns a backend, tracks all resources, and creates Tensors."""

    def __init__(self, backend: str = "cpu", *, n_threads: int = 4):
        self._backend = Backend(backend, n_threads=n_threads)
        self._contexts: list = []
        self._buffers: list = []

    def tensor(self, data, *, name: str | None = None) -> Tensor:
        """Create a leaf Tensor bound to this session. Data is uploaded to the backend immediately."""
        data = np.asarray(data, dtype=np.float32)
        shape = tuple(reversed(data.shape))
        ctx = Context(n_tensors=1)
        t = ctx.new_tensor(ggml.Type.F32, *shape, name=name)
        ggml.set_input(t)
        buf = self._backend.alloc_ctx(ctx)
        self._backend.tensor_set(t, data)
        self._contexts.append(ctx)
        self._buffers.append(buf)
        return Tensor._from_ggml(self, t, shape=shape, name=name)

    def load_gguf(self, fname: str) -> dict[str, Tensor]:
        """Load a GGUF model. Weights reside directly on the current backend."""
        ctx_w, buf_w, raw_tensors = _load_gguf(fname, self._backend)
        self._contexts.append(ctx_w)
        self._buffers.append(buf_w)
        tensors = {}
        for name, t in raw_tensors.items():
            shape = _tensor_shape(t)
            tensors[name] = Tensor._from_ggml(self, t, shape=shape, name=name)
        return tensors

    def close(self):
        """Release all resources in reverse order. Safe to call multiple times."""
        for ctx in reversed(self._contexts):
            if isinstance(ctx, Context):
                ctx.close()
            else:
                ggml.context_free(ctx)
        self._contexts.clear()
        for buf in reversed(self._buffers):
            ggml.backend_buffer_free(buf)
        self._buffers.clear()
        if self._backend:
            self._backend.close()
            self._backend = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

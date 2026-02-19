"""Lightweight session that binds a Backend and tracks resources for Tensor creation."""

from __future__ import annotations

import numpy as np

from ggbond import ggml
from ggbond.backend import Backend
from ggbond.context import Context
from ggbond.gguf import load_gguf as _load_gguf, GGUF, GGUFMeta
from ggbond.tensor import Tensor


def _tensor_shape(t: ggml.Tensor) -> tuple[int, ...]:
    """Extract GGML shape (ne0, ne1, ...) from a raw ggml tensor."""
    ndim = ggml.n_dims(t)
    return tuple(ggml.tensor_ne(t, i) for i in range(ndim))


class Session:
    """Single entry-point that owns a backend, tracks all resources, and creates Tensors."""

    def __init__(self, backend: str = "cpu", *, n_threads: int = 4):
        ggml.time_init()
        ggml.log_set_default()
        self._backend = Backend(backend, n_threads=n_threads)
        self._contexts: list = []
        self._buffers: list = []

    @property
    def backend(self) -> Backend:
        """The underlying backend (read-only)."""
        return self._backend

    def tensor(self, data, *, dtype=None, shape=None, name: str | None = None) -> Tensor:
        """Create a leaf Tensor bound to this session. Data is uploaded to the backend immediately.

        *dtype* defaults to F32. Pass e.g. ``ggml.Type.I32`` for integer tensors.
        *shape* overrides the shape in GGML order (ne0, ne1, ...); by default
        it is inferred from the numpy array shape (reversed).
        *data* may be a numpy array, a scalar, or raw ``bytes``.
        """
        _NP_DTYPE = {
            ggml.Type.F32: np.float32,
            ggml.Type.F16: np.float16,
            ggml.Type.I32: np.int32,
            ggml.Type.I8: np.int8,
            ggml.Type.I16: np.int16,
            ggml.Type.I64: np.int64,
            ggml.Type.F64: np.float64,
        }
        if dtype is None:
            dtype = ggml.Type.F32
        np_dtype = _NP_DTYPE.get(dtype, np.float32)

        if isinstance(data, bytes):
            raw_bytes = data
            if shape is None:
                raise ValueError("shape is required when data is raw bytes")
        else:
            data = np.asarray(data, dtype=np_dtype)
            raw_bytes = None
            if shape is None:
                shape = tuple(reversed(data.shape))

        ctx = Context(n_tensors=1)
        t = ctx.new_tensor(dtype, *shape, name=name)
        ggml.set_input(t)
        buf = self._backend.alloc_ctx(ctx)
        if raw_bytes is not None:
            ggml.backend_tensor_set(t, np.frombuffer(raw_bytes, dtype=np.uint8), 0, len(raw_bytes))
        else:
            self._backend.tensor_set(t, data)
        self._contexts.append(ctx)
        self._buffers.append(buf)
        return Tensor._from_ggml(self, t, shape=shape, dtype=dtype, name=name)

    def empty(self, dtype, *shape, name: str | None = None) -> Tensor:
        """Create an uninitialized leaf Tensor (e.g. for KV cache buffers)."""
        ctx = Context(n_tensors=1)
        t = ctx.new_tensor(dtype, *shape, name=name)
        ggml.set_input(t)
        buf = self._backend.alloc_ctx(ctx)
        self._contexts.append(ctx)
        self._buffers.append(buf)
        return Tensor._from_ggml(self, t, shape=shape, dtype=dtype, name=name)

    def load_gguf(self, fname: str) -> GGUF:
        """Load a GGUF model. Returns a GGUF object with .weights and .meta."""
        ctx_w, buf_w, raw_tensors, meta = _load_gguf(fname, self._backend)
        self._contexts.append(ctx_w)
        self._buffers.append(buf_w)
        tensors = {}
        for name, t in raw_tensors.items():
            shape = _tensor_shape(t)
            tensors[name] = Tensor._from_ggml(self, t, shape=shape, name=name)
        return GGUF(tensors, meta)

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

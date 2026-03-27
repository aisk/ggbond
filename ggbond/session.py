"""Lightweight session that binds a Backend and tracks resources for Tensor creation."""

from __future__ import annotations

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

    def __init__(self, backend: str = "cpu", *, n_threads: int = 4, device: int = 0):
        ggml.time_init()
        ggml.log_set_default()
        self._backend = Backend(backend, n_threads=n_threads, device=device)
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
        return Tensor(data, session=self, dtype=dtype, shape=shape, name=name)

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
        for ctx in reversed(getattr(self, '_contexts', [])):
            if isinstance(ctx, Context):
                ctx.close()
            else:
                ggml.context_free(ctx)
        self._contexts = []
        for buf in reversed(getattr(self, '_buffers', [])):
            ggml.backend_buffer_free(buf)
        self._buffers = []
        if getattr(self, '_backend', None):
            self._backend.close()
            self._backend = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

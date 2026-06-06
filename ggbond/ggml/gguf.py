"""Minimal GGUF model loader built on the OO wrapper.

Loads a ``.gguf`` file into backend-resident weights. The returned :class:`GGUF`
owns the weights' :class:`Context` (and therefore must outlive any graph that
references those weights); the backend owns the weight buffer and frees it on
``Backend.close()``.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

from ._ffi import _ggml, nonnull
from .context import Context
from .tensor import Tensor

if TYPE_CHECKING:
    from .backend import Backend


# GGUF KV value type tags we know how to read.
_GGUF_TYPE_UINT32 = int(_ggml.GGUF_TYPE_UINT32)
_GGUF_TYPE_INT32 = int(_ggml.GGUF_TYPE_INT32)
_GGUF_TYPE_FLOAT32 = int(_ggml.GGUF_TYPE_FLOAT32)
_GGUF_TYPE_STRING = int(_ggml.GGUF_TYPE_STRING)
_GGUF_TYPE_ARRAY = int(_ggml.GGUF_TYPE_ARRAY)

_SCALAR_GETTERS = {
    _GGUF_TYPE_UINT32: _ggml.gguf_get_val_u32,
    _GGUF_TYPE_INT32: _ggml.gguf_get_val_i32,
    _GGUF_TYPE_FLOAT32: _ggml.gguf_get_val_f32,
}


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else value


def _read_metadata(gguf_ctx) -> dict:
    """Copy the scalar / string-array KV pairs out before ``gguf_free``."""
    meta: dict = {}
    for i in range(_ggml.gguf_get_n_kv(gguf_ctx)):
        key = _decode(_ggml.gguf_get_key(gguf_ctx, i))
        kv_type = int(_ggml.gguf_get_kv_type(gguf_ctx, i))
        if kv_type in _SCALAR_GETTERS:
            meta[key] = _SCALAR_GETTERS[kv_type](gguf_ctx, i)
        elif kv_type == _GGUF_TYPE_STRING:
            meta[key] = _decode(_ggml.gguf_get_val_str(gguf_ctx, i))
        elif kv_type == _GGUF_TYPE_ARRAY:
            if int(_ggml.gguf_get_arr_type(gguf_ctx, i)) == _GGUF_TYPE_STRING:
                n = _ggml.gguf_get_arr_n(gguf_ctx, i)
                meta[key] = [_decode(_ggml.gguf_get_arr_str(gguf_ctx, i, j)) for j in range(n)]
    return meta


class GGUF:
    """A loaded GGUF model: weights as :class:`Tensor` objects plus metadata.

    Owns the weights :class:`Context`; use as a context manager (or call
    :meth:`close`) so the context is freed once computation is done.
    """

    def __init__(self, ctx: Context, weights: dict, meta: dict):
        self.ctx = ctx
        self.weights = weights  # dict[str, Tensor]
        self.meta = meta        # dict[str, scalar | str | list[str]]

    def __getitem__(self, name: str) -> Tensor:
        return self.weights[name]

    def __contains__(self, name: str) -> bool:
        return name in self.weights

    def close(self) -> None:
        self.ctx.close()

    def __enter__(self) -> "GGUF":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"GGUF(weights={len(self.weights)}, meta={len(self.meta)} keys)"


def load_gguf(path: str, backend: "Backend") -> GGUF:
    """Load ``path`` and allocate its tensors on ``backend``.

    The tensor metadata lives in a fresh ``ggml_context`` (owned by the returned
    :class:`GGUF`); the tensor *data* is uploaded into a backend buffer tracked
    by ``backend``.
    """
    # gguf_init_from_file fills `ctx_holder` with the metadata context it creates.
    ctx_holder = _ggml.ggml_context_p_ctypes()
    params = _ggml.gguf_init_params(no_alloc=True, ctx=ctypes.pointer(ctx_holder))
    gguf_ctx = nonnull(
        _ggml.gguf_init_from_file(str(path).encode(), params),
        "gguf_init_from_file",
    )
    try:
        ctx = Context.from_ptr(ctx_holder, no_alloc=True, owns=True)
        backend.alloc_ctx_tensors(ctx)
        meta = _read_metadata(gguf_ctx)

        n_tensors = _ggml.gguf_get_n_tensors(gguf_ctx)
        data_offset = _ggml.gguf_get_data_offset(gguf_ctx)

        weights: dict[str, Tensor] = {}
        with open(path, "rb") as f:
            for i in range(n_tensors):
                name = _decode(_ggml.gguf_get_tensor_name(gguf_ctx, i))
                ptr = nonnull(_ggml.ggml_get_tensor(ctx.ptr, name.encode()), f"tensor {name!r}")
                offset = data_offset + _ggml.gguf_get_tensor_offset(gguf_ctx, i)
                nbytes = _ggml.ggml_nbytes(ptr)
                f.seek(offset)
                raw = f.read(nbytes)
                buf = (ctypes.c_char * nbytes).from_buffer_copy(raw)
                _ggml.ggml_backend_tensor_set(ptr, ctypes.cast(buf, ctypes.c_void_p), 0, nbytes)
                weights[name] = Tensor(ctx, ptr)
    finally:
        _ggml.gguf_free(gguf_ctx)

    return GGUF(ctx, weights, meta)

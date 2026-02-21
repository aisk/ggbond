"""GGUF model loading helper."""

from __future__ import annotations

import numpy as np

from ggbond import ggml
from ggbond.backend import Backend


class GGUFMeta:
    """Pre-fetched GGUF metadata. All values are copied out before gguf_ctx is freed."""

    _TYPE_U32 = 4   # GGUF_TYPE_UINT32
    _TYPE_I32 = 5   # GGUF_TYPE_INT32
    _TYPE_F32 = 6   # GGUF_TYPE_FLOAT32
    _TYPE_STR = 8   # GGUF_TYPE_STRING
    _TYPE_ARR = 9   # GGUF_TYPE_ARRAY

    def __init__(self):
        self._scalars: dict[str, int | float | str] = {}
        self._arr_str: dict[str, list[str]] = {}

    @classmethod
    def _from_gguf_ctx(cls, gguf_ctx) -> "GGUFMeta":
        meta = cls()
        n_kv = ggml.gguf_get_n_kv(gguf_ctx)
        for i in range(n_kv):
            key = ggml.gguf_get_key(gguf_ctx, i)
            kv_type = ggml.gguf_get_kv_type(gguf_ctx, i)
            if kv_type == cls._TYPE_U32:
                meta._scalars[key] = ggml.gguf_get_val_u32(gguf_ctx, i)
            elif kv_type == cls._TYPE_I32:
                meta._scalars[key] = ggml.gguf_get_val_i32(gguf_ctx, i)
            elif kv_type == cls._TYPE_F32:
                meta._scalars[key] = ggml.gguf_get_val_f32(gguf_ctx, i)
            elif kv_type == cls._TYPE_STR:
                meta._scalars[key] = ggml.gguf_get_val_str(gguf_ctx, i)
            elif kv_type == cls._TYPE_ARR:
                if ggml.gguf_get_arr_type(gguf_ctx, i) == cls._TYPE_STR:
                    n = ggml.gguf_get_arr_n(gguf_ctx, i)
                    meta._arr_str[key] = [
                        ggml.gguf_get_arr_str(gguf_ctx, i, j) for j in range(n)
                    ]
        return meta

    def get(self, key: str, default=None) -> int | float | str | list[str] | None:
        if key in self._scalars:
            return self._scalars[key]
        if key in self._arr_str:
            return self._arr_str[key]
        return default

    def __contains__(self, key: str) -> bool:
        return key in self._scalars or key in self._arr_str

    def keys(self) -> list[str]:
        return list(self._scalars) + list(self._arr_str)

    def __repr__(self) -> str:
        return f"GGUFMeta({len(self._scalars)} scalars, {len(self._arr_str)} arrays)"


class GGUF:
    """Result of loading a GGUF file: holds both tensor weights and metadata."""

    def __init__(self, weights: dict, meta: GGUFMeta):
        self.weights = weights   # dict[str, Tensor] — tensor name -> Tensor
        self.meta = meta         # GGUFMeta — hyperparameters, vocabulary, etc.

    def __repr__(self) -> str:
        return f"GGUF(weights={len(self.weights)}, meta={self.meta!r})"


def load_gguf(
    fname: str, backend: Backend
) -> tuple[ggml.Context, ggml.Buffer, dict[str, ggml.Tensor], GGUFMeta]:
    """Load a GGUF model file and return ``(ctx_w, buf_w, tensors, meta)``.

    The caller is responsible for freeing *ctx_w* and *buf_w* when done.
    """
    gguf_ctx, ctx_w = ggml.gguf_init_from_file(fname, no_alloc=True)
    buf_w = ggml.backend_alloc_ctx_tensors(ctx_w, backend.raw)

    meta = GGUFMeta._from_gguf_ctx(gguf_ctx)

    n_tensors = ggml.gguf_get_n_tensors(gguf_ctx)
    data_offset = ggml.gguf_get_data_offset(gguf_ctx)

    tensors: dict[str, ggml.Tensor] = {}
    with open(fname, "rb") as f:
        for i in range(n_tensors):
            name = ggml.gguf_get_tensor_name(gguf_ctx, i)
            tensor = ggml.get_tensor(ctx_w, name)
            tensors[name] = tensor
            offset = data_offset + ggml.gguf_get_tensor_offset(gguf_ctx, i)
            nbytes = ggml.nbytes(tensor)
            f.seek(offset)
            data = np.frombuffer(f.read(nbytes), dtype=np.uint8)
            ggml.backend_tensor_set(tensor, data, 0, nbytes)

    ggml.gguf_free(gguf_ctx)
    return ctx_w, buf_w, tensors, meta

"""GGUF model loading helper."""

from __future__ import annotations

import numpy as np

from ggbond import ggml
from ggbond.backend import Backend


def load_gguf(
    fname: str, backend: Backend
) -> tuple[ggml.Context, ggml.Buffer, dict[str, ggml.Tensor]]:
    """Load a GGUF model file and return ``(ctx_w, buf_w, tensors)``.

    The caller is responsible for freeing *ctx_w* and *buf_w* when done.
    """
    gguf_ctx, ctx_w = ggml.gguf_init_from_file(fname, no_alloc=True)
    buf_w = ggml.backend_alloc_ctx_tensors(ctx_w, backend.raw)

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
    return ctx_w, buf_w, tensors

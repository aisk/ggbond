"""Graph allocator wrapper and GGUF model loading helper."""

from __future__ import annotations

import numpy as np

from ggbond import ggml
from ggbond.backend import Backend


class GraphRunner:
    """Wraps ``ggml_gallocr`` for graph allocation, execution, and result extraction."""

    def __init__(self, backend: Backend):
        self._backend = backend
        self._allocr = ggml.gallocr_new(backend.buffer_type)

    # -- core operations ------------------------------------------------------

    def reserve(self, graph: ggml.Graph) -> int:
        """Reserve memory for *graph*. Returns buffer size in bytes."""
        ggml.gallocr_reserve(self._allocr, graph)
        return ggml.gallocr_get_buffer_size(self._allocr, 0)

    def compute(
        self,
        graph: ggml.Graph,
        inputs: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Allocate graph, optionally set inputs by name, and compute."""
        ggml.gallocr_alloc_graph(self._allocr, graph)
        if inputs:
            for name, data in inputs.items():
                tensor = ggml.graph_get_tensor(graph, name)
                flat = data.flatten() if data.ndim > 1 else data
                ggml.backend_tensor_set(tensor, flat, 0, ggml.nbytes(tensor))
        self._backend.compute(graph)

    # -- result extraction ----------------------------------------------------

    def get(
        self, graph: ggml.Graph, tensor_or_name, dtype=np.float32
    ) -> np.ndarray:
        """Read full tensor data from the graph.

        *tensor_or_name* can be a ``ggml.Tensor`` object or a string name.
        """
        tensor = (
            ggml.graph_get_tensor(graph, tensor_or_name)
            if isinstance(tensor_or_name, str)
            else tensor_or_name
        )
        out = np.empty(ggml.nelements(tensor), dtype=dtype)
        ggml.backend_tensor_get(tensor, out, 0, ggml.nbytes(tensor))
        return out

    def get_slice(
        self,
        graph: ggml.Graph,
        name: str,
        offset: int,
        count: int,
        dtype=np.float32,
    ) -> np.ndarray:
        """Read *count* elements from tensor *name* starting at byte *offset*."""
        tensor = ggml.graph_get_tensor(graph, name)
        out = np.empty(count, dtype=dtype)
        ggml.backend_tensor_get(tensor, out, offset, count * out.itemsize)
        return out

    # -- properties -----------------------------------------------------------

    @property
    def buffer_size(self) -> int:
        """Current buffer size (call after :meth:`reserve`)."""
        return ggml.gallocr_get_buffer_size(self._allocr, 0)

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        """Free the graph allocator.  Safe to call multiple times."""
        if self._allocr:
            ggml.gallocr_free(self._allocr)
            self._allocr = None

    def __enter__(self) -> GraphRunner:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


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

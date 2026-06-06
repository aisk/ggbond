"""Thin OO wrapper around ``ggml_context``."""

from __future__ import annotations

from typing import Optional

from ._ffi import _ggml, nonnull
from .types import DType
from .tensor import Tensor
from .graph import Graph


class Context:
    """Owns a ``ggml_context``: the arena that holds tensor metadata (and,
    when ``no_alloc=False``, tensor data).

    ``mem_size`` (bytes) is the size of that arena, passed straight to
    ``ggml_init_params``. Size it for the tensor metadata and graphs you will
    create (use ``ggml_tensor_overhead()`` / ``ggml_graph_overhead()`` from the
    bindings), plus the tensor data itself when ``no_alloc=False``.

    A :class:`Context` must outlive every :class:`Tensor` / :class:`Graph` it
    produces, because ``ggml_free`` releases all of their metadata at once.
    Tensors hold a strong reference back to their context to enforce this.
    """

    def __init__(
        self,
        mem_size: int,
        *,
        no_alloc: bool = True,
        mem_buffer=None,
    ):
        params = _ggml.ggml_init_params(mem_size, mem_buffer, no_alloc)
        self.ptr = nonnull(_ggml.ggml_init(params), "ggml_init")
        self.no_alloc = no_alloc
        self._owns = True
        self._closed = False

    @classmethod
    def from_ptr(cls, ptr, *, no_alloc: bool = True, owns: bool = True) -> "Context":
        """Wrap an externally created ``ggml_context``.

        Used for contexts allocated outside this class (e.g. the metadata
        context filled in by ``gguf_init_from_file``). When ``owns`` is True the
        context is freed on :meth:`close`; set it False to wrap a context whose
        lifetime is managed elsewhere.
        """
        self = cls.__new__(cls)
        self.ptr = nonnull(ptr, "ggml_context")
        self.no_alloc = no_alloc
        self._owns = owns
        self._closed = False
        return self

    # -- tensor factories ---------------------------------------------------
    #
    # Dimensions are in **ggml order** ``(ne0, ne1, ...)`` -- the reverse of
    # numpy's row-major order. Each factory maps 1:1 to ``ggml_new_tensor_Nd``.

    def _new_tensor(self, ptr, name: Optional[str]) -> Tensor:
        tensor = Tensor(self, nonnull(ptr, "ggml_new_tensor"))
        if name is not None:
            tensor.name = name
        return tensor

    def new_tensor_1d(self, dtype: "DType", ne0: int, *, name: Optional[str] = None) -> Tensor:
        return self._new_tensor(_ggml.ggml_new_tensor_1d(self.ptr, int(dtype), ne0), name)

    def new_tensor_2d(self, dtype: "DType", ne0: int, ne1: int, *, name: Optional[str] = None) -> Tensor:
        return self._new_tensor(_ggml.ggml_new_tensor_2d(self.ptr, int(dtype), ne0, ne1), name)

    def new_tensor_3d(self, dtype: "DType", ne0: int, ne1: int, ne2: int, *, name: Optional[str] = None) -> Tensor:
        return self._new_tensor(_ggml.ggml_new_tensor_3d(self.ptr, int(dtype), ne0, ne1, ne2), name)

    def new_tensor_4d(self, dtype: "DType", ne0: int, ne1: int, ne2: int, ne3: int, *, name: Optional[str] = None) -> Tensor:
        return self._new_tensor(_ggml.ggml_new_tensor_4d(self.ptr, int(dtype), ne0, ne1, ne2, ne3), name)

    # -- graph factory ------------------------------------------------------

    def new_graph(self, size: Optional[int] = None, grads: bool = False) -> Graph:
        """Create a computation graph backed by this context."""
        return Graph(self, size=size, grads=grads)

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        if not self._closed and self.ptr is not None:
            if self._owns:
                _ggml.ggml_free(self.ptr)
            self._closed = True
            self.ptr = None

    def __enter__(self) -> "Context":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

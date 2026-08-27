"""Thin OO wrapper around ``ggml_context``."""

from __future__ import annotations

import ctypes
from typing import Optional

import ggml as _ggml

from .errors import nonnull
from .types import DType
from .tensor import Tensor
from .graph import Graph, graph_overhead_custom


def tensor_overhead() -> int:
    """Return bytes of context arena needed per tensor (``ggml_tensor_overhead``)."""
    return _ggml.ggml_tensor_overhead()


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
        # ggml borrows external arena memory for the entire context lifetime.
        # Retain the object passed by the caller so ctypes-owned buffers are not
        # garbage-collected while the native context still points into them.
        self._mem_buffer = mem_buffer
        self.no_alloc = no_alloc
        self._owns = True
        self._closed = False

    @classmethod
    def for_tensors(
        cls,
        n_tensors: int,
        *,
        graph_size: Optional[int] = None,
        grads: bool = False,
        extra: int = 0,
    ) -> "Context":
        """Create a ``no_alloc`` context sized for ``n_tensors`` plus a graph.

        Does the ``ggml_tensor_overhead() * n + ggml_graph_overhead_custom(...)``
        arithmetic that every graph-building program otherwise repeats. Pass
        ``graph_size`` to reserve a graph of that many nodes (``None`` reserves
        none), and ``extra`` for any additional bytes you need.

        Only for metadata: a context holding tensor *data* (``no_alloc=False``)
        must be sized by the caller with :meth:`Context`.
        """
        if n_tensors < 0:
            raise ValueError(f"n_tensors must be non-negative, got {n_tensors}")
        mem_size = tensor_overhead() * n_tensors + extra
        if graph_size is not None:
            # A graph reserves metadata for its own nodes on top of the graph
            # struct, so count those tensors too.
            mem_size += tensor_overhead() * graph_size
            mem_size += graph_overhead_custom(graph_size, grads)
        return cls(mem_size, no_alloc=True)

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
        self._mem_buffer = None
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

    def new_tensor(self, dtype: "DType", *ne: int, name: Optional[str] = None) -> Tensor:
        """Create a tensor of up to four dimensions (``ggml_new_tensor``).

        ``ne`` is in ggml order, so build one from a numpy array with
        ``ctx.new_tensor(F32, *reversed(array.shape))``.
        """
        if not 1 <= len(ne) <= 4:
            raise ValueError(f"a tensor has between 1 and 4 dimensions, got {len(ne)}")
        dims = (ctypes.c_int64 * len(ne))(*(int(n) for n in ne))
        return self._new_tensor(
            _ggml.ggml_new_tensor(self.ptr, int(dtype), len(ne), dims), name
        )

    # -- tensor lookup ------------------------------------------------------
    #
    # ggml stores every tensor created in a context in an internal list. These
    # let you fetch them back by name (e.g. weights filled in by
    # ``gguf_init_from_file``) or walk all of them.

    def get_tensor(self, name: str) -> Optional[Tensor]:
        """Return the tensor named ``name`` in this context, or ``None``.

        Wraps ``ggml_get_tensor``. Used to bind GGUF weight tensors by name.
        """
        ptr = _ggml.ggml_get_tensor(self.ptr, name.encode())
        return Tensor(self, ptr) if ptr else None

    def first_tensor(self) -> Optional[Tensor]:
        """Return the first tensor in this context (``ggml_get_first_tensor``)."""
        ptr = _ggml.ggml_get_first_tensor(self.ptr)
        return Tensor(self, ptr) if ptr else None

    def next_tensor(self, tensor: Tensor) -> Optional[Tensor]:
        """Return the tensor after ``tensor`` (``ggml_get_next_tensor``)."""
        ptr = _ggml.ggml_get_next_tensor(self.ptr, tensor.ptr)
        return Tensor(self, ptr) if ptr else None

    def tensors(self):
        """Iterate over every tensor in this context, in creation order."""
        cur = self.first_tensor()
        while cur is not None:
            yield cur
            cur = self.next_tensor(cur)

    def ref(self, tensor: Tensor) -> Tensor:
        """Return ``tensor`` rebound to this context, sharing its ``ggml_tensor``.

        Op methods allocate their result in the receiver's context, so a graph
        built here that reads a tensor created elsewhere (model weights, a KV
        cache) must go through this first, or the result nodes would land in the
        weight context and exhaust it. The underlying ``ggml_tensor`` is shared,
        not copied, and a graph may freely span both contexts as long as both
        outlive it.
        """
        return Tensor(self, tensor.ptr)

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
            self._mem_buffer = None

    def __enter__(self) -> "Context":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

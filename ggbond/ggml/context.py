"""Thin OO wrapper around ``ggml_context``."""

from __future__ import annotations

from typing import Optional

from ._ffi import _ggml, _utils, nonnull
from .types import DType
from .tensor import Tensor
from .graph import Graph


class Context:
    """Owns a ``ggml_context``: the arena that holds tensor metadata (and,
    when ``no_alloc=False``, tensor data).

    Memory can be specified directly via ``mem_size`` (bytes) or estimated from
    ``n_tensors`` / ``n_graphs`` (metadata overhead only). When
    ``no_alloc=False`` the context also stores tensor *data*, so its size cannot
    be guessed from tensor counts alone -- an explicit ``mem_size`` is required.

    A :class:`Context` must outlive every :class:`Tensor` / :class:`Graph` it
    produces, because ``ggml_free`` releases all of their metadata at once.
    Tensors hold a strong reference back to their context to enforce this.
    """

    def __init__(
        self,
        *,
        mem_size: Optional[int] = None,
        n_tensors: Optional[int] = None,
        n_graphs: int = 0,
        no_alloc: bool = True,
        mem_buffer=None,
    ):
        if mem_size is None:
            if not no_alloc:
                raise ValueError(
                    "no_alloc=False stores tensor data in the context; "
                    "an explicit mem_size is required"
                )
            if n_tensors is None:
                raise ValueError("provide either mem_size or n_tensors")
            mem_size = (
                _ggml.ggml_tensor_overhead() * (n_tensors + 8)
                + _ggml.ggml_graph_overhead() * n_graphs
            )

        params = _ggml.ggml_init_params(mem_size, mem_buffer, no_alloc)
        self.ptr = nonnull(_ggml.ggml_init(params), "ggml_init")
        self.no_alloc = no_alloc
        self._closed = False

    # -- tensor factories ---------------------------------------------------

    def new_tensor(self, dtype: "DType", *shape: int, name: Optional[str] = None) -> Tensor:
        """Create a new tensor with dimensions in **ggml order** ``(ne0, ne1, ...)``.

        Note this is the reverse of numpy's row-major order. See
        :attr:`Tensor.ne` vs :attr:`Tensor.shape`.
        """
        t = int(dtype)
        n = len(shape)
        if n == 1:
            ptr = _ggml.ggml_new_tensor_1d(self.ptr, t, shape[0])
        elif n == 2:
            ptr = _ggml.ggml_new_tensor_2d(self.ptr, t, shape[0], shape[1])
        elif n == 3:
            ptr = _ggml.ggml_new_tensor_3d(self.ptr, t, shape[0], shape[1], shape[2])
        elif n == 4:
            ptr = _ggml.ggml_new_tensor_4d(self.ptr, t, shape[0], shape[1], shape[2], shape[3])
        else:
            raise ValueError("ggml tensors support 1 to 4 dimensions")
        tensor = Tensor(self, nonnull(ptr, "ggml_new_tensor"))
        if name is not None:
            tensor.name = name
        return tensor

    def tensor_from_numpy(self, x, name: Optional[str] = None) -> Tensor:
        """Create a tensor with data copied from a numpy array.

        Only usable when ``no_alloc=False`` (the context must own the data).
        Accepts numpy-order shape; ``ggml.utils.from_numpy`` reverses the
        dimensions internally.
        """
        if self.no_alloc:
            raise ValueError("tensor_from_numpy requires a context with no_alloc=False")
        ptr = nonnull(_utils.from_numpy(x, self.ptr), "from_numpy")
        tensor = Tensor(self, ptr)
        if name is not None:
            tensor.name = name
        return tensor

    # -- graph factory ------------------------------------------------------

    def new_graph(self, size: Optional[int] = None, grads: bool = False) -> Graph:
        """Create a computation graph backed by this context."""
        return Graph(self, size=size, grads=grads)

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        if not self._closed and self.ptr is not None:
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

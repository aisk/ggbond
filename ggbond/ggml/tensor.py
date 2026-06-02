"""Thin OO wrapper around ``ggml_tensor``.

Each :class:`Tensor` wraps ``(Context, ggml_tensor_p)``. Operation methods map
1:1 to ggml functions and return a new :class:`Tensor` bound to the same
context. Building a graph and computing it remain explicit -- nothing here is
lazy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from ._ffi import _ggml, _utils
from .types import DType, dtype_to_numpy

if TYPE_CHECKING:
    from .context import Context
    from .backend import Backend

# Simple ops generated onto the class: name -> ggml function name.
_UNARY_OPS = {
    "relu": "ggml_relu",
    "gelu": "ggml_gelu",
    "silu": "ggml_silu",
    "tanh": "ggml_tanh",
    "sigmoid": "ggml_sigmoid",
    "soft_max": "ggml_soft_max",
    "cont": "ggml_cont",
    "transpose": "ggml_transpose",
}

_BINARY_OPS = {
    "add": "ggml_add",
    "sub": "ggml_sub",
    "mul": "ggml_mul",
    "div": "ggml_div",
    "mul_mat": "ggml_mul_mat",
    "get_rows": "ggml_get_rows",
}


class Tensor:
    """A ggml tensor bound to a :class:`Context`.

    Shape conventions (the most common source of confusion):

    * :attr:`ne` -- ggml order ``(ne0, ne1, ...)`` (column-major-ish).
    * :attr:`shape` -- numpy order ``(..., ne1, ne0)``, matching :meth:`numpy`.

    ``mul_mat(a, b)`` requires ``a.ne[0] == b.ne[0]`` (the shared inner dim).
    """

    __slots__ = ("ctx", "ptr")

    def __init__(self, ctx: "Context", ptr):
        # Strong reference to the context keeps it (and this tensor's metadata)
        # alive; op methods use ctx.ptr as the ggml context argument.
        self.ctx = ctx
        self.ptr = ptr

    # -- internals ----------------------------------------------------------

    def _wrap(self, ptr) -> "Tensor":
        return Tensor(self.ctx, ptr)

    @property
    def _c(self):
        return self.ctx.ptr

    # -- properties ---------------------------------------------------------

    @property
    def ne(self) -> tuple:
        """Dimensions in ggml order ``(ne0, ne1, ...)``."""
        n = _ggml.ggml_n_dims(self.ptr)
        return tuple(self.ptr.contents.ne[:n])

    @property
    def shape(self) -> tuple:
        """Dimensions in numpy order (reverse of :attr:`ne`)."""
        return tuple(reversed(self.ne))

    @property
    def nb(self) -> tuple:
        """Byte strides in ggml order."""
        n = _ggml.ggml_n_dims(self.ptr)
        return tuple(self.ptr.contents.nb[:n])

    @property
    def dtype(self) -> "DType":
        return DType(self.ptr.contents.type)

    @property
    def ndims(self) -> int:
        return _ggml.ggml_n_dims(self.ptr)

    @property
    def nelements(self) -> int:
        return _ggml.ggml_nelements(self.ptr)

    @property
    def nbytes(self) -> int:
        return _ggml.ggml_nbytes(self.ptr)

    @property
    def op(self) -> int:
        return self.ptr.contents.op

    @property
    def name(self) -> str:
        return _ggml.ggml_get_name(self.ptr).decode()

    @name.setter
    def name(self, value: str) -> None:
        _ggml.ggml_set_name(self.ptr, value.encode())

    # -- scalar-parameter ops (cannot be table-generated) -------------------

    def scale(self, s: float) -> "Tensor":
        """Element-wise multiply by a scalar ``s``."""
        return self._wrap(_ggml.ggml_scale(self._c, self.ptr, float(s)))

    def norm(self, eps: float = 1e-5) -> "Tensor":
        """Layer normalization with epsilon ``eps``."""
        return self._wrap(_ggml.ggml_norm(self._c, self.ptr, float(eps)))

    def rms_norm(self, eps: float = 1e-5) -> "Tensor":
        """RMS normalization with epsilon ``eps``."""
        return self._wrap(_ggml.ggml_rms_norm(self._c, self.ptr, float(eps)))

    # -- shape ops ----------------------------------------------------------

    def reshape(self, *ne: int) -> "Tensor":
        """Reshape to dimensions in **ggml order** ``(ne0, ne1, ...)``."""
        n = len(ne)
        if n == 1:
            ptr = _ggml.ggml_reshape_1d(self._c, self.ptr, ne[0])
        elif n == 2:
            ptr = _ggml.ggml_reshape_2d(self._c, self.ptr, ne[0], ne[1])
        elif n == 3:
            ptr = _ggml.ggml_reshape_3d(self._c, self.ptr, ne[0], ne[1], ne[2])
        elif n == 4:
            ptr = _ggml.ggml_reshape_4d(self._c, self.ptr, ne[0], ne[1], ne[2], ne[3])
        else:
            raise ValueError("reshape supports 1 to 4 dimensions")
        return self._wrap(ptr)

    def permute(self, axis0: int, axis1: int, axis2: int, axis3: int) -> "Tensor":
        return self._wrap(_ggml.ggml_permute(self._c, self.ptr, axis0, axis1, axis2, axis3))

    def view_1d(self, ne0: int, offset: int = 0) -> "Tensor":
        return self._wrap(_ggml.ggml_view_1d(self._c, self.ptr, ne0, offset))

    def view_2d(self, ne0: int, ne1: int, nb1: int, offset: int = 0) -> "Tensor":
        return self._wrap(_ggml.ggml_view_2d(self._c, self.ptr, ne0, ne1, nb1, offset))

    def view_3d(self, ne0: int, ne1: int, ne2: int, nb1: int, nb2: int, offset: int = 0) -> "Tensor":
        return self._wrap(_ggml.ggml_view_3d(self._c, self.ptr, ne0, ne1, ne2, nb1, nb2, offset))

    # -- numpy I/O ----------------------------------------------------------

    def numpy(self, backend: "Optional[Backend]" = None) -> np.ndarray:
        """Read the tensor data as a numpy array (numpy-order shape).

        * ``backend`` is ``None`` or a CPU backend: zero-copy view via
          ``ggml.utils.to_numpy`` (requires a valid data pointer, i.e. a
          ``no_alloc=False`` context or CPU-resident buffer).
        * ``backend`` is a non-CPU (e.g. CUDA): the data pointer is NULL, so
          copy to host via ``ggml_backend_tensor_get``.
        """
        if backend is None or backend.is_cpu:
            return _utils.to_numpy(self.ptr)
        out = np.empty(self.shape, dtype=dtype_to_numpy(self.dtype))
        backend.tensor_get(self, out)
        return out

    def set_numpy(self, x, backend: "Optional[Backend]" = None) -> None:
        """Write ``x`` into this (already-allocated) tensor.

        * ``backend`` is ``None`` or CPU: in-place assignment through a
          ``to_numpy`` view (requires a valid data pointer).
        * non-CPU backend: upload via ``ggml_backend_tensor_set``.
        """
        if backend is None or backend.is_cpu:
            _utils.to_numpy(self.ptr)[:] = x
        else:
            backend.tensor_set(self, x)

    def __repr__(self) -> str:
        return f"<ggml.Tensor name={self.name!r} ne={self.ne} dtype={self.dtype.name}>"


def _make_unary(fn_name: str):
    fn = getattr(_ggml, fn_name)

    def method(self) -> "Tensor":
        return self._wrap(fn(self._c, self.ptr))

    method.__name__ = fn_name.removeprefix("ggml_")
    method.__doc__ = f"Apply ``{fn_name}`` element-wise, returning a new tensor."
    return method


def _make_binary(fn_name: str):
    fn = getattr(_ggml, fn_name)

    def method(self, other: "Tensor") -> "Tensor":
        return self._wrap(fn(self._c, self.ptr, other.ptr))

    method.__name__ = fn_name.removeprefix("ggml_")
    method.__doc__ = f"Apply ``{fn_name}(self, other)``, returning a new tensor."
    return method


for _name, _fn in _UNARY_OPS.items():
    setattr(Tensor, _name, _make_unary(_fn))
for _name, _fn in _BINARY_OPS.items():
    setattr(Tensor, _name, _make_binary(_fn))

del _name, _fn

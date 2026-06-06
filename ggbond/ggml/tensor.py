"""Thin OO wrapper around ``ggml_tensor``.

Each :class:`Tensor` wraps ``(Context, ggml_tensor_p)``. Operation methods map
1:1 to ggml functions and return a new :class:`Tensor` bound to the same
context. Building a graph and computing it remain explicit -- nothing here is
lazy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._ffi import _ggml
from .types import DType

if TYPE_CHECKING:
    from .context import Context

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

    :attr:`ne` is the ggml-order shape ``(ne0, ne1, ...)`` (column-major-ish,
    the reverse of numpy's row-major order).

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
    def nb(self) -> tuple:
        """Byte strides in ggml order."""
        n = _ggml.ggml_n_dims(self.ptr)
        return tuple(self.ptr.contents.nb[:n])

    @property
    def dtype(self) -> "DType":
        return DType(self.ptr.contents.type)

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

    def reshape_1d(self, ne0: int) -> "Tensor":
        return self._wrap(_ggml.ggml_reshape_1d(self._c, self.ptr, ne0))

    def reshape_2d(self, ne0: int, ne1: int) -> "Tensor":
        return self._wrap(_ggml.ggml_reshape_2d(self._c, self.ptr, ne0, ne1))

    def reshape_3d(self, ne0: int, ne1: int, ne2: int) -> "Tensor":
        return self._wrap(_ggml.ggml_reshape_3d(self._c, self.ptr, ne0, ne1, ne2))

    def reshape_4d(self, ne0: int, ne1: int, ne2: int, ne3: int) -> "Tensor":
        return self._wrap(_ggml.ggml_reshape_4d(self._c, self.ptr, ne0, ne1, ne2, ne3))

    def permute(self, axis0: int, axis1: int, axis2: int, axis3: int) -> "Tensor":
        return self._wrap(_ggml.ggml_permute(self._c, self.ptr, axis0, axis1, axis2, axis3))

    def view_1d(self, ne0: int, offset: int = 0) -> "Tensor":
        return self._wrap(_ggml.ggml_view_1d(self._c, self.ptr, ne0, offset))

    def view_2d(self, ne0: int, ne1: int, nb1: int, offset: int = 0) -> "Tensor":
        return self._wrap(_ggml.ggml_view_2d(self._c, self.ptr, ne0, ne1, nb1, offset))

    def view_3d(self, ne0: int, ne1: int, ne2: int, nb1: int, nb2: int, offset: int = 0) -> "Tensor":
        return self._wrap(_ggml.ggml_view_3d(self._c, self.ptr, ne0, ne1, ne2, nb1, nb2, offset))

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

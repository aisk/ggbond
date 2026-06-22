"""Thin OO wrapper around ``ggml_tensor``.

Each :class:`Tensor` wraps ``(Context, ggml_tensor_p)``. Operation methods map
1:1 to ggml functions and return a new :class:`Tensor` bound to the same
context. Building a graph and computing it remain explicit -- nothing here is
lazy.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import numpy as np

import ggml as _ggml

from .types import DType, dtype_to_numpy

if TYPE_CHECKING:
    from .context import Context

# Pooling ops for :meth:`Tensor.pool_1d` (mirror ggml's ``enum ggml_op_pool``).
POOL_MAX = _ggml.GGML_OP_POOL_MAX
POOL_AVG = _ggml.GGML_OP_POOL_AVG

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
    "cpy": "ggml_cpy",
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
    def element_size(self) -> int:
        """Size in bytes of one element of this tensor's dtype (``ggml_element_size``)."""
        return _ggml.ggml_element_size(self.ptr)

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

    def norm(self, eps: float) -> "Tensor":
        """Layer normalization with epsilon ``eps`` (``ggml_norm``)."""
        return self._wrap(_ggml.ggml_norm(self._c, self.ptr, float(eps)))

    def rms_norm(self, eps: float) -> "Tensor":
        """RMS normalization with epsilon ``eps`` (``ggml_rms_norm``)."""
        return self._wrap(_ggml.ggml_rms_norm(self._c, self.ptr, float(eps)))

    def diag_mask_inf(self, n_past: int) -> "Tensor":
        """Mask elements above the diagonal with ``-inf`` (``ggml_diag_mask_inf``).

        ``n_past`` shifts the diagonal so the first ``n_past`` columns stay
        unmasked -- this is the causal attention mask for cached keys/values.
        """
        return self._wrap(_ggml.ggml_diag_mask_inf(self._c, self.ptr, int(n_past)))

    # -- pooling ------------------------------------------------------------

    def pool_1d(self, op: int, k0: int, s0: int, p0: int) -> "Tensor":
        """1D pooling: kernel ``k0``, stride ``s0``, padding ``p0`` (``ggml_pool_1d``).

        ``op`` is a ggml pooling op (:data:`POOL_MAX` / :data:`POOL_AVG`).
        """
        return self._wrap(_ggml.ggml_pool_1d(self._c, self.ptr, int(op), k0, s0, p0))

    # -- shape ops ----------------------------------------------------------

    def reshape_1d(self, ne0: int) -> "Tensor":
        return self._wrap(_ggml.ggml_reshape_1d(self._c, self.ptr, ne0))

    def reshape_2d(self, ne0: int, ne1: int) -> "Tensor":
        return self._wrap(_ggml.ggml_reshape_2d(self._c, self.ptr, ne0, ne1))

    def reshape_3d(self, ne0: int, ne1: int, ne2: int) -> "Tensor":
        return self._wrap(_ggml.ggml_reshape_3d(self._c, self.ptr, ne0, ne1, ne2))

    def reshape_4d(self, ne0: int, ne1: int, ne2: int, ne3: int) -> "Tensor":
        return self._wrap(_ggml.ggml_reshape_4d(self._c, self.ptr, ne0, ne1, ne2, ne3))

    def cont_2d(self, ne0: int, ne1: int) -> "Tensor":
        """Make contiguous and reshape to 2D ``(ne0, ne1)`` (``ggml_cont_2d``)."""
        return self._wrap(_ggml.ggml_cont_2d(self._c, self.ptr, ne0, ne1))

    def cont_3d(self, ne0: int, ne1: int, ne2: int) -> "Tensor":
        """Make contiguous and reshape to 3D ``(ne0, ne1, ne2)`` (``ggml_cont_3d``)."""
        return self._wrap(_ggml.ggml_cont_3d(self._c, self.ptr, ne0, ne1, ne2))

    def permute(self, axis0: int, axis1: int, axis2: int, axis3: int) -> "Tensor":
        return self._wrap(_ggml.ggml_permute(self._c, self.ptr, axis0, axis1, axis2, axis3))

    def view_1d(self, ne0: int, offset: int = 0) -> "Tensor":
        return self._wrap(_ggml.ggml_view_1d(self._c, self.ptr, ne0, offset))

    def view_2d(self, ne0: int, ne1: int, nb1: int, offset: int = 0) -> "Tensor":
        return self._wrap(_ggml.ggml_view_2d(self._c, self.ptr, ne0, ne1, nb1, offset))

    def view_3d(self, ne0: int, ne1: int, ne2: int, nb1: int, nb2: int, offset: int = 0) -> "Tensor":
        return self._wrap(_ggml.ggml_view_3d(self._c, self.ptr, ne0, ne1, ne2, nb1, nb2, offset))

    # -- graph I/O flags ----------------------------------------------------

    def set_input(self) -> "Tensor":
        """Mark this tensor as a graph input (``ggml_set_input``).

        Tells the allocator not to reuse this tensor's buffer for intermediate
        results, so a value uploaded after ``alloc_graph`` survives the compute.
        Required for inputs when using :class:`GAllocr` / :class:`Scheduler`.
        Returns ``self`` so it can be chained.
        """
        _ggml.ggml_set_input(self.ptr)
        return self

    def set_output(self) -> "Tensor":
        """Mark this tensor as a graph output (``ggml_set_output``).

        Keeps the allocator from overwriting this tensor's buffer, so it can be
        read back after compute. Returns ``self``.
        """
        _ggml.ggml_set_output(self.ptr)
        return self

    # -- data transfer ------------------------------------------------------
    #
    # ggml's sync transfer (``ggml_backend_tensor_set/get``) is keyed on the
    # tensor's own buffer and takes no backend handle, so it lives here rather
    # than on Backend. The tensor must already be allocated (e.g. via
    # ``Backend.alloc_ctx_tensors``, ``GAllocr.alloc_graph``, or
    # ``Scheduler.alloc_graph``). This is where numpy enters the wrapper.

    def set(self, x) -> None:
        """Upload ``x`` (any array-like, incl. a Python list) into this tensor.

        ``x`` is coerced to this tensor's dtype and made contiguous; its total
        byte size must then match the tensor's. Wraps ``ggml_backend_tensor_set``.
        """
        arr = np.ascontiguousarray(x, dtype=dtype_to_numpy(self.dtype))
        if arr.nbytes != self.nbytes:
            raise ValueError(
                f"set: source has {arr.nbytes} bytes but tensor needs "
                f"{self.nbytes} (ne={self.ne}, dtype={self.dtype.name})"
            )
        ptr = arr.ctypes.data_as(ctypes.c_void_p)
        _ggml.ggml_backend_tensor_set(self.ptr, ptr, 0, self.nbytes)

    def set_raw(self, data, offset: int = 0) -> None:
        """Upload raw bytes into this tensor verbatim (no dtype coercion).

        ``data`` is any bytes-like buffer (``bytes``, ``bytearray``,
        ``memoryview``, ...) whose layout already matches the tensor's. Use this
        for weights read straight from a file -- including quantized dtypes that
        have no numpy equivalent. ``offset`` is the byte offset into the tensor.
        Wraps ``ggml_backend_tensor_set``.
        """
        mv = memoryview(data).cast("B")
        if offset + mv.nbytes > self.nbytes:
            raise ValueError(
                f"set_raw: {mv.nbytes} bytes at offset {offset} exceed tensor "
                f"size {self.nbytes} (ne={self.ne}, dtype={self.dtype.name})"
            )
        ptr = (ctypes.c_char * mv.nbytes).from_buffer_copy(mv)
        _ggml.ggml_backend_tensor_set(self.ptr, ptr, offset, mv.nbytes)

    def get(self, out: "np.ndarray") -> "np.ndarray":
        """Download this tensor into the pre-allocated, contiguous numpy array ``out``.

        ``out`` must be contiguous and exactly the tensor's byte size. Wraps
        ``ggml_backend_tensor_get`` and returns ``out``.
        """
        if not out.flags["C_CONTIGUOUS"]:
            raise ValueError("get: out must be a C-contiguous array")
        if out.nbytes != self.nbytes:
            raise ValueError(
                f"get: out has {out.nbytes} bytes but tensor holds "
                f"{self.nbytes} (ne={self.ne}, dtype={self.dtype.name})"
            )
        ptr = out.ctypes.data_as(ctypes.c_void_p)
        _ggml.ggml_backend_tensor_get(self.ptr, ptr, 0, self.nbytes)
        return out

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

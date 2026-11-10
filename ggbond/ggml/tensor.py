"""Thin OO wrapper around ``ggml_tensor``.

Each :class:`Tensor` wraps ``(Context, ggml_tensor_p)``, where the context is
the one whose arena holds the tensor's metadata. Building a graph and computing
it remain explicit -- nothing here is lazy.

Every ggml op exists in two spellings generated from the same tables:
``ctx.add(a, b)`` on :class:`Context`, which is ``ggml_add(ctx, a, b)`` with the
target context explicit, and ``a.add(b)`` on :class:`Tensor`, sugar for
``a.ctx.add(a, b)``. The tables are :data:`_UNARY_OPS` for ops taking one
tensor, :data:`_BINARY_OPS` for ops taking two, and :data:`_PARAM_OPS` for
everything with extra arguments (scalars, optional tensors, enums). See
``_make_op`` for the argument vocabulary.
"""

from __future__ import annotations

import ctypes
import numbers
from enum import IntEnum, IntFlag
from typing import TYPE_CHECKING, Optional

import numpy as np

import ggml as _ggml

from .errors import GGMLError
from .types import DType, F32, I32, dtype_to_numpy

if TYPE_CHECKING:
    from .context import Context



class PoolOp(IntEnum):
    """Pooling op for :meth:`Tensor.pool_1d` / :meth:`Tensor.pool_2d`
    (``enum ggml_op_pool``)."""

    MAX = _ggml.GGML_OP_POOL_MAX
    AVG = _ggml.GGML_OP_POOL_AVG


class RopeType(IntEnum):
    """Rope mode for :meth:`Tensor.rope` / :meth:`Tensor.rope_ext`
    (``GGML_ROPE_TYPE_*``)."""

    NORMAL = _ggml.GGML_ROPE_TYPE_NORMAL
    NEOX = _ggml.GGML_ROPE_TYPE_NEOX
    MROPE = _ggml.GGML_ROPE_TYPE_MROPE
    VISION = _ggml.GGML_ROPE_TYPE_VISION
    IMROPE = _ggml.GGML_ROPE_TYPE_IMROPE


# The operation that produced a tensor (``enum ggml_op``), as reported by
# :attr:`Tensor.op`. Built from the bindings since it has ~100 members; the
# ``GGML_OP_POOL_MAX``/``AVG``/``COUNT`` constants belong to ``ggml_op_pool``.
_NOT_OPS = {"GGML_OP_COUNT", "GGML_OP_POOL_MAX", "GGML_OP_POOL_AVG", "GGML_OP_POOL_COUNT"}
Op = IntEnum(
    "Op",
    sorted(
        ((name[len("GGML_OP_"):], getattr(_ggml, name))
         for name in dir(_ggml)
         if name.startswith("GGML_OP_") and name not in _NOT_OPS),
        key=lambda item: item[1],
    ),
    module=__name__,
)
Op.__doc__ = "The ggml operation that produced a tensor (``enum ggml_op``)."



class SortOrder(IntEnum):
    """Order for :meth:`Tensor.argsort` (``enum ggml_sort_order``)."""

    ASC = _ggml.GGML_SORT_ORDER_ASC
    DESC = _ggml.GGML_SORT_ORDER_DESC


class ScaleMode(IntEnum):
    """Interpolation for :meth:`Tensor.upscale` (``enum ggml_scale_mode``)."""

    NEAREST = _ggml.GGML_SCALE_MODE_NEAREST
    BILINEAR = _ggml.GGML_SCALE_MODE_BILINEAR
    BICUBIC = _ggml.GGML_SCALE_MODE_BICUBIC


class TensorFlag(IntFlag):
    """Bits of :attr:`Tensor.flags` (``enum ggml_tensor_flag``)."""

    INPUT = _ggml.GGML_TENSOR_FLAG_INPUT
    OUTPUT = _ggml.GGML_TENSOR_FLAG_OUTPUT
    PARAM = _ggml.GGML_TENSOR_FLAG_PARAM
    LOSS = _ggml.GGML_TENSOR_FLAG_LOSS
    COMPUTE = _ggml.GGML_TENSOR_FLAG_COMPUTE


# Python scalars accepted by the arithmetic operators (bool and numpy scalars
# included).
_SCALAR = numbers.Real

# Argument kinds understood by the op tables.
_TENSOR = "tensor"  # required Tensor operand
_TENSOR_OPT = "tensor?"  # optional Tensor operand; None becomes a NULL pointer
_INT = "int"
_FLOAT = "float"
_BOOL = "bool"
_DTYPE = "dtype"
_POOL_OP = "pool_op"
_ROPE_TYPE = "rope_type"
_SORT_ORDER = "sort_order"
_SCALE_MODE = "scale_mode"

# How each kind is converted to the ctypes argument, and how it is annotated.
_ARG_CONV = {
    _TENSOR: "{0}.ptr",
    _TENSOR_OPT: "({0}.ptr if {0} is not None else None)",
    _INT: "int({0})",
    _FLOAT: "float({0})",
    _BOOL: "bool({0})",
    _DTYPE: "int({0})",
    _POOL_OP: "int({0})",
    _ROPE_TYPE: "int({0})",
    _SORT_ORDER: "int({0})",
    _SCALE_MODE: "int({0})",
}

_ARG_ANNOTATION = {
    _TENSOR: "Tensor",
    _TENSOR_OPT: "Optional[Tensor]",
    _INT: "int",
    _FLOAT: "float",
    _BOOL: "bool",
    _DTYPE: "DType",
    _POOL_OP: "PoolOp",
    _ROPE_TYPE: "RopeType",
    _SORT_ORDER: "SortOrder",
    _SCALE_MODE: "ScaleMode",
}

# Ops taking one tensor: name -> ggml function name.
_UNARY_OPS = {
    "relu": "ggml_relu",
    "gelu": "ggml_gelu",
    "gelu_quick": "ggml_gelu_quick",
    "gelu_erf": "ggml_gelu_erf",
    "swiglu": "ggml_swiglu",
    "silu": "ggml_silu",
    "elu": "ggml_elu",
    "step": "ggml_step",
    "hardswish": "ggml_hardswish",
    "hardsigmoid": "ggml_hardsigmoid",
    "tanh": "ggml_tanh",
    "sigmoid": "ggml_sigmoid",
    "soft_max": "ggml_soft_max",
    "abs": "ggml_abs",
    "neg": "ggml_neg",
    "sqr": "ggml_sqr",
    "sqrt": "ggml_sqrt",
    "log": "ggml_log",
    "exp": "ggml_exp",
    "sin": "ggml_sin",
    "cos": "ggml_cos",
    "floor": "ggml_floor",
    "ceil": "ggml_ceil",
    "round": "ggml_round",
    "trunc": "ggml_trunc",
    "sgn": "ggml_sgn",
    "softplus": "ggml_softplus",
    "expm1": "ggml_expm1",
    "cumsum": "ggml_cumsum",
    "sum": "ggml_sum",
    "sum_rows": "ggml_sum_rows",
    "mean": "ggml_mean",
    "argmax": "ggml_argmax",
    "dup": "ggml_dup",
    "cont": "ggml_cont",
    "transpose": "ggml_transpose",
}

# Ops taking two tensors: name -> ggml function name.
_BINARY_OPS = {
    "add": "ggml_add",
    "add1": "ggml_add1",
    "sub": "ggml_sub",
    "mul": "ggml_mul",
    "div": "ggml_div",
    "mul_mat": "ggml_mul_mat",
    "out_prod": "ggml_out_prod",
    "get_rows": "ggml_get_rows",
    "cpy": "ggml_cpy",
    "repeat": "ggml_repeat",
    "reshape": "ggml_reshape",
}

# Ops with extra arguments: name -> (ggml function, params, doc). Each param is
# ``(name, kind)`` or ``(name, kind, default)``, listed after ggml's ``ctx`` and
# ``a`` arguments. Docs say ``self`` for ``a``, the receiver of the Tensor form.
_PARAM_OPS = {
    # -- metadata clones (not graph nodes) --
    "dup_tensor": (
        "ggml_dup_tensor",
        (),
        "Create a new uninitialized tensor with this one's shape and dtype.",
    ),
    "view_tensor": (
        "ggml_view_tensor",
        (),
        "Create a tensor sharing this one's data, shape and dtype.",
    ),
    # -- scalar-parameter math --
    "scale": (
        "ggml_scale",
        (("s", _FLOAT),),
        "Element-wise multiply by the scalar ``s``.",
    ),
    "scale_bias": (
        "ggml_scale_bias",
        (("s", _FLOAT), ("b", _FLOAT)),
        "Element-wise ``self * s + b``.",
    ),
    "clamp": (
        "ggml_clamp",
        (("min", _FLOAT), ("max", _FLOAT)),
        "Clamp every element into ``[min, max]``.\n\n"
        "ggml has no out-of-place clamp: this writes into ``self`` and returns a\n"
        "view of it, so anything else reading ``self`` sees the clamped values.",
    ),
    "leaky_relu": (
        "ggml_leaky_relu",
        (("negative_slope", _FLOAT), ("inplace", _BOOL, False)),
        "Leaky ReLU with the given negative slope.",
    ),
    "argsort": (
        "ggml_argsort",
        (("order", _SORT_ORDER),),
        "Return the indices that sort each row. ``order`` is a :class:`SortOrder`.",
    ),
    "top_k": (
        "ggml_top_k",
        (("k", _INT),),
        "Return the indices of the ``k`` largest elements of each row.",
    ),
    "cast": (
        "ggml_cast",
        (("dtype", _DTYPE),),
        "Convert this tensor to ``dtype``.",
    ),
    # -- normalization --
    "norm": (
        "ggml_norm",
        (("eps", _FLOAT),),
        "Layer normalization with epsilon ``eps``.",
    ),
    "rms_norm": (
        "ggml_rms_norm",
        (("eps", _FLOAT),),
        "RMS normalization with epsilon ``eps``.",
    ),
    "l2_norm": (
        "ggml_l2_norm",
        (("eps", _FLOAT),),
        "L2 normalization with epsilon ``eps``.",
    ),
    "group_norm": (
        "ggml_group_norm",
        (("n_groups", _INT), ("eps", _FLOAT)),
        "Group normalization over ``n_groups`` groups with epsilon ``eps``.",
    ),
    # -- attention --
    "diag_mask_inf": (
        "ggml_diag_mask_inf",
        (("n_past", _INT),),
        "Mask elements above the diagonal with ``-inf``.\n\n"
        "``n_past`` shifts the diagonal so the first ``n_past`` columns stay\n"
        "unmasked -- this is the causal attention mask for cached keys/values.",
    ),
    "diag_mask_zero": (
        "ggml_diag_mask_zero",
        (("n_past", _INT),),
        "Mask elements above the diagonal with zero (see :meth:`diag_mask_inf`).",
    ),
    "soft_max_ext": (
        "ggml_soft_max_ext",
        (("mask", _TENSOR_OPT), ("scale", _FLOAT, 1.0), ("max_bias", _FLOAT, 0.0)),
        "Fused scaled, masked softmax: ``softmax(self * scale + mask)``.\n\n"
        "``mask`` may be None. ``max_bias`` > 0 enables ALiBi slopes.",
    ),
    "rope": (
        "ggml_rope",
        (("positions", _TENSOR), ("n_dims", _INT), ("mode", _ROPE_TYPE)),
        "Rotary position embedding over the first ``n_dims`` dimensions.\n\n"
        "``positions`` is an I32 tensor of token positions, one per sequence\n"
        "element. ``mode`` is a :class:`RopeType`.",
    ),
    "rope_ext": (
        "ggml_rope_ext",
        (
            ("positions", _TENSOR),
            ("freq_factors", _TENSOR_OPT),
            ("n_dims", _INT),
            ("mode", _ROPE_TYPE),
            ("n_ctx_orig", _INT),
            ("freq_base", _FLOAT, 10000.0),
            ("freq_scale", _FLOAT, 1.0),
            ("ext_factor", _FLOAT, 0.0),
            ("attn_factor", _FLOAT, 1.0),
            ("beta_fast", _FLOAT, 32.0),
            ("beta_slow", _FLOAT, 1.0),
        ),
        "Rotary position embedding with YaRN / NTK scaling parameters.\n\n"
        "``freq_factors`` may be None. ``n_ctx_orig`` is the context length the\n"
        "model was trained with, used by the ``ext_factor`` interpolation.",
    ),
    "flash_attn_ext": (
        "ggml_flash_attn_ext",
        (
            ("k", _TENSOR),
            ("v", _TENSOR),
            ("mask", _TENSOR_OPT),
            ("scale", _FLOAT),
            ("max_bias", _FLOAT, 0.0),
            ("logit_softcap", _FLOAT, 0.0),
        ),
        "Fused attention: ``softmax(self @ k^T * scale + mask) @ v``.\n\n"
        "``self`` is the query tensor, matching ggml's ``q`` argument.",
    ),
    # -- shape --
    "reshape_1d": ("ggml_reshape_1d", (("ne0", _INT),), "Reshape to 1D."),
    "reshape_2d": ("ggml_reshape_2d", (("ne0", _INT), ("ne1", _INT)), "Reshape to 2D."),
    "reshape_3d": (
        "ggml_reshape_3d",
        (("ne0", _INT), ("ne1", _INT), ("ne2", _INT)),
        "Reshape to 3D.",
    ),
    "reshape_4d": (
        "ggml_reshape_4d",
        (("ne0", _INT), ("ne1", _INT), ("ne2", _INT), ("ne3", _INT)),
        "Reshape to 4D.",
    ),
    "cont_1d": ("ggml_cont_1d", (("ne0", _INT),), "Make contiguous and reshape to 1D."),
    "cont_2d": (
        "ggml_cont_2d",
        (("ne0", _INT), ("ne1", _INT)),
        "Make contiguous and reshape to 2D.",
    ),
    "cont_3d": (
        "ggml_cont_3d",
        (("ne0", _INT), ("ne1", _INT), ("ne2", _INT)),
        "Make contiguous and reshape to 3D.",
    ),
    "cont_4d": (
        "ggml_cont_4d",
        (("ne0", _INT), ("ne1", _INT), ("ne2", _INT), ("ne3", _INT)),
        "Make contiguous and reshape to 4D.",
    ),
    "repeat_4d": (
        "ggml_repeat_4d",
        (("ne0", _INT), ("ne1", _INT), ("ne2", _INT), ("ne3", _INT)),
        "Tile this tensor up to the shape ``(ne0, ne1, ne2, ne3)``.",
    ),
    "concat": (
        "ggml_concat",
        (("other", _TENSOR), ("dim", _INT)),
        "Concatenate ``other`` onto this tensor along ggml dimension ``dim``.",
    ),
    "pad": (
        "ggml_pad",
        (("p0", _INT), ("p1", _INT), ("p2", _INT), ("p3", _INT)),
        "Zero-pad each ggml dimension by the given amount.",
    ),
    "pad_ext": (
        "ggml_pad_ext",
        (
            ("lp0", _INT), ("rp0", _INT), ("lp1", _INT), ("rp1", _INT),
            ("lp2", _INT), ("rp2", _INT), ("lp3", _INT), ("rp3", _INT),
        ),
        "Zero-pad each ggml dimension on the left (``lp``) and right (``rp``).",
    ),
    "upscale": (
        "ggml_upscale",
        (("scale_factor", _INT), ("mode", _SCALE_MODE)),
        "Upscale the first two ggml dimensions by ``scale_factor``.\n\n"
        "``mode`` is a :class:`ScaleMode`.",
    ),
    "set": (
        "ggml_set",
        (("b", _TENSOR), ("nb1", _INT), ("nb2", _INT), ("nb3", _INT), ("offset", _INT)),
        "Write ``b`` into a view of ``self`` with the given byte strides and offset.\n\n"
        "Returns a view of ``self``. This is the graph op ``ggml_set``; uploading\n"
        "host data is :meth:`write`.",
    ),
    "set_1d": (
        "ggml_set_1d",
        (("b", _TENSOR), ("offset", _INT)),
        "Write ``b`` into ``self`` starting at byte ``offset``; returns a view of ``self``.",
    ),
    "set_2d": (
        "ggml_set_2d",
        (("b", _TENSOR), ("nb1", _INT), ("offset", _INT)),
        "Write ``b`` into a 2D view of ``self`` with row stride ``nb1`` bytes.",
    ),
    "permute": (
        "ggml_permute",
        (("axis0", _INT), ("axis1", _INT), ("axis2", _INT), ("axis3", _INT)),
        "Permute the axes, producing a non-contiguous view.",
    ),
    "view_1d": (
        "ggml_view_1d",
        (("ne0", _INT), ("offset", _INT, 0)),
        "View ``ne0`` elements starting at byte ``offset``.",
    ),
    "view_2d": (
        "ggml_view_2d",
        (("ne0", _INT), ("ne1", _INT), ("nb1", _INT), ("offset", _INT, 0)),
        "View a 2D block with row stride ``nb1`` bytes.",
    ),
    "view_3d": (
        "ggml_view_3d",
        (
            ("ne0", _INT),
            ("ne1", _INT),
            ("ne2", _INT),
            ("nb1", _INT),
            ("nb2", _INT),
            ("offset", _INT, 0),
        ),
        "View a 3D block with the given byte strides.",
    ),
    "view_4d": (
        "ggml_view_4d",
        (
            ("ne0", _INT),
            ("ne1", _INT),
            ("ne2", _INT),
            ("ne3", _INT),
            ("nb1", _INT),
            ("nb2", _INT),
            ("nb3", _INT),
            ("offset", _INT, 0),
        ),
        "View a 4D block with the given byte strides.",
    ),
    # -- convolution and pooling --
    "conv_1d": (
        "ggml_conv_1d",
        (("data", _TENSOR), ("s0", _INT), ("p0", _INT), ("d0", _INT)),
        "1D convolution: stride ``s0``, padding ``p0``, dilation ``d0``.\n\n"
        "``self`` is the kernel and ``data`` the input, matching ggml's\n"
        "``ggml_conv_1d(ctx, kernel, data, ...)`` argument order.",
    ),
    "conv_2d": (
        "ggml_conv_2d",
        (
            ("data", _TENSOR),
            ("s0", _INT),
            ("s1", _INT),
            ("p0", _INT),
            ("p1", _INT),
            ("d0", _INT),
            ("d1", _INT),
        ),
        "2D convolution. ``self`` is the kernel and ``data`` the input.",
    ),
    "conv_transpose_1d": (
        "ggml_conv_transpose_1d",
        (("data", _TENSOR), ("s0", _INT), ("p0", _INT), ("d0", _INT)),
        "1D transposed convolution. ``self`` is the kernel and ``data`` the input.",
    ),
    "im2col": (
        "ggml_im2col",
        (
            ("data", _TENSOR),
            ("s0", _INT),
            ("s1", _INT),
            ("p0", _INT),
            ("p1", _INT),
            ("d0", _INT),
            ("d1", _INT),
            ("is_2d", _BOOL),
            ("dst_type", _DTYPE, F32),
        ),
        "Convert convolution windows to columns.\n\n"
        "``self`` is the kernel and ``data`` the input, matching ggml's\n"
        "``ggml_im2col(ctx, kernel, data, ...)`` argument order.",
    ),
    "pool_1d": (
        "ggml_pool_1d",
        (("op", _POOL_OP), ("k0", _INT), ("s0", _INT), ("p0", _INT)),
        "1D pooling: kernel ``k0``, stride ``s0``, padding ``p0``.\n\n"
        "``op`` is a :class:`PoolOp` (``PoolOp.MAX`` / ``PoolOp.AVG``).",
    ),
    "pool_2d": (
        "ggml_pool_2d",
        (
            ("op", _POOL_OP),
            ("k0", _INT),
            ("k1", _INT),
            ("s0", _INT),
            ("s1", _INT),
            ("p0", _FLOAT),
            ("p1", _FLOAT),
        ),
        "2D pooling. ``op`` is a :class:`PoolOp`.",
    ),
}


# -- preconditions --------------------------------------------------------------
#
# ggml validates op arguments with GGML_ASSERT, which aborts the whole process.
# These mirror the asserts of the most commonly misused ops so that a bad shape
# raises ValueError instead. Each check takes the op's arguments positionally.


def _check_broadcast(name: str):
    def check(self, other):
        if not _ggml.ggml_can_repeat(other.ptr, self.ptr):
            raise ValueError(
                f"{name}: cannot broadcast other ne={other.ne} onto ne={self.ne}; "
                f"every dimension of self must be a multiple of other's"
            )
    return check


def _check_repeat(self, other):
    if not _ggml.ggml_can_repeat(self.ptr, other.ptr):
        raise ValueError(
            f"repeat: cannot tile ne={self.ne} up to ne={other.ne}; every "
            f"dimension of other must be a multiple of self's"
        )


def _check_mul_mat(self, other):
    a, b = self.ne, other.ne
    if a[0] != b[0] or 0 in a[2:] or b[2] % a[2] or b[3] % a[3]:
        raise ValueError(
            f"mul_mat: needs a.ne[0] == b.ne[0] and b.ne[2], b.ne[3] divisible by "
            f"a.ne[2], a.ne[3]; got a.ne={a}, b.ne={b}"
        )


def _check_reshape(self, *ne):
    if not _ggml.ggml_is_contiguous(self.ptr):
        raise ValueError("reshape: the tensor is not contiguous; call cont() first")
    n = 1
    for d in ne:
        n *= d
    if n != self.nelements:
        raise ValueError(
            f"reshape: cannot reshape {self.nelements} elements (ne={self.ne}) to ne={ne}"
        )


def _check_cpy_like(name: str):
    def check(self, other):
        if self.nelements != other.nelements:
            raise ValueError(
                f"{name}: element counts differ: {self.nelements} (ne={self.ne}) vs "
                f"{other.nelements} (ne={other.ne})"
            )
    return check


def _check_get_rows(self, other):
    a, b = self.ne, other.ne
    if other.dtype != I32:
        raise ValueError(f"get_rows: row indices must be I32, got {other.dtype.name}")
    if a[2] != b[1] or a[3] != b[2] or b[3] != 1:
        raise ValueError(
            f"get_rows: needs a.ne[2] == b.ne[1], a.ne[3] == b.ne[2] and b.ne[3] == 1; "
            f"got a.ne={a}, b.ne={b}"
        )


def _check_concat(self, other, dim):
    if not 0 <= dim < 4:
        raise ValueError(f"concat: dim must be in [0, 4), got {dim}")
    if self.dtype != other.dtype:
        raise ValueError(
            f"concat: dtypes differ: {self.dtype.name} vs {other.dtype.name}"
        )
    a, b = self.ne, other.ne
    if any(a[d] != b[d] for d in range(4) if d != dim):
        raise ValueError(
            f"concat: shapes must match outside dim {dim}; got ne={a} and ne={b}"
        )


_CHECKS = {
    "reshape": _check_cpy_like("reshape"),
    "add": _check_broadcast("add"),
    "sub": _check_broadcast("sub"),
    "mul": _check_broadcast("mul"),
    "div": _check_broadcast("div"),
    "repeat": _check_repeat,
    "mul_mat": _check_mul_mat,
    "reshape_1d": _check_reshape,
    "reshape_2d": _check_reshape,
    "reshape_3d": _check_reshape,
    "reshape_4d": _check_reshape,
    "cpy": _check_cpy_like("cpy"),
    "get_rows": _check_get_rows,
    "concat": _check_concat,
}

# Tensors an op creates in its context, for ops that create more than one (the
# convolutions are composed of im2col, mul_mat and reshapes). Measured against
# ggml_used_mem; every other op creates exactly one.
_OP_TENSORS = {
    "conv_1d": 5,
    "conv_2d": 7,
}


class Tensor:
    """A ggml tensor and the :class:`Context` whose arena holds its metadata.

    :attr:`ne` is the ggml-order shape ``(ne0, ne1, ne2, ne3)`` exactly as ggml
    stores it; :attr:`shape` is the numpy-order view of it.

    Op methods are sugar for the same op on :attr:`ctx`: ``a.add(b)`` is
    ``a.ctx.add(a, b)``. To build a graph in a different context than the one
    holding ``a`` (model weights, a KV cache), call the op on that context:
    ``graph_ctx.mul_mat(weight, x)``.

    Two wrappers of the same ``ggml_tensor`` compare equal and hash alike.
    """

    __slots__ = ("ctx", "_ptr")

    def __init__(self, ctx: "Context", ptr):
        self.ctx = ctx
        self._ptr = ptr

    @property
    def ptr(self):
        """The ``ggml_tensor *``, or :class:`GGMLError` once its context is closed.

        ``ggml_free`` releases the metadata of every tensor in a context, so a
        tensor whose context was closed would otherwise dereference freed
        memory.
        """
        if self.ctx._ptr is None:
            raise GGMLError("tensor used after its context was closed")
        return self._ptr

    @property
    def address(self) -> int:
        """Address of the ``ggml_tensor`` struct (identity of the tensor)."""
        return ctypes.cast(self._ptr, ctypes.c_void_p).value

    def __eq__(self, other) -> bool:
        if not isinstance(other, Tensor):
            return NotImplemented
        return self.address == other.address

    def __hash__(self) -> int:
        return hash(self.address)

    def _require_buffer(self, what: str) -> None:
        # ggml_backend_tensor_set/get abort the process on an unallocated tensor.
        if not self.ptr.contents.buffer:
            raise GGMLError(
                f"{what}: tensor {self.name!r} has no backend buffer; allocate it "
                f"first (Backend.alloc_ctx_tensors, GAllocr.alloc_graph or "
                f"Scheduler.alloc_graph)"
            )

    # -- properties ---------------------------------------------------------

    @property
    def ne(self) -> tuple:
        """All four dimensions in ggml order, exactly as ggml stores them.

        Unused trailing dimensions are 1, so ``t.ne[1]`` is always valid, as it
        is in C. See :attr:`shape` for the numpy-order view.
        """
        return tuple(self.ptr.contents.ne[:4])

    @property
    def nb(self) -> tuple:
        """All four byte strides in ggml order, exactly as ggml stores them."""
        return tuple(self.ptr.contents.nb[:4])

    @property
    def n_dims(self) -> int:
        """Number of dimensions ggml reports (``ggml_n_dims``).

        Trailing 1-sized dimensions do not count, so a ``[n, 1]`` tensor
        reports 1.
        """
        return _ggml.ggml_n_dims(self.ptr)

    @property
    def shape(self) -> tuple:
        """Dimensions in numpy order: the first :attr:`n_dims` of :attr:`ne`, reversed.

        This is the shape :meth:`to_numpy` returns.
        """
        ptr = self.ptr
        return tuple(reversed(ptr.contents.ne[:_ggml.ggml_n_dims(ptr)]))

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
    def op(self) -> "Op":
        """The ggml operation that produced this tensor (``Op.NONE`` for leaves)."""
        return Op(self.ptr.contents.op)

    @property
    def flags(self) -> "TensorFlag":
        """The tensor's flag bits (input, output, param, loss)."""
        return TensorFlag(self.ptr.contents.flags)

    @property
    def src(self) -> "tuple[Tensor, ...]":
        """The operands this tensor was computed from (empty for leaves)."""
        srcs = self.ptr.contents.src
        return tuple(self.ctx._adopt(p) for p in srcs if p)

    @property
    def view_src(self) -> "Optional[Tensor]":
        """The tensor whose data this one views, or ``None``."""
        ptr = self.ptr.contents.view_src
        return self.ctx._adopt(ptr) if ptr else None

    @property
    def view_offs(self) -> int:
        """Byte offset of this view into :attr:`view_src`."""
        return self.ptr.contents.view_offs

    @property
    def name(self) -> str:
        return _ggml.ggml_get_name(self.ptr).decode()

    @name.setter
    def name(self, value: str) -> None:
        _ggml.ggml_set_name(self.ptr, value.encode())

    # -- predicates (``ggml_is_*``) -----------------------------------------

    @property
    def is_contiguous(self) -> bool:
        return bool(_ggml.ggml_is_contiguous(self.ptr))

    @property
    def is_view(self) -> bool:
        return bool(_ggml.ggml_is_view(self.ptr))

    @property
    def is_permuted(self) -> bool:
        return bool(_ggml.ggml_is_permuted(self.ptr))

    @property
    def is_transposed(self) -> bool:
        return bool(_ggml.ggml_is_transposed(self.ptr))

    @property
    def is_empty(self) -> bool:
        return bool(_ggml.ggml_is_empty(self.ptr))

    @property
    def is_scalar(self) -> bool:
        return bool(_ggml.ggml_is_scalar(self.ptr))

    @property
    def is_vector(self) -> bool:
        return bool(_ggml.ggml_is_vector(self.ptr))

    @property
    def is_matrix(self) -> bool:
        return bool(_ggml.ggml_is_matrix(self.ptr))

    @property
    def is_3d(self) -> bool:
        return bool(_ggml.ggml_is_3d(self.ptr))

    @property
    def is_quantized(self) -> bool:
        """True when the dtype is a quantized one (``ggml_is_quantized``)."""
        return bool(_ggml.ggml_is_quantized(self.ptr.contents.type))

    # -- graph flags --------------------------------------------------------

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

    def set_param(self) -> "Tensor":
        """Mark this leaf as a trainable parameter (``ggml_set_param``).

        Parameters are what :meth:`Graph.build_backward_expand` computes
        gradients for. Returns ``self``.
        """
        if self.op != Op.NONE:
            raise ValueError("set_param: only leaf tensors can be parameters")
        _ggml.ggml_set_param(self.ptr)
        return self

    def set_loss(self) -> "Tensor":
        """Mark this scalar as the loss to differentiate (``ggml_set_loss``)."""
        if not self.is_scalar:
            raise ValueError(f"set_loss: the loss must be a scalar, got ne={self.ne}")
        _ggml.ggml_set_loss(self.ptr)
        return self

    # -- data transfer ------------------------------------------------------
    #
    # ggml's sync transfer (``ggml_backend_tensor_set/get``) is keyed on the
    # tensor's own buffer and takes no backend handle, so it lives here rather
    # than on Backend. The tensor must already be allocated (e.g. via
    # ``Backend.alloc_ctx_tensors``, ``GAllocr.alloc_graph``, or
    # ``Scheduler.alloc_graph``). This is where numpy enters the wrapper. The
    # methods are named ``write`` / ``read`` because ``set`` is a ggml graph op.

    def write(self, x) -> None:
        """Upload ``x`` (any array-like, incl. a Python list) into this tensor.

        ``x`` is coerced to this tensor's dtype. It must either be flat with
        :attr:`nelements` elements, or have this tensor's :attr:`shape`
        (1-sized axes may be added or dropped freely). Anything else is almost
        always a numpy-order versus ggml-order mix-up and raises
        :class:`ValueError`. Wraps ``ggml_backend_tensor_set``.
        """
        self._require_buffer("write")
        arr = np.ascontiguousarray(x, dtype=dtype_to_numpy(self.dtype))
        shape = self.shape
        if arr.size != self.nelements or (
            arr.ndim > 1
            and [d for d in arr.shape if d != 1] != [d for d in shape if d != 1]
        ):
            raise ValueError(
                f"write: source shape {arr.shape} does not match tensor shape "
                f"{shape} (numpy order; ggml ne={self.ne})"
            )
        ptr = arr.ctypes.data_as(ctypes.c_void_p)
        _ggml.ggml_backend_tensor_set(self.ptr, ptr, 0, self.nbytes)

    def write_raw(self, data, offset: int = 0) -> None:
        """Upload raw bytes into this tensor verbatim (no dtype coercion).

        ``data`` is any bytes-like buffer (``bytes``, ``bytearray``,
        ``memoryview``, a numpy array, ...) whose layout already matches the
        tensor's. Use this for weights read straight from a file -- including
        quantized dtypes that have no numpy equivalent. ``offset`` is the byte
        offset into the tensor. Wraps ``ggml_backend_tensor_set``.
        """
        self._require_buffer("write_raw")
        mv = memoryview(data).cast("B")
        if offset < 0:
            raise ValueError(f"write_raw: offset must be non-negative, got {offset}")
        if offset + mv.nbytes > self.nbytes:
            raise ValueError(
                f"write_raw: {mv.nbytes} bytes at offset {offset} exceed tensor "
                f"size {self.nbytes} (ne={self.ne}, dtype={self.dtype.name})"
            )
        # Borrow the caller's memory when it is a contiguous writable buffer or
        # an immutable ``bytes``; copy only for the remaining read-only cases.
        if isinstance(data, bytes):
            ptr = ctypes.c_char_p(data)
        elif mv.readonly or not mv.c_contiguous:
            ptr = (ctypes.c_char * mv.nbytes).from_buffer_copy(mv)
        else:
            ptr = (ctypes.c_char * mv.nbytes).from_buffer(mv)
        _ggml.ggml_backend_tensor_set(
            self.ptr, ctypes.cast(ptr, ctypes.c_void_p), offset, mv.nbytes
        )

    def read(self, out: "np.ndarray") -> "np.ndarray":
        """Download this tensor into the pre-allocated, contiguous numpy array ``out``.

        ``out`` must be contiguous and exactly the tensor's byte size. Wraps
        ``ggml_backend_tensor_get`` and returns ``out``.
        """
        self._require_buffer("read")
        if not out.flags["C_CONTIGUOUS"]:
            raise ValueError("read: out must be a C-contiguous array")
        if out.nbytes != self.nbytes:
            raise ValueError(
                f"read: out has {out.nbytes} bytes but tensor holds "
                f"{self.nbytes} (ne={self.ne}, dtype={self.dtype.name})"
            )
        ptr = out.ctypes.data_as(ctypes.c_void_p)
        _ggml.ggml_backend_tensor_get(self.ptr, ptr, 0, self.nbytes)
        return out

    def to_numpy(self) -> "np.ndarray":
        """Download this tensor into a freshly allocated numpy array.

        Shape is :attr:`shape` (numpy order) and dtype follows this tensor's.
        Use :meth:`read` instead to reuse a buffer or to control the layout.
        """
        return self.read(np.empty(self.shape, dtype=dtype_to_numpy(self.dtype)))

    def copy_to(self, dst: "Tensor") -> None:
        """Copy this tensor's data into ``dst``, across backends if needed.

        Both tensors must be allocated and have the same layout. Wraps
        ``ggml_backend_tensor_copy``.
        """
        self._require_buffer("copy_to")
        dst._require_buffer("copy_to")
        if self.ne != dst.ne or self.dtype != dst.dtype:
            raise ValueError(
                f"copy_to: layouts differ: ne={self.ne} {self.dtype.name} vs "
                f"ne={dst.ne} {dst.dtype.name}"
            )
        _ggml.ggml_backend_tensor_copy(self.ptr, dst.ptr)

    # -- operators ----------------------------------------------------------
    #
    # Each operator maps to exactly one ggml op; scalars go through ``scale`` /
    # ``scale_bias``. Anything without a single matching op (``2.0 / t``, ``**``)
    # is left unsupported rather than composed here. ``mul_mat`` deliberately
    # has no ``@``: it takes its operands transposed relative to
    # ``numpy.matmul``, and an infix spelling would invite exactly that confusion.

    def __add__(self, other) -> "Tensor":
        """``a + b`` is element-wise add; ``a + 1.0`` is :meth:`scale_bias`."""
        if isinstance(other, Tensor):
            return self.add(other)
        if isinstance(other, _SCALAR):
            return self.scale_bias(1.0, other)
        return NotImplemented

    def __radd__(self, other) -> "Tensor":
        if isinstance(other, _SCALAR):
            return self.scale_bias(1.0, other)
        return NotImplemented

    def __sub__(self, other) -> "Tensor":
        if isinstance(other, Tensor):
            return self.sub(other)
        if isinstance(other, _SCALAR):
            return self.scale_bias(1.0, -other)
        return NotImplemented

    def __rsub__(self, other) -> "Tensor":
        if isinstance(other, _SCALAR):
            return self.scale_bias(-1.0, other)
        return NotImplemented

    def __mul__(self, other) -> "Tensor":
        """``a * b`` is element-wise multiply; ``a * 2.0`` is :meth:`scale`."""
        if isinstance(other, Tensor):
            return self.mul(other)
        if isinstance(other, _SCALAR):
            return self.scale(other)
        return NotImplemented

    def __rmul__(self, other) -> "Tensor":
        if isinstance(other, _SCALAR):
            return self.scale(other)
        return NotImplemented

    def __truediv__(self, other) -> "Tensor":
        if isinstance(other, Tensor):
            return self.div(other)
        if isinstance(other, _SCALAR):
            return self.scale(1.0 / other)
        return NotImplemented

    def __neg__(self) -> "Tensor":
        return self.neg()

    @property
    def T(self) -> "Tensor":
        """Swap the first two ggml dimensions (:meth:`transpose`), as a view."""
        return self.transpose()

    def __array__(self, dtype=None, copy=None) -> "np.ndarray":
        """Let ``np.asarray(t)`` download the tensor via :meth:`to_numpy`."""
        out = self.to_numpy()
        return out if dtype is None else out.astype(dtype, copy=False)

    def __repr__(self) -> str:
        if self.ctx._ptr is None:
            return "<ggml.Tensor closed>"
        return f"<ggml.Tensor name={self.name!r} ne={self.ne} dtype={self.dtype.name}>"


# -- op generation ----------------------------------------------------------------

_INPLACE_DOC = (
    "In-place variant of :meth:`{base}`: writes the result into ``self`` and\n"
    "returns a view of it, so anything else reading ``self`` sees the new values."
)


def _op_specs():
    """Yield ``(name, fn_name, params, doc, check)`` for every op to generate.

    Besides the three tables, this adds an ``<op>_inplace`` entry for each op
    whose ``ggml_<op>_inplace`` counterpart exists in the bindings.
    """
    specs = []
    for name, fn_name in _UNARY_OPS.items():
        doc = f"Apply ``{fn_name}`` to this tensor, returning a new tensor."
        specs.append((name, fn_name, (), doc))
    for name, fn_name in _BINARY_OPS.items():
        doc = f"Apply ``{fn_name}(self, other)``, returning a new tensor."
        specs.append((name, fn_name, (("other", _TENSOR),), doc))
    for name, (fn_name, params, doc) in _PARAM_OPS.items():
        specs.append((name, fn_name, params, doc))

    for name, fn_name, params, doc in specs:
        yield name, fn_name, params, doc, _CHECKS.get(name)
        if hasattr(_ggml, f"{fn_name}_inplace"):
            yield (f"{name}_inplace", f"{fn_name}_inplace", params,
                   _INPLACE_DOC.format(base=name), _CHECKS.get(name))


def _make_op(name: str, fn_name: str, params, doc: str, check, *, on_context: bool):
    """Build the method ``name`` calling ``fn_name``.

    ``params`` describes the arguments after ggml's ``ctx`` and ``a``: each
    entry is ``(arg_name, kind)`` or ``(arg_name, kind, default)``, where
    ``kind`` is one of the ``_TENSOR``/``_INT``/... constants above.

    With ``on_context`` the method goes on :class:`Context` and takes ``a``
    explicitly (``ctx.add(a, b)``, ggml's own argument order); otherwise it goes
    on :class:`Tensor`, ``a`` is the receiver and the target context is
    ``self.ctx``. Either way the result lands in the target context, and the
    target records every other context its operands live in so those stay alive
    with it.

    The method body is generated as source so each op ends up with a real
    signature (keyword arguments, defaults, ``help()``) and no per-call argument
    binding overhead -- graph building calls these once per node.
    """
    if on_context:
        signature = ["self", "a: Tensor"]
        target, a = "self", "a"
        tensors = ["a"]
    else:
        signature = ["self"]
        target, a = "self.ctx", "self"
        tensors = []
    call_args = ["ctx", f"{a}.ptr"]
    arg_names = [a]
    optional = []
    namespace = {"_fn": getattr(_ggml, fn_name), "Tensor": Tensor, "GGMLError": GGMLError,
                 "DType": DType, "PoolOp": PoolOp, "RopeType": RopeType,
                 "SortOrder": SortOrder, "ScaleMode": ScaleMode,
                 "Optional": Optional,
                 "_used_mem": _ggml.ggml_used_mem,
                 "_need": _OP_TENSORS.get(name, 1) * _ggml.ggml_tensor_overhead()}

    for param in params:
        arg_name, kind = param[0], param[1]
        annotation = _ARG_ANNOTATION[kind]
        if len(param) > 2:
            default_name = f"_default_{arg_name}"
            namespace[default_name] = param[2]
            signature.append(f"{arg_name}: {annotation} = {default_name}")
        else:
            signature.append(f"{arg_name}: {annotation}")
        call_args.append(_ARG_CONV[kind].format(arg_name))
        arg_names.append(arg_name)
        if kind == _TENSOR:
            tensors.append(arg_name)
        elif kind == _TENSOR_OPT:
            optional.append(arg_name)

    body = [
        f"target = {target}",
        "ctx = target._ptr",
        "if ctx is None:",
        f"    raise GGMLError('{name}: the target context is closed')",
    ]
    for t in tensors:
        body += [f"if {t}.ctx is not target:", f"    target._deps.add({t}.ctx)"]
    for t in optional:
        body += [f"if {t} is not None and {t}.ctx is not target:",
                 f"    target._deps.add({t}.ctx)"]
    if check is not None:
        namespace["_check"] = check
        body.append(f"_check({', '.join(arg_names)})")
    # The arena check covers tensor metadata only. Ops in a ``no_alloc=False``
    # context also need room for their data, which is not known before the op
    # runs, so there it is a lower bound.
    body += [
        "if _used_mem(ctx) + _need > target.mem_size:",
        f"    target._reserve(_need, '{name}')",
        f"return Tensor(target, _fn({', '.join(call_args)}))",
    ]
    source = f"def {name}({', '.join(signature)}) -> Tensor:\n" + "".join(
        f"    {line}\n" for line in body
    )
    owner = "Context" if on_context else "Tensor"
    exec(compile(source, f"<ggbond op {owner}.{name}>", "exec"), namespace)

    method = namespace[name]
    method.__qualname__ = f"{owner}.{name}"
    if on_context:
        doc = (
            doc.replace("(self, other)", "(a, other)")
            .replace("``self``", "``a``")
            .replace("to this tensor", "to ``a``")
            .replace("this tensor", "``a``")
            + "\n\nThe result is created in this context, whichever contexts the"
            "\noperands live in."
        )
    method.__doc__ = f"{doc}\n\nWraps ``{fn_name}``."
    return method


def install_ops(cls, *, on_context: bool) -> None:
    """Install every generated op on ``cls`` (:class:`Tensor` or :class:`Context`)."""
    for name, fn_name, params, doc, check in _op_specs():
        if name in cls.__dict__:
            raise RuntimeError(f"op {name!r} would shadow {cls.__name__}.{name}")
        setattr(cls, name, _make_op(name, fn_name, params, doc, check, on_context=on_context))


install_ops(Tensor, on_context=False)

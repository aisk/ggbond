"""Thin OO wrapper around ``ggml_tensor``.

Each :class:`Tensor` wraps ``(Context, ggml_tensor_p)``. Operation methods map
1:1 to ggml functions and return a new :class:`Tensor` bound to the same
context. Building a graph and computing it remain explicit -- nothing here is
lazy.

Operation methods are generated from the three tables below rather than written
out one by one: :data:`_UNARY_OPS` for ops taking only the receiver,
:data:`_BINARY_OPS` for ops taking one more tensor, and :data:`_PARAM_OPS` for
everything with extra arguments (scalars, optional tensors, dtypes). See
``_make_op`` for the argument vocabulary.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Optional

import numpy as np

import ggml as _ggml

from .types import DType, F32, dtype_to_numpy

if TYPE_CHECKING:
    from .context import Context

# Pooling ops for :meth:`Tensor.pool_1d` / :meth:`Tensor.pool_2d`
# (mirror ggml's ``enum ggml_op_pool``).
POOL_MAX = _ggml.GGML_OP_POOL_MAX
POOL_AVG = _ggml.GGML_OP_POOL_AVG

# Rope modes for :meth:`Tensor.rope` / :meth:`Tensor.rope_ext`. Mode 0 is the
# original interleaved GPT-NeoX-free layout; the rest mirror ggml's
# ``GGML_ROPE_TYPE_*``.
ROPE_TYPE_NORMAL = 0
ROPE_TYPE_NEOX = _ggml.GGML_ROPE_TYPE_NEOX
ROPE_TYPE_MROPE = _ggml.GGML_ROPE_TYPE_MROPE
ROPE_TYPE_VISION = _ggml.GGML_ROPE_TYPE_VISION

# Argument kinds understood by the op tables.
_TENSOR = "tensor"  # required Tensor operand
_TENSOR_OPT = "tensor?"  # optional Tensor operand; None becomes a NULL pointer
_INT = "int"
_FLOAT = "float"
_BOOL = "bool"
_DTYPE = "dtype"

# How each kind is converted to the ctypes argument, and how it is annotated.
_ARG_CONV = {
    _TENSOR: "{0}.ptr",
    _TENSOR_OPT: "({0}.ptr if {0} is not None else None)",
    _INT: "int({0})",
    _FLOAT: "float({0})",
    _BOOL: "bool({0})",
    _DTYPE: "int({0})",
}

_ARG_ANNOTATION = {
    _TENSOR: "Tensor",
    _TENSOR_OPT: "Optional[Tensor]",
    _INT: "int",
    _FLOAT: "float",
    _BOOL: "bool",
    _DTYPE: "DType",
}

# Ops taking only the receiver: name -> ggml function name.
_UNARY_OPS = {
    "relu": "ggml_relu",
    "gelu": "ggml_gelu",
    "gelu_quick": "ggml_gelu_quick",
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
    "sum": "ggml_sum",
    "sum_rows": "ggml_sum_rows",
    "mean": "ggml_mean",
    "argmax": "ggml_argmax",
    "dup": "ggml_dup",
    "cont": "ggml_cont",
    "transpose": "ggml_transpose",
}

# Ops taking the receiver plus one more tensor: name -> ggml function name.
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
}

# Ops with extra arguments: name -> (ggml function, params, doc). Each param is
# ``(name, kind)`` or ``(name, kind, default)``; ``self`` is always passed as
# the first tensor, matching ggml's ``a`` argument.
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
    "soft_max_ext": (
        "ggml_soft_max_ext",
        (("mask", _TENSOR_OPT), ("scale", _FLOAT, 1.0), ("max_bias", _FLOAT, 0.0)),
        "Fused scaled, masked softmax: ``softmax(self * scale + mask)``.\n\n"
        "``mask`` may be None. ``max_bias`` > 0 enables ALiBi slopes.",
    ),
    "rope": (
        "ggml_rope",
        (("positions", _TENSOR), ("n_dims", _INT), ("mode", _INT)),
        "Rotary position embedding over the first ``n_dims`` dimensions.\n\n"
        "``positions`` is an I32 tensor of token positions, one per sequence\n"
        "element. ``mode`` is a ``ROPE_TYPE_*`` constant.",
    ),
    "rope_ext": (
        "ggml_rope_ext",
        (
            ("positions", _TENSOR),
            ("freq_factors", _TENSOR_OPT),
            ("n_dims", _INT),
            ("mode", _INT),
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
        (("op", _INT), ("k0", _INT), ("s0", _INT), ("p0", _INT)),
        "1D pooling: kernel ``k0``, stride ``s0``, padding ``p0``.\n\n"
        "``op`` is a ggml pooling op (:data:`POOL_MAX` / :data:`POOL_AVG`).",
    ),
    "pool_2d": (
        "ggml_pool_2d",
        (
            ("op", _INT),
            ("k0", _INT),
            ("k1", _INT),
            ("s0", _INT),
            ("s1", _INT),
            ("p0", _FLOAT),
            ("p1", _FLOAT),
        ),
        "2D pooling. ``op`` is :data:`POOL_MAX` or :data:`POOL_AVG`.",
    ),
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

    # -- properties ---------------------------------------------------------

    @property
    def ne(self) -> tuple:
        """Dimensions in ggml order ``(ne0, ne1, ...)``, trailing 1s trimmed.

        The length follows ``ggml_n_dims``, so a ``[n, 1]`` tensor reports a
        single dimension. Use :attr:`ne4` for the untrimmed tuple ggml actually
        stores.
        """
        n = _ggml.ggml_n_dims(self.ptr)
        return tuple(self.ptr.contents.ne[:n])

    @property
    def nb(self) -> tuple:
        """Byte strides in ggml order, trimmed like :attr:`ne`.

        Trimming drops the stride of any trailing 1-sized dimension, which is
        exactly the stride you need to build a view over such a tensor. Use
        :attr:`nb4` to get all four strides.
        """
        n = _ggml.ggml_n_dims(self.ptr)
        return tuple(self.ptr.contents.nb[:n])

    @property
    def ne4(self) -> tuple:
        """All four dimensions in ggml order, exactly as ggml stores them."""
        return tuple(self.ptr.contents.ne[:4])

    @property
    def nb4(self) -> tuple:
        """All four byte strides in ggml order, exactly as ggml stores them."""
        return tuple(self.ptr.contents.nb[:4])

    @property
    def shape(self) -> tuple:
        """Dimensions in numpy order -- :attr:`ne` reversed.

        The shape to allocate a download buffer for :meth:`get` with.
        """
        return tuple(reversed(self.ne))

    @property
    def n_dims(self) -> int:
        """Number of dimensions ggml reports (``ggml_n_dims``)."""
        return _ggml.ggml_n_dims(self.ptr)

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
        if offset < 0:
            raise ValueError(f"set_raw: offset must be non-negative, got {offset}")
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


def _make_op(name: str, fn_name: str, params, doc: str):
    """Build the :class:`Tensor` method ``name`` calling ``fn_name``.

    ``params`` describes the arguments after ggml's ``ctx`` and ``a`` (which are
    always the receiver's context and pointer): each entry is
    ``(arg_name, kind)`` or ``(arg_name, kind, default)``, where ``kind`` is one
    of the ``_TENSOR``/``_INT``/... constants above.

    The method body is generated as source so each op ends up with a real
    signature (keyword arguments, defaults, ``help()``) and no per-call argument
    binding overhead -- graph building calls these once per node.
    """
    signature = ["self"]
    call_args = ["self.ctx.ptr", "self.ptr"]
    namespace = {"_fn": getattr(_ggml, fn_name), "Tensor": Tensor,
                 "DType": DType, "Optional": Optional}

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

    source = (
        f"def {name}({', '.join(signature)}) -> Tensor:\n"
        f"    return Tensor(self.ctx, _fn({', '.join(call_args)}))\n"
    )
    exec(compile(source, f"<ggbond op {name}>", "exec"), namespace)

    method = namespace[name]
    method.__qualname__ = f"Tensor.{name}"
    method.__doc__ = f"{doc}\n\nWraps ``{fn_name}``."
    return method


def _install_ops() -> None:
    for name, fn_name in _UNARY_OPS.items():
        doc = f"Apply ``{fn_name}`` to this tensor, returning a new tensor."
        setattr(Tensor, name, _make_op(name, fn_name, (), doc))

    for name, fn_name in _BINARY_OPS.items():
        doc = f"Apply ``{fn_name}(self, other)``, returning a new tensor."
        setattr(Tensor, name, _make_op(name, fn_name, (("other", _TENSOR),), doc))

    for name, (fn_name, params, doc) in _PARAM_OPS.items():
        setattr(Tensor, name, _make_op(name, fn_name, params, doc))


_install_ops()

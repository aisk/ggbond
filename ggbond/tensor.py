"""Lazy-evaluated high-level Tensor built on top of Session."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from ggbond import ggml

if TYPE_CHECKING:
    from ggbond.session import Session

# dtype mapping: ggml.Type → numpy dtype
_GGML_TO_NP = {
    ggml.Type.F32: np.float32,
    ggml.Type.F16: np.float16,
    ggml.Type.I8: np.int8,
    ggml.Type.I16: np.int16,
    ggml.Type.I32: np.int32,
    ggml.Type.I64: np.int64,
    ggml.Type.F64: np.float64,
}

# Simple unary ops: method_name → ggml function name
_UNARY_OPS = (
    "relu", "gelu", "silu", "sigmoid", "tanh",
    "sqrt", "sqr", "log", "exp", "sin", "cos",
    "neg", "abs", "cont",
    "gelu_quick", "hardswish", "hardsigmoid", "elu",
    "swiglu", "reglu", "geglu",
)

# Simple binary ops: op_name → ggml function name
_BINARY_OPS = ("add", "sub", "mul", "div")

# Simple reduction ops
_REDUCTION_OPS = ("sum", "sum_rows", "mean", "argmax")


def _materialize_op(op: str, ctx_raw, ggml_inputs: list, kwargs: dict):
    """Dispatch *op* to the corresponding ``ggml.*`` call."""
    if op in _BINARY_OPS:
        return getattr(ggml, op)(ctx_raw, ggml_inputs[0], ggml_inputs[1])
    if op == "mul_mat":
        return ggml.mul_mat(ctx_raw, ggml_inputs[0], ggml_inputs[1])
    if op in _UNARY_OPS:
        return getattr(ggml, op)(ctx_raw, ggml_inputs[0])
    if op in _REDUCTION_OPS:
        return getattr(ggml, op)(ctx_raw, ggml_inputs[0])
    if op == "soft_max":
        return ggml.soft_max(ctx_raw, ggml_inputs[0])
    if op == "transpose":
        return ggml.transpose(ctx_raw, ggml_inputs[0])
    if op == "scale":
        return ggml.scale(ctx_raw, ggml_inputs[0], kwargs["s"])
    if op == "scale_add":
        return ggml.scale_bias(ctx_raw, ggml_inputs[0], kwargs["s"], kwargs["b"])
    if op == "clamp":
        return ggml.clamp(ctx_raw, ggml_inputs[0], kwargs["min"], kwargs["max"])
    if op == "norm":
        return ggml.norm(ctx_raw, ggml_inputs[0], kwargs["eps"])
    if op == "rms_norm":
        return ggml.rms_norm(ctx_raw, ggml_inputs[0], kwargs["eps"])
    if op == "l2_norm":
        return ggml.l2_norm(ctx_raw, ggml_inputs[0], kwargs["eps"])
    if op == "leaky_relu":
        return ggml.leaky_relu(ctx_raw, ggml_inputs[0], kwargs["negative_slope"], False)
    if op.startswith("reshape_"):
        dim = int(op[-2])
        shape = kwargs["shape"]
        if dim == 1:
            return ggml.reshape_1d(ctx_raw, ggml_inputs[0], shape[0])
        if dim == 2:
            return ggml.reshape_2d(ctx_raw, ggml_inputs[0], shape[0], shape[1])
        if dim == 3:
            return ggml.reshape_3d(ctx_raw, ggml_inputs[0], shape[0], shape[1], shape[2])
        return ggml.reshape_4d(ctx_raw, ggml_inputs[0], shape[0], shape[1], shape[2], shape[3])
    raise ValueError(f"unknown op: {op!r}")


def _topo_sort(root: Tensor) -> tuple[list[Tensor], list[Tensor]]:
    """BFS topological sort. Returns (ordered, leaves)."""
    visited: set[int] = set()
    ordered: list[Tensor] = []
    leaves: list[Tensor] = []

    # BFS to discover all nodes, then reverse for topo order
    queue = deque([root])
    visited.add(id(root))
    while queue:
        node = queue.popleft()
        ordered.append(node)
        for inp in node._inputs:
            if id(inp) not in visited:
                visited.add(id(inp))
                queue.append(inp)

    ordered.reverse()  # leaves first, root last

    for node in ordered:
        if node._op is None:
            leaves.append(node)

    return ordered, leaves


def _infer_shape(op: str, inputs: tuple[Tensor, ...], kwargs: dict) -> tuple[int, ...]:
    """Infer the output shape for an op given input shapes."""
    if op == "mul_mat":
        # ggml mul_mat(first, second): ne0 = first.ne1, ne1 = second.ne1
        first, second = inputs[0]._shape, inputs[1]._shape
        return (first[1], second[1]) + second[2:]
    if op == "transpose":
        s = inputs[0]._shape
        return (s[1], s[0]) + s[2:] if len(s) >= 2 else s
    if op == "sum":
        return (1,)
    if op in ("sum_rows", "mean", "argmax"):
        return (1,) + inputs[0]._shape[1:]
    if op.startswith("reshape_"):
        return kwargs["shape"]
    return inputs[0]._shape


class Tensor:
    """Lazy tensor that records operations and materializes on ``.compute()``."""

    __slots__ = (
        "_session", "_data", "_op", "_inputs", "_kwargs",
        "_dtype", "_shape", "_name", "_cached",
    )

    def __init__(
        self,
        session: Session,
        *,
        data: np.ndarray | None = None,
        shape: tuple[int, ...] | None = None,
        dtype=None,
        name: str | None = None,
    ):
        self._session = session
        self._op: str | None = None
        self._inputs: tuple[Tensor, ...] = ()
        self._kwargs: dict = {}
        self._name = name
        self._cached: np.ndarray | None = None

        if data is not None:
            data = np.asarray(data, dtype=np.float32 if dtype is None else dtype)
            self._data = data
            self._dtype = ggml.Type.F32  # TODO: map from numpy dtype
            # GGML shape is reversed numpy shape
            self._shape = tuple(reversed(data.shape))
        elif shape is not None:
            self._data = None
            self._dtype = dtype if dtype is not None else ggml.Type.F32
            self._shape = shape
        else:
            raise ValueError("must provide either data= or shape=")

    @classmethod
    def _from_op(
        cls,
        session: Session,
        op: str,
        inputs: tuple[Tensor, ...],
        dtype=None,
        kwargs: dict | None = None,
    ) -> Tensor:
        obj = object.__new__(cls)
        obj._session = session
        obj._op = op
        obj._inputs = inputs
        obj._kwargs = kwargs or {}
        obj._data = None
        obj._name = None
        obj._cached = None
        obj._dtype = dtype if dtype is not None else inputs[0]._dtype
        obj._shape = _infer_shape(op, inputs, obj._kwargs)
        return obj

    # -- properties -----------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape in GGML order (ne0, ne1, ...)."""
        return self._shape

    @property
    def numpy_shape(self) -> tuple[int, ...]:
        """Shape in numpy order (rows, cols, ...)."""
        return tuple(reversed(self._shape))

    @property
    def dtype(self):
        return self._dtype

    @property
    def T(self) -> Tensor:
        return self._unary_op("transpose")

    # -- operators ------------------------------------------------------------

    def _check_session(self, other: Tensor) -> None:
        if self._session is not other._session:
            raise ValueError("cannot mix tensors from different sessions")

    def _binary_op(self, other: Tensor | float | int, op: str) -> Tensor:
        if isinstance(other, (int, float)):
            if op == "mul":
                return Tensor._from_op(self._session, "scale", (self,), kwargs={"s": float(other)})
            if op == "div":
                return Tensor._from_op(self._session, "scale", (self,), kwargs={"s": 1.0 / float(other)})
            if op == "add":
                return Tensor._from_op(self._session, "scale_add", (self,), kwargs={"s": 1.0, "b": float(other)})
            if op == "sub":
                return Tensor._from_op(self._session, "scale_add", (self,), kwargs={"s": 1.0, "b": -float(other)})
            raise TypeError(f"unsupported scalar op: {op}")
        self._check_session(other)
        return Tensor._from_op(self._session, op, (self, other))

    def _unary_op(self, op: str, **kwargs) -> Tensor:
        return Tensor._from_op(self._session, op, (self,), kwargs=kwargs)

    def __add__(self, other):
        return self._binary_op(other, "add")

    def __radd__(self, other):
        return self._binary_op(other, "add")

    def __sub__(self, other):
        return self._binary_op(other, "sub")

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Tensor._from_op(
                self._session, "scale_add", (self,),
                kwargs={"s": -1.0, "b": float(other)},
            )
        return NotImplemented

    def __mul__(self, other):
        return self._binary_op(other, "mul")

    def __rmul__(self, other):
        return self._binary_op(other, "mul")

    def __truediv__(self, other):
        return self._binary_op(other, "div")

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented
        self._check_session(other)
        # Swap operands: ggml mul_mat(b, a) so that (a @ b).numpy() == a_np @ b_np.T
        return Tensor._from_op(self._session, "mul_mat", (other, self))

    def __neg__(self):
        return self._unary_op("neg")

    def __abs__(self):
        return self._unary_op("abs")

    # -- method chain ---------------------------------------------------------

    def relu(self) -> Tensor:
        return self._unary_op("relu")

    def gelu(self) -> Tensor:
        return self._unary_op("gelu")

    def silu(self) -> Tensor:
        return self._unary_op("silu")

    def sigmoid(self) -> Tensor:
        return self._unary_op("sigmoid")

    def tanh(self) -> Tensor:
        return self._unary_op("tanh")

    def softmax(self) -> Tensor:
        return self._unary_op("soft_max")

    def sqrt(self) -> Tensor:
        return self._unary_op("sqrt")

    def sqr(self) -> Tensor:
        return self._unary_op("sqr")

    def log(self) -> Tensor:
        return self._unary_op("log")

    def exp(self) -> Tensor:
        return self._unary_op("exp")

    def sin(self) -> Tensor:
        return self._unary_op("sin")

    def cos(self) -> Tensor:
        return self._unary_op("cos")

    def cont(self) -> Tensor:
        return self._unary_op("cont")

    def sum(self) -> Tensor:
        return self._unary_op("sum")

    def sum_rows(self) -> Tensor:
        return self._unary_op("sum_rows")

    def mean(self) -> Tensor:
        return self._unary_op("mean")

    def argmax(self) -> Tensor:
        return self._unary_op("argmax")

    def norm(self, eps: float = 1e-5) -> Tensor:
        return self._unary_op("norm", eps=eps)

    def rms_norm(self, eps: float = 1e-5) -> Tensor:
        return self._unary_op("rms_norm", eps=eps)

    def scale(self, s: float) -> Tensor:
        return self._unary_op("scale", s=s)

    def clamp(self, min: float, max: float) -> Tensor:
        return self._unary_op("clamp", min=min, max=max)

    def leaky_relu(self, negative_slope: float = 0.01) -> Tensor:
        return self._unary_op("leaky_relu", negative_slope=negative_slope)

    def transpose(self) -> Tensor:
        return self._unary_op("transpose")

    def reshape(self, *shape: int) -> Tensor:
        ndim = len(shape)
        if ndim < 1 or ndim > 4:
            raise ValueError(f"reshape supports 1-4 dims, got {ndim}")
        return Tensor._from_op(
            self._session, f"reshape_{ndim}d", (self,),
            kwargs={"shape": shape},
        )

    # -- compute --------------------------------------------------------------

    def compute(self) -> Tensor:
        """Materialize the computation graph and execute it. Returns self."""
        ordered, leaves = _topo_sort(self)

        if not ordered:
            return self

        # If this is a bare leaf with data, just cache it directly
        if self._op is None and self._data is not None:
            self._cached = self._data.flatten().astype(np.float32)
            return self

        # Count total tensors needed: leaves + ops + graph overhead
        n_ops = sum(1 for n in ordered if n._op is not None)
        # Each op may produce intermediate tensors; estimate generously
        n_model_tensors = len(leaves)
        n_graph_tensors = n_ops * 2 + 10  # ops + intermediates + slack

        # Phase 1: Create model context for leaf tensors
        ctx_model = self._session.context(n_tensors=n_model_tensors)

        # Create ggml tensors for leaves and set as input
        ggml_map: dict[int, ggml.Tensor] = {}
        for i, leaf in enumerate(leaves):
            t = ctx_model.new_tensor(leaf._dtype, *leaf._shape)
            ggml.set_input(t)
            name = leaf._name or f"leaf_{i}"
            ggml.set_name(t, name)
            ggml_map[id(leaf)] = t

        # Allocate leaf memory on backend
        self._session.alloc(ctx_model)

        # Set leaf data
        for leaf in leaves:
            if leaf._data is not None:
                self._session.set(ggml_map[id(leaf)], leaf._data)

        # Phase 2: Build computation graph
        g = self._session.graph(max_nodes=n_graph_tensors)

        # Materialize ops in topological order
        for node in ordered:
            if node._op is None:
                continue
            ggml_inputs = [ggml_map[id(inp)] for inp in node._inputs]
            ggml_result = _materialize_op(node._op, g.ctx, ggml_inputs, node._kwargs)
            if node._name:
                ggml.set_name(ggml_result, node._name)
            ggml_map[id(node)] = ggml_result

        # Mark output and build graph
        result_ggml = ggml_map[id(self)]
        ggml.set_output(result_ggml)
        g.build_forward(result_ggml)

        # Phase 3: Reserve, execute, extract
        self._session.reserve(g)
        self._session.run(g)

        out = self._session.get(g, result_ggml)
        self._cached = out
        return self

    def numpy(self) -> np.ndarray:
        """Return result as numpy array. Triggers compute() if needed."""
        if self._cached is None:
            self.compute()
        np_shape = tuple(reversed(self._shape))
        return self._cached.reshape(np_shape)

    def __repr__(self) -> str:
        np_shape = tuple(reversed(self._shape))
        kind = self._op or "leaf"
        computed = self._cached is not None
        return f"Tensor(shape={np_shape}, op={kind}, computed={computed})"

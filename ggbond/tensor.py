"""Lazy-evaluated high-level Tensor built on top of GGML."""

from __future__ import annotations

from collections import deque

import numpy as np

from ggbond import ggml
from ggbond.context import Context
from ggbond.graph import GAllocr, Graph

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
    if op == "pool_1d":
        return ggml.pool_1d(ctx_raw, ggml_inputs[0], kwargs["op_pool"], kwargs["k"], kwargs["s"], kwargs["p"])
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
    if op == "get_rows":
        return ggml.get_rows(ctx_raw, ggml_inputs[0], ggml_inputs[1])
    if op == "permute":
        return ggml.permute(ctx_raw, ggml_inputs[0], kwargs["a0"], kwargs["a1"], kwargs["a2"], kwargs["a3"])
    if op == "diag_mask_inf":
        return ggml.diag_mask_inf(ctx_raw, ggml_inputs[0], kwargs["n_past"])
    if op == "view_1d":
        return ggml.view_1d(ctx_raw, ggml_inputs[0], kwargs["ne0"], kwargs["offset"])
    if op == "view_2d":
        return ggml.view_2d(ctx_raw, ggml_inputs[0], kwargs["ne0"], kwargs["ne1"], kwargs["nb1"], kwargs["offset"])
    if op == "cont_2d":
        return ggml.cont_2d(ctx_raw, ggml_inputs[0], kwargs["ne0"], kwargs["ne1"])
    if op == "cont_3d":
        return ggml.cont_3d(ctx_raw, ggml_inputs[0], kwargs["ne0"], kwargs["ne1"], kwargs["ne2"])
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
    if op == "pool_1d":
        s = inputs[0]._shape
        k, stride, pad = kwargs["k"], kwargs["s"], kwargs["p"]
        ne0 = (s[0] + 2 * pad - k) // stride + 1
        return (ne0,) + s[1:]
    if op.startswith("reshape_"):
        return kwargs["shape"]
    if op == "get_rows":
        a, b = inputs[0]._shape, inputs[1]._shape
        return (a[0],) + b
    if op == "permute":
        s = inputs[0]._shape
        perm = (kwargs["a0"], kwargs["a1"], kwargs["a2"], kwargs["a3"])
        padded = s + (1,) * (4 - len(s))
        return tuple(padded[p] for p in perm)[:len(s)]
    if op == "diag_mask_inf":
        return inputs[0]._shape
    if op == "view_1d":
        return (kwargs["ne0"],)
    if op == "view_2d":
        return (kwargs["ne0"], kwargs["ne1"])
    if op == "cont_2d":
        return (kwargs["ne0"], kwargs["ne1"])
    if op == "cont_3d":
        return (kwargs["ne0"], kwargs["ne1"], kwargs["ne2"])
    return inputs[0]._shape


def _ensure_same_session(inputs: tuple[Tensor, ...]) -> None:
    """Validate all input tensors are bound to the same Session."""
    if not inputs:
        raise ValueError("inputs must not be empty")
    base_session = inputs[0]._session
    for tensor in inputs[1:]:
        if tensor._session is not base_session:
            raise ValueError("all Tensor inputs must belong to the same Session")


class Tensor:
    """Lazy tensor that records operations and materializes on ``.compute()``."""

    __slots__ = (
        "_op", "_inputs", "_kwargs",
        "_dtype", "_shape", "_name", "_cached",
        "_session", "_ggml_tensor",
    )

    @classmethod
    def _from_ggml(cls, session, ggml_tensor, shape, dtype=None, name=None):
        """Create a leaf node backed by a backend-resident ggml tensor."""
        obj = object.__new__(cls)
        obj._session = session
        obj._ggml_tensor = ggml_tensor
        obj._op = None
        obj._inputs = ()
        obj._kwargs = {}
        obj._dtype = dtype or ggml.Type.F32
        obj._shape = shape
        obj._name = name
        obj._cached = None
        return obj

    @classmethod
    def _from_op(
        cls,
        op: str,
        inputs: tuple[Tensor, ...],
        dtype=None,
        kwargs: dict | None = None,
    ) -> Tensor:
        _ensure_same_session(inputs)
        obj = object.__new__(cls)
        obj._session = inputs[0]._session
        obj._ggml_tensor = None
        obj._op = op
        obj._inputs = inputs
        obj._kwargs = kwargs or {}
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

    def _binary_op(self, other: Tensor | float | int, op: str) -> Tensor:
        if isinstance(other, (int, float)):
            if op == "mul":
                return Tensor._from_op("scale", (self,), kwargs={"s": float(other)})
            if op == "div":
                return Tensor._from_op("scale", (self,), kwargs={"s": 1.0 / float(other)})
            if op == "add":
                return Tensor._from_op("scale_add", (self,), kwargs={"s": 1.0, "b": float(other)})
            if op == "sub":
                return Tensor._from_op("scale_add", (self,), kwargs={"s": 1.0, "b": -float(other)})
            raise TypeError(f"unsupported scalar op: {op}")
        return Tensor._from_op(op, (self, other))

    def _unary_op(self, op: str, **kwargs) -> Tensor:
        return Tensor._from_op(op, (self,), kwargs=kwargs)

    def __add__(self, other):
        return self._binary_op(other, "add")

    def __radd__(self, other):
        return self._binary_op(other, "add")

    def __sub__(self, other):
        return self._binary_op(other, "sub")

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Tensor._from_op(
                "scale_add", (self,),
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
        # Swap operands: ggml mul_mat(b, a) so that (a @ b).numpy() == a_np @ b_np.T
        return Tensor._from_op("mul_mat", (other, self))

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
            f"reshape_{ndim}d", (self,),
            kwargs={"shape": shape},
        )

    def get_rows(self, indices: Tensor) -> Tensor:
        return Tensor._from_op("get_rows", (self, indices))

    def permute(self, a0: int, a1: int, a2: int, a3: int) -> Tensor:
        return Tensor._from_op("permute", (self,), kwargs={"a0": a0, "a1": a1, "a2": a2, "a3": a3})

    def diag_mask_inf(self, n_past: int) -> Tensor:
        return Tensor._from_op("diag_mask_inf", (self,), kwargs={"n_past": n_past})

    def view_1d(self, ne0: int, offset: int) -> Tensor:
        return Tensor._from_op("view_1d", (self,), kwargs={"ne0": ne0, "offset": offset})

    def view_2d(self, ne0: int, ne1: int, nb1: int, offset: int) -> Tensor:
        return Tensor._from_op("view_2d", (self,), kwargs={"ne0": ne0, "ne1": ne1, "nb1": nb1, "offset": offset})

    def cont_2d(self, ne0: int, ne1: int) -> Tensor:
        return Tensor._from_op("cont_2d", (self,), kwargs={"ne0": ne0, "ne1": ne1})

    def cont_3d(self, ne0: int, ne1: int, ne2: int) -> Tensor:
        return Tensor._from_op("cont_3d", (self,), kwargs={"ne0": ne0, "ne1": ne1, "ne2": ne2})

    def pool_1d(self, op_pool, k: int, s: int, p: int = 0) -> Tensor:
        return Tensor._from_op(
            "pool_1d", (self,),
            kwargs={"op_pool": op_pool, "k": k, "s": s, "p": p},
        )

    def max_pool_1d(self, k: int, s: int, p: int = 0) -> Tensor:
        return self.pool_1d(ggml.OpPool.MAX, k, s, p)

    # -- compute --------------------------------------------------------------

    def compute(self) -> Tensor:
        """Materialize the computation graph and execute it. Returns self."""
        backend = self._session._backend
        ordered, leaves = _topo_sort(self)

        if self._op is None:
            # Bare leaf node: read directly from backend
            self._cached = backend.tensor_get(self._ggml_tensor)
            return self

        # All leaves are backend-resident, map directly
        ggml_map = {id(leaf): leaf._ggml_tensor for leaf in leaves}

        # Build graph, allocate, compute and extract with exception-safe cleanup.
        n_ops = sum(1 for n in ordered if n._op is not None)
        with Graph(max_nodes=n_ops * 2 + 10) as g:
            for node in ordered:
                if node._op is None:
                    continue
                ggml_inputs = [ggml_map[id(inp)] for inp in node._inputs]
                ggml_result = _materialize_op(node._op, g.ctx, ggml_inputs, node._kwargs)
                if node._name:
                    ggml.set_name(ggml_result, node._name)
                ggml_map[id(node)] = ggml_result

            result_ggml = ggml_map[id(self)]
            ggml.set_output(result_ggml)
            g.build_forward(result_ggml)

            with GAllocr(backend) as allocr:
                allocr.reserve(g.raw)
                allocr.alloc(g.raw)
                backend.compute(g.raw)
                self._cached = backend.tensor_get(result_ggml)
        return self

    def numpy(self) -> np.ndarray:
        """Return result as numpy array. Triggers compute() if needed."""
        if self._cached is None:
            self.compute()
        return self._cached.reshape(tuple(reversed(self._shape)))

    def tolist(self) -> list:
        return self.numpy().tolist()

    def __repr__(self) -> str:
        np_shape = tuple(reversed(self._shape))
        kind = self._op or "leaf"
        computed = self._cached is not None
        return f"Tensor(shape={np_shape}, op={kind}, computed={computed})"

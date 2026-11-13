"""The high-level API: a :class:`Session` that runs graphs over variables.

:mod:`ggbond.ggml` maps ggml one to one, which means a program assembles
backends, contexts, buffers, a scheduler and graphs by hand and manages their
lifetimes. This module is the layer for everyone else. Its nouns describe what
things are *for*, not how ggml implements them:

* :class:`Session` -- where things run. Picks the devices and the thread
  count, and owns everything created through it.
* **variables** -- tensors that persist: weights, caches, any state. Created
  by :meth:`Session.variables`, :meth:`Session.load_arrays` or
  :meth:`Session.load_gguf`; they live until the session closes.
* **a graph** -- one computation, written inside ``with session.graph() as g:``.
  Inputs come from :meth:`GraphScope.input`, :meth:`GraphScope.run` computes
  and returns numpy arrays. Everything built inside the block ends with it.

::

    import numpy as np
    from ggbond import Session

    a = np.random.rand(4, 2).astype(np.float32)
    b = np.random.rand(3, 2).astype(np.float32)

    with Session() as sess:
        w = sess.load_arrays({"a": a})          # a variable, kept across graphs

        with sess.graph() as g:
            x = g.input(b)                      # an input of this graph
            y = w["a"].mul_mat(x)               # ggml_mul_mat(a, x): x @ a.T
            out = g.run(y)                      # numpy array

Operations keep ggml's names, argument order and shape order (dimensions are
``(ne0, ne1, ...)``, the reverse of numpy's): only the nouns are new. ggml
graphs have fixed shapes, so a model whose shapes change between evaluations
(a language model decoding token by token) opens a new graph per evaluation,
exactly as ggml's own C examples rebuild theirs; a fixed-shape model opens one
graph and calls :meth:`GraphScope.run` in a loop.

The tensors handed out are :class:`ggbond.ggml.Tensor` objects, and
:attr:`Session.backends`, :attr:`Session.sched`, :attr:`GraphScope.ctx` and
:attr:`GraphScope.cgraph` expose the low-level objects underneath.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import mmap
import os
import sys
import weakref
from typing import Dict, Mapping, Optional, Tuple, Union

import numpy as np

import ggml as _ggml

from .ggml import (
    GGUF,
    Backend,
    Context,
    Device,
    DeviceType,
    DType,
    GGMLError,
    Scheduler,
    Tensor,
    dtype_from_numpy,
    dtype_to_numpy,
    tensor_overhead,
)
from .ggml.graph import Graph as _CGraph
from .ggml.ops import ContextOps

__all__ = ["Session", "GraphScope", "Variable", "default_n_threads"]


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------


def _sysctl_int(name: str) -> int:
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    value = ctypes.c_int(0)
    size = ctypes.c_size_t(ctypes.sizeof(value))
    if libc.sysctlbyname(name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0):
        return 0
    return value.value


def default_n_threads() -> int:
    """A good CPU thread count for this machine: its physical cores.

    Compute-bound ggml kernels gain nothing from hyper-threads and lose speed
    on efficiency cores, so this counts the performance cores on Apple silicon
    and the physical cores this process may run on elsewhere, falling back to
    the logical CPU count. (ggml's own default is a fixed 4.)
    """
    try:
        if sys.platform == "darwin":
            n = _sysctl_int("hw.perflevel0.physicalcpu") or _sysctl_int("hw.physicalcpu")
            if n > 0:
                return n
        elif sys.platform.startswith("linux"):
            cores = set()
            for cpu in os.sched_getaffinity(0):
                topology = f"/sys/devices/system/cpu/cpu{cpu}/topology/"
                with open(topology + "physical_package_id") as package, \
                        open(topology + "core_id") as core:
                    cores.add((package.read().strip(), core.read().strip()))
            if cores:
                return len(cores)
    except Exception:
        pass
    return os.cpu_count() or 4


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------


def _as_array(data, dtype: Optional[DType] = None) -> "Tuple[np.ndarray, DType]":
    """Return ``data`` as an array plus the ggml type a tensor holding it gets.

    Without an explicit ``dtype`` numpy's defaults are narrowed to the types
    ggml computes with: float64 becomes F32 and int64 becomes I32 (ggml has
    next to no F64 / I64 operations, and using one aborts the process).
    """
    if dtype is not None:
        return np.asarray(data, dtype=dtype_to_numpy(dtype)), dtype
    array = np.asarray(data)
    if array.dtype == np.float64:
        array = array.astype(np.float32)
    elif array.dtype == np.int64:
        if array.size and (array.min() < -(1 << 31) or array.max() >= (1 << 31)):
            raise ValueError("int64 data does not fit I32; pass dtype=I64 to keep it")
        array = array.astype(np.int32)
    return array, dtype_from_numpy(array.dtype)


class Variable(Tensor):
    """A tensor that persists across graphs: a weight, a cache, any state.

    It is a :class:`ggbond.ggml.Tensor` whose data lives on one of the
    session's devices until the session closes; fill it with :meth:`write` /
    :meth:`write_raw` and read it with :meth:`to_numpy` at any time.

    Used in an expression, a variable puts the result into the graph that is
    currently open, so ``w.mul_mat(x)`` and ``w + x`` work inside
    ``with session.graph():`` whichever operand comes first. In-place
    operations (``v.add_inplace(x)``, ``value.cpy(v.view_1d(...))``) write into
    the variable's own memory when the graph runs, which is how state such as a
    KV cache is carried from one graph to the next.
    """

    __slots__ = ("_session",)

    def __init__(self, ctx: Context, ptr, session: "Session"):
        super().__init__(ctx, ptr)
        self._session = session

    def __repr__(self) -> str:
        try:
            return f"<ggbond.Variable {self.name!r} ne={self.ne[:self.n_dims]} {self.dtype.name}>"
        except GGMLError:
            return "<ggbond.Variable closed>"


def _route(name: str, template):
    def op(self, *args, **kwargs):
        return getattr(self._session._graph_ctx(name), name)(self, *args, **kwargs)

    op.__name__ = name
    op.__qualname__ = f"Variable.{name}"
    op.__doc__ = getattr(template, "__doc__", None)
    return op


# Every ggml op: the ``Tensor`` form ``a.op(b)`` is ``a.ctx.op(a, b)``, which
# for a variable would be its own (full) context. Send it to the open graph.
_OP_NAMES = frozenset(
    name
    for name, value in vars(ContextOps).items()
    if not name.startswith("_") and callable(value)
)
for _name in _OP_NAMES:
    setattr(Variable, _name, _route(_name, getattr(Tensor, _name, None)))
del _name


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

Spec = Mapping[str, tuple]


class Session:
    """Where graphs run, and the owner of everything created through it.

    ``Session()`` uses the best available device, with the CPU behind it for
    the operations that device cannot run. ``Session(device="cpu")`` (or any
    name listed by :meth:`devices`, or a device type such as ``"gpu"``) picks
    one explicitly. Already opened :class:`ggbond.ggml.Backend` objects can be
    passed instead, in priority order; those stay yours to close.

    ``n_threads`` is the CPU thread count, by default the machine's physical
    cores (:func:`default_n_threads`). ``graph_size`` bounds how many tensors
    one graph may hold (operations plus the variables and inputs they read);
    raise it for large models.

    Closing the session frees everything: variables, and the graph that is
    open if any. Use it as a context manager.
    """

    def __init__(
        self,
        *backends: Backend,
        device: Optional[str] = None,
        n_threads: Optional[int] = None,
        graph_size: Optional[int] = None,
    ):
        if graph_size is None:
            graph_size = _ggml.GGML_DEFAULT_GRAPH_SIZE
        elif graph_size <= 0:
            raise ValueError(f"graph_size must be positive, got {graph_size}")
        if backends and device is not None:
            raise ValueError("pass either backends or device=, not both")
        self.graph_size = graph_size
        self.n_threads = default_n_threads() if n_threads is None else n_threads
        self._closed = False
        self._owned: list = []
        self._backends: Tuple[Backend, ...] = ()
        self._sched: Optional[Scheduler] = None
        self._ctx: Optional[Context] = None
        self._scope: Optional["GraphScope"] = None
        self._groups: list = []  # contexts holding variables
        self._ggufs: list = []
        self._counts = weakref.WeakKeyDictionary()  # context -> (stamp, tensors)
        try:
            if backends:
                for backend in backends:
                    backend.ptr  # raises GGMLError on a closed backend
                backends = list(backends)
            else:
                Backend.load_all()
                first = Backend.init_best() if device is None else self._open(device)
                self._owned.append(first)
                backends = [first]
            # The scheduler needs a CPU backend last, for whatever the others cannot run.
            last = backends[-1].device
            if last is None or last.type != DeviceType.CPU:
                cpu = Backend.init_by_type(DeviceType.CPU)
                self._owned.append(cpu)
                backends.append(cpu)
            for backend in backends:
                if backend.is_cpu:
                    backend.set_n_threads(self.n_threads)
            self._backends = tuple(backends)
            self._sched = Scheduler(*backends, graph_size=graph_size)
            # Room for the graph and its nodes, plus as many inputs and views again.
            self._ctx = Context.for_tensors(graph_size, graph_size=graph_size)
        except BaseException:
            self.close()
            raise

    # -- devices ------------------------------------------------------------

    @staticmethod
    def devices() -> "list[str]":
        """Names of the devices this installation can run on, e.g. ``["CPU"]``."""
        Backend.load_all()
        return [device.name for device in Device.all()]

    @staticmethod
    def _open(device: str) -> Backend:
        available = Device.all()
        for candidate in available:
            if candidate.name.lower() == device.lower():
                return candidate.init_backend()
        try:
            kind = DeviceType[device.upper()]
        except KeyError:
            kind = None
        if kind is not None:
            for candidate in available:
                if candidate.type == kind:
                    return candidate.init_backend()
        names = ", ".join(repr(d.name) for d in available)
        kinds = ", ".join(repr(k.name.lower()) for k in sorted({d.type for d in available}))
        raise ValueError(
            f"no device {device!r}; available devices: {names}; device types: {kinds}"
        )

    def _backend_for(self, device: Optional[str]) -> Backend:
        if device is None:
            return self.backends[0]
        for backend in self.backends:
            dev = backend.device
            if dev is not None and device.lower() in (dev.name.lower(), dev.type.name.lower()):
                return backend
        names = ", ".join(repr(b.device.name) for b in self._backends if b.device is not None)
        raise ValueError(f"the session does not run on a device {device!r}; it uses {names}")

    # -- the low-level objects underneath -----------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise GGMLError("Session is closed")

    @property
    def backends(self) -> "Tuple[Backend, ...]":
        """The :class:`ggbond.ggml.Backend` objects in use, main device first, CPU last."""
        self._check_open()
        return self._backends

    @property
    def sched(self) -> Scheduler:
        """The :class:`ggbond.ggml.Scheduler` that allocates and computes the graphs."""
        self._check_open()
        return self._sched

    @property
    def closed(self) -> bool:
        return self._closed

    # -- variables ----------------------------------------------------------

    def variables(self, spec: Spec, *, device: Optional[str] = None) -> Dict[str, Variable]:
        """Create a group of variables, zero-filled, and return them by name.

        ``spec`` maps each name to ``(dtype, ne0, ne1, ...)`` with the
        dimensions in ggml order. The group is allocated together on the
        session's main device, or on ``device`` (one of the session's devices,
        e.g. ``"cpu"``). Fill the variables with :meth:`Tensor.write` /
        :meth:`Tensor.write_raw`.
        """
        backend = self._backend_for(device)
        if not spec:
            return {}
        ctx = Context.for_tensors(len(spec))
        try:
            tensors = {}
            for name, entry in spec.items():
                try:
                    dtype, *ne = entry
                except TypeError:
                    raise ValueError(
                        f"variable {name!r}: expected (dtype, ne0, ne1, ...), got {entry!r}"
                    ) from None
                if not ne or any(isinstance(n, int) and n < 1 for n in ne):
                    raise ValueError(
                        f"variable {name!r}: dimensions must be at least 1, got ne={tuple(ne)}"
                    )
                tensors[name] = ctx.new_tensor(dtype, *ne, name=str(name))
            backend.alloc_ctx_tensors(ctx).clear(0)
        except BaseException:
            ctx.close()
            raise
        return self._adopt_group(ctx, tensors)

    def variable(self, dtype: DType, *ne: int, name: Optional[str] = None,
                 device: Optional[str] = None) -> Variable:
        """Create one zero-filled variable. Prefer :meth:`variables` for many."""
        key = name if name is not None else ""
        return self.variables({key: (dtype, *ne)}, device=device)[key]

    def load_arrays(self, arrays: Mapping[str, object], *,
                    device: Optional[str] = None) -> Dict[str, Variable]:
        """Create variables holding ``arrays`` (name -> array-like).

        Each variable takes its type from the array and its ggml dimensions
        from the reversed numpy shape. float64 and int64 data, numpy's
        defaults, are stored as F32 and I32.
        """
        arrays = {name: _as_array(value)[0] for name, value in arrays.items()}
        spec = {
            name: (dtype_from_numpy(array.dtype), *(reversed(array.shape) if array.ndim else (1,)))
            for name, array in arrays.items()
        }
        variables = self.variables(spec, device=device)
        for name, array in arrays.items():
            variables[name].write(array)
        return variables

    def load_gguf(self, path, *, device: Optional[str] = None) -> "Tuple[Dict[str, Variable], GGUF]":
        """Load every tensor of a GGUF file as a variable.

        Returns ``(variables, gguf)``: the variables by tensor name, and the
        parsed :class:`ggbond.ggml.GGUF` for the file's metadata
        (``gguf["general.architecture"]``, ``gguf.keys()``, ...). Both belong
        to the session.
        """
        backend = self._backend_for(device)
        gguf = GGUF.from_file(path, no_alloc=True)
        ctx = gguf.context
        try:
            tensors = {}
            if gguf.n_tensors:
                backend.alloc_ctx_tensors(ctx)
                with open(path, "rb") as f, \
                        mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                    view = memoryview(mapped)
                    try:
                        for i in range(gguf.n_tensors):
                            name = gguf.get_tensor_name(i)
                            tensor = ctx.get_tensor(name)
                            if tensor is None:
                                raise ValueError(f"{path}: no metadata for tensor {name!r}")
                            start = gguf.data_offset + gguf.get_tensor_offset(i)
                            end = start + tensor.nbytes
                            if end > len(view):
                                raise ValueError(
                                    f"{path}: truncated file, tensor {name!r} needs bytes "
                                    f"{start}..{end} of {len(view)}"
                                )
                            tensor.write_raw(view[start:end])
                            tensors[name] = tensor
                    finally:
                        view.release()
        except BaseException:
            ctx.close()
            gguf.close()
            raise
        self._ggufs.append(gguf)
        return self._adopt_group(ctx, tensors), gguf

    def _adopt_group(self, ctx: Context, tensors: Mapping[str, Tensor]) -> Dict[str, Variable]:
        self._groups.append(ctx)
        return {name: Variable(ctx, tensor._ptr, self) for name, tensor in tensors.items()}

    # -- graphs -------------------------------------------------------------

    def _n_read_tensors(self, graph_ctx: Context) -> int:
        """How many tensors live in the contexts the graph context reads from."""
        total, seen, stack = 0, {graph_ctx}, list(graph_ctx._deps)
        while stack:
            ctx = stack.pop()
            if ctx in seen:
                continue
            seen.add(ctx)
            stack.extend(ctx._deps)
            if ctx._ptr is None:
                continue
            # Counting walks the context, so remember it until it changes.
            stamp = (ctx._epoch, ctx.used_mem)
            cached = self._counts.get(ctx)
            if cached is None or cached[0] != stamp:
                n, ptr = 0, _ggml.ggml_get_first_tensor(ctx.ptr)
                while ptr:
                    n += 1
                    ptr = _ggml.ggml_get_next_tensor(ctx.ptr, ptr)
                cached = self._counts[ctx] = (stamp, n)
            total += cached[1]
        return total

    def graph(self) -> "GraphScope":
        """Open a graph: ``with session.graph() as g:``.

        One graph is open at a time. Tensors built inside the block (inputs,
        results of operations) end with it; variables do not.
        """
        self._check_open()
        return GraphScope(self)

    def _graph_ctx(self, what: str) -> Context:
        self._check_open()
        if self._scope is None:
            raise GGMLError(
                f"{what}: no graph is open; build expressions inside "
                "`with session.graph() as g:`"
            )
        return self._ctx

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        if self._scope is not None:
            self._scope._end()
        self._closed = True
        if self._sched is not None:
            self._sched.close()
        if self._ctx is not None:
            self._ctx.close()
        for ctx in self._groups:
            ctx.close()
        for gguf in self._ggufs:
            gguf.close()
        for backend in self._owned:
            backend.close()
        self._groups, self._ggufs, self._owned, self._backends = [], [], [], ()

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        if self._closed:
            return "<ggbond.Session closed>"
        names = ", ".join(backend.name for backend in self._backends)
        return f"<ggbond.Session devices=[{names}] n_threads={self.n_threads}>"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

_POINTER = ctypes.sizeof(ctypes.c_void_p)


class GraphScope:
    """One computation: what ``with session.graph() as g:`` gives you.

    Inside the block, create inputs with :meth:`input`, write expressions with
    the ggml operations (as tensor methods, ``x.add(y)``, or through the graph,
    ``g.add(x, y)``), and call :meth:`run` to compute and download results.
    :meth:`run` can be called again, with other inputs or for other results,
    and more expressions can be added between calls.

    When the block ends, so does everything built in it: using one of its
    tensors afterwards raises :class:`GGMLError`. Variables are unaffected.
    """

    def __init__(self, session: Session):
        self._session = session
        self._state = "new"  # new -> open -> closed
        self._ctx: Optional[Context] = None
        self._cgraph: Optional[_CGraph] = None
        self._inputs: Dict[int, Tensor] = {}
        self._data: Dict[int, np.ndarray] = {}
        self._outputs: set = set()
        self._dirty = True
        self._base = 0

    # -- the block ----------------------------------------------------------

    def __enter__(self) -> "GraphScope":
        session = self._session
        session._check_open()
        if self._state != "new":
            raise GGMLError("this graph was already used; open a new one with session.graph()")
        if session._scope is not None:
            raise GGMLError(
                "another graph is still open in this session; a session runs one graph at a time"
            )
        session._sched.reset()
        session._ctx.reset()
        self._ctx = session._ctx
        self._cgraph = self._ctx.new_graph(size=session.graph_size)
        self._base = self._ctx.used_mem
        self._state = "open"
        session._scope = self
        return self

    def __exit__(self, *exc) -> None:
        self._end()

    def _end(self) -> None:
        if self._state != "open":
            return
        self._state = "closed"
        session = self._session
        session._scope = None
        self._inputs, self._data, self._outputs = {}, {}, set()
        if not session._closed:
            session._sched.reset()
            session._ctx.reset()

    def _open(self, what: str) -> Context:
        if self._state == "new":
            raise GGMLError(f"{what}: enter the graph first: `with session.graph() as g:`")
        if self._state == "closed" or self._session._closed:
            raise GGMLError(f"{what}: this graph has ended; open a new one with session.graph()")
        return self._ctx

    # -- the low-level objects underneath -----------------------------------

    @property
    def ctx(self) -> Context:
        """The :class:`ggbond.ggml.Context` the graph is built in."""
        return self._open("ctx")

    @property
    def cgraph(self) -> _CGraph:
        """The :class:`ggbond.ggml.Graph` holding what was registered so far."""
        self._open("cgraph")
        return self._cgraph

    # -- building -----------------------------------------------------------

    def input(self, data, *ne: int, dtype: Optional[DType] = None,
              name: Optional[str] = None) -> Tensor:
        """Create an input of this graph.

        ``g.input(array)`` makes an input shaped and typed like the array and
        remembers the array as its data; it is uploaded when the graph runs.
        float64 and int64 data, numpy's defaults, become F32 and I32 inputs.
        Pass ``dtype=`` to choose the type, e.g. ``g.input(x, dtype=F16)``.

        ``g.input(F32, ne0, ne1, ...)`` makes an input without data
        (dimensions in ggml order); give it data with ``feed=`` in :meth:`run`.
        """
        ctx = self._open("input")
        if isinstance(data, DType):
            if dtype is not None:
                raise ValueError("input: give the type once, as the first argument")
            if not ne:
                raise ValueError("input: g.input(dtype, ne0, ...) needs the dimensions")
            tensor = ctx.new_tensor(data, *ne, name=name).set_input()
            self._inputs[tensor.address] = tensor
            return tensor
        if ne:
            raise ValueError("input: dimensions come from the array; pass either data or a dtype")
        array, dtype = _as_array(data, dtype)
        shape = tuple(reversed(array.shape)) if array.ndim else (1,)
        tensor = ctx.new_tensor(dtype, *shape, name=name).set_input()
        self._inputs[tensor.address] = tensor
        self._data[tensor.address] = array
        return tensor

    def update(self, tensor: Tensor) -> Tensor:
        """Make ``tensor`` part of the graph even if no result depends on it.

        For operations wanted for their effect, typically a copy into a
        variable (``g.update(value.cpy(cache.view_1d(...)))``). They run in the
        order they were registered, before the results :meth:`run` asks for,
        so register a write before building what reads it back.
        """
        self._open("update")
        self._expand(tensor, "update")
        return tensor

    def __getattr__(self, name: str):
        # g.add(a, b), g.mul_mat(w, x), g.arange(...): the operations, with the
        # graph as the explicit destination.
        if name in _OP_NAMES or name == "arange":
            return getattr(self._open(name), name)
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")

    def _expand(self, tensor: Tensor, what: str) -> None:
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{what}: expected a tensor, got {type(tensor).__name__}")
        if isinstance(tensor, Variable):
            raise ValueError(
                f"{what}: {tensor.name!r} is a variable, not part of a graph; "
                "read it with .to_numpy()"
            )
        try:
            tensor.ptr
        except GGMLError:
            raise GGMLError(f"{what}: the tensor belongs to a graph that has ended") from None
        if tensor.ctx is not self._ctx:
            raise ValueError(f"{what}: the tensor was not built in this graph")
        # ggml aborts the process when a graph outgrows its capacity, so bound
        # it first: every node is a tensor built in this graph, every leaf one
        # of those or a tensor of a context its operations read from
        # (variables, or anything else the caller brought along).
        built = (self._ctx.used_mem - self._base) // tensor_overhead()
        read = self._session._n_read_tensors(self._ctx)
        limit = self._session.graph_size
        if built + read >= limit:
            raise GGMLError(
                f"{what}: the graph may not fit: {built} tensors built in it plus "
                f"{read} variables and other tensors in the contexts it reads, "
                f"against graph_size={limit}. Create the Session with a larger graph_size"
            )
        self._cgraph.build_forward_expand(tensor)
        self._dirty = True

    def _fetch(self, fetches, what: str) -> None:
        for tensor in fetches:
            if not isinstance(tensor, Tensor):
                raise TypeError(f"{what}: expected tensors, got {type(tensor).__name__}")
            if not isinstance(tensor, Variable) and tensor.address in self._outputs:
                tensor.ptr
                continue
            try:
                contiguous = isinstance(tensor, Variable) or tensor.is_contiguous
            except GGMLError:
                contiguous = True  # a stale tensor: _expand reports it
            if not contiguous:
                raise ValueError(
                    f"{what}: {tensor.name!r} is a permuted view and cannot be "
                    "downloaded as it is; fetch tensor.cont() instead"
                )
            self._expand(tensor, what)
            tensor.set_output()
            # A view's data is its source's memory, and ggml only looks at the
            # source's own flag before handing that memory to a later node.
            source = tensor.view_src
            while source is not None and source.ctx is self._ctx:
                source.set_output()
                source = source.view_src
            self._outputs.add(tensor.address)

    # -- running ------------------------------------------------------------

    def reserve(self, *fetches: Tensor) -> None:
        """Size the compute memory for this graph without running it.

        Build the largest graph you will run (longest input, fullest cache),
        reserve it once, and later graphs never have to grow the memory.
        """
        self._open("reserve")
        self._fetch(fetches, "reserve")
        sched = self._session._sched
        sched.reset()
        sched.reserve(self._cgraph)
        self._dirty = True

    def run(self, *fetches: Tensor, feed: Optional[Mapping[Tensor, object]] = None):
        """Compute the graph and return ``fetches`` as numpy arrays.

        One fetch returns one array, several return a tuple, none returns
        ``None`` (the graph still runs, for its :meth:`update` operations).
        ``feed`` maps inputs to their data for this and later runs; inputs
        created from an array already have theirs.
        """
        self._open("run")
        self._fetch(fetches, "run")
        fed = set()
        previous = dict(self._data)
        for tensor, value in (feed or {}).items():
            if isinstance(tensor, Variable):
                raise ValueError(
                    f"run: {tensor.name!r} is a variable; set it with .write(), not feed="
                )
            address = getattr(tensor, "address", None)
            try:
                known = isinstance(tensor, Tensor) and tensor.ptr and address in self._inputs
            except GGMLError:
                known = False
            if not known:
                raise ValueError("run: feed= keys must be inputs of this graph (g.input(...))")
            self._data[address] = np.asarray(value)
            fed.add(address)

        sched = self._session._sched
        graph = self._cgraph
        if self._dirty:
            # Lay the graph out afresh (the scheduler reserves for it first).
            sched.reset()
            sched.alloc_graph(graph)
            self._dirty = False

        # Upload every input each time: ggml may compute in place over them.
        try:
            for address, tensor in self._inputs.items():
                if not tensor.ptr.contents.buffer:
                    if address in fed:
                        raise ValueError(
                            f"run: input {tensor.name!r} was fed but nothing computed here uses it"
                        )
                    continue
                if address not in self._data:
                    raise ValueError(
                        f"run: input {tensor.name!r} has no data; pass it with feed={{input: array}}"
                    )
                tensor.write(self._data[address])
        except Exception:
            self._data = previous  # a rejected feed must not replace good data
            raise

        sched.compute(graph)
        results = tuple(tensor.to_numpy() for tensor in fetches)
        if not results:
            return None
        return results[0] if len(results) == 1 else results

    def __repr__(self) -> str:
        if self._state != "open":
            return f"<ggbond.GraphScope {self._state}>"
        return f"<ggbond.GraphScope n_nodes={self._cgraph.n_nodes} n_leafs={self._cgraph.n_leafs}>"

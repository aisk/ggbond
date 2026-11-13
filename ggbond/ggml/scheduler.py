"""Thin OO wrapper around ``ggml_backend_sched`` (multi-backend scheduler)."""

from __future__ import annotations

import ctypes
import weakref
from typing import TYPE_CHECKING, Optional

import ggml as _ggml

from .buffer import BufferType
from .device import DeviceType
from .errors import GGMLError, check_status, nonnull
from .gallocr import GraphAllocator

if TYPE_CHECKING:
    from .backend import Backend
    from .graph import Graph
    from .tensor import Tensor


# ``ggml_hash_size``: the scheduler's hash set holds the smallest of these that
# is at least ``graph_size``, and a graph's nodes plus leafs must fit in it.
_HASH_PRIMES = (
    2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031, 2053, 4099, 8209, 16411,
    32771, 65537, 131101, 262147, 524309, 1048583, 2097169, 4194319, 8388617,
    16777259, 33554467, 67108879, 134217757, 268435459, 536870923, 1073741827,
    2147483659,
)


def _hash_size(min_size: int) -> int:
    for prime in _HASH_PRIMES:
        if prime >= min_size:
            return prime
    return min_size | 1


def _ptr_array(items):
    arr = (ctypes.c_void_p * len(items))()
    for i, p in enumerate(items):
        arr[i] = p
    return arr


class Scheduler(GraphAllocator):
    """Splits a computation graph across one or more backends, allocates its
    working memory, and computes it -- the multi-backend analog of
    :class:`GAllocr` + :meth:`Backend.compute`.

    ``Scheduler(gpu, cpu)`` takes its backends in priority order, as
    ``ggml_backend_sched_new`` does. The scheduler keeps them alive, and closing
    a backend closes the scheduler first. Closing the scheduler frees only the
    ``ggml_backend_sched``; the backends are owned by their own wrappers.

    Like :class:`GAllocr`, the scheduler owns the memory of the graph tensors
    it allocated: once it is closed they are unallocated again and reading them
    raises :class:`GGMLError`, so download results first. The same happens to
    the tensors of the previous graph when a *different* graph is allocated
    (it is laid out over the same memory), and when a larger :meth:`reserve`
    regrows the buffers. Results stay readable after :meth:`reset` until then,
    an input shared with the next graph must be uploaded again, and allocating
    the unchanged graph again after a :meth:`reset` keeps its tensors and data.

    The scheduler holds one allocated graph at a time. :meth:`alloc_graph` (or
    a :meth:`compute` that allocates for you) makes a graph the current one, and
    :meth:`reset` releases it; allocating or computing another graph in between
    raises :class:`GGMLError`, where ggml would abort or silently compute the
    old graph. So each evaluation is ``reset()``, build, ``alloc_graph()``,
    upload inputs, ``compute()``.
    """

    def __init__(
        self,
        *backends: "Backend",
        bufts=None,
        graph_size: Optional[int] = None,
        parallel: bool = False,
        op_offload: bool = True,
    ):
        if not backends:
            raise ValueError("Scheduler needs at least one backend")
        n = len(backends)
        be_arr = _ptr_array([b.ptr for b in backends])
        last = backends[-1].device
        if last is None or last.type != DeviceType.CPU:
            raise ValueError(
                "the last backend of a Scheduler must be a CPU backend, got "
                f"{backends[-1].name!r}"
            )
        if bufts is None:
            buft_arr = None
        else:
            bufts = [b.ptr if isinstance(b, BufferType) else b for b in bufts]
            if len(bufts) != n:
                raise ValueError(
                    f"Scheduler needs one buffer type per backend: got {len(bufts)} for {n} backends"
                )
            if any(not buft for buft in bufts):
                raise ValueError("Scheduler buffer types must be non-null")
            buft_arr = _ptr_array(bufts)
        if graph_size is None:
            graph_size = _ggml.GGML_DEFAULT_GRAPH_SIZE
        elif graph_size <= 0:
            raise ValueError(f"Scheduler graph_size must be positive, got {graph_size}")
        self._ptr = nonnull(
            _ggml.ggml_backend_sched_new(
                be_arr, buft_arr, n, graph_size, parallel, op_offload
            ),
            "ggml_backend_sched_new",
        )
        self._backends = backends
        self._capacity = _hash_size(graph_size)
        # The graph ggml currently holds allocated (weakly), if any.
        self._current = None
        for backend in backends:
            backend._track(self)
        self._init_tracking()

    def _buffer_sizes(self) -> tuple:
        return tuple(
            _ggml.ggml_backend_sched_get_buffer_size(self._ptr, backend._ptr)
            for backend in self._backends
        )

    # -- state checks -------------------------------------------------------
    #
    # ggml asserts these (aborting the process) or, for a graph other than the
    # allocated one, quietly computes the allocated one instead.

    def _check_fits(self, graph: "Graph", what: str) -> None:
        n = graph.n_nodes + graph.n_leafs
        if n > self._capacity:
            raise GGMLError(
                f"{what}: graph has {n} nodes and leafs but the scheduler holds "
                f"{self._capacity}; create it with a larger graph_size"
            )

    def _check_current(self, graph: "Graph", what: str) -> bool:
        """Whether ``graph`` is the allocated one; raise if another one is."""
        if self._current is None:
            return False
        if self._current() is not graph:
            raise GGMLError(
                f"{what}: the scheduler still holds another allocated graph; "
                "call reset() before allocating or computing a new one"
            )
        return True

    # -- allocation ---------------------------------------------------------

    def reserve(self, graph: "Graph") -> None:
        """Pre-allocate backend buffers from a measure graph.

        ggml resets the scheduler afterwards, so no graph is allocated.
        """
        self._check_fits(graph, "reserve")
        if not _ggml.ggml_backend_sched_reserve(self.ptr, graph.ptr):
            raise GGMLError("ggml_backend_sched_reserve failed")
        self._current = None
        self._reserved(graph)

    def alloc_graph(self, graph: "Graph") -> None:
        """Split ``graph`` across the backends and allocate its tensors.

        Call :meth:`reset` first when a graph is already allocated.
        """
        self._check_fits(graph, "alloc_graph")
        if self._current is not None:
            raise GGMLError(
                "alloc_graph: the scheduler already holds an allocated graph; "
                "call reset() first"
            )
        self._allocating(graph)
        if not self._planned(graph):
            # ggml would lay a graph of the same shape out with its last plan.
            self.reserve(graph)
        if not _ggml.ggml_backend_sched_alloc_graph(self.ptr, graph.ptr):
            raise GGMLError("ggml_backend_sched_alloc_graph failed")
        self._current = weakref.ref(graph)
        self._allocated(graph)

    def buffer_size(self, backend: "Backend") -> int:
        """Bytes of compute buffer held for ``backend`` (``ggml_backend_sched_get_buffer_size``)."""
        if backend not in self._backends:
            raise ValueError("buffer_size: backend does not belong to this scheduler")
        return _ggml.ggml_backend_sched_get_buffer_size(self.ptr, backend.ptr)

    # -- compute ------------------------------------------------------------

    def compute(self, graph: "Graph") -> None:
        """Compute ``graph`` and wait for it (``ggml_backend_sched_graph_compute``).

        ggml synchronizes before returning, so results can be read right away.
        ``graph`` is allocated first if :meth:`alloc_graph` was not called.
        """
        self._compute(graph, _ggml.ggml_backend_sched_graph_compute, "compute")

    def compute_async(self, graph: "Graph") -> None:
        """Start computing ``graph``; call :meth:`synchronize` before reading results."""
        self._compute(graph, _ggml.ggml_backend_sched_graph_compute_async, "compute_async")

    def _compute(self, graph: "Graph", function, what: str) -> None:
        allocated = self._check_current(graph, what)
        if not allocated:
            self._check_fits(graph, what)
            self._allocating(graph)
            if not self._planned(graph):
                self.reserve(graph)
        status = function(self.ptr, graph.ptr)
        if not allocated and status != _ggml.GGML_STATUS_ALLOC_FAILED:
            # ggml allocated the graph itself.
            self._current = weakref.ref(graph)
            self._allocated(graph)
        check_status(status, f"ggml_backend_sched_graph_{what}")

    def synchronize(self) -> None:
        _ggml.ggml_backend_sched_synchronize(self.ptr)

    def reset(self) -> None:
        """Release the allocated graph and all backend assignments (``ggml_backend_sched_reset``).

        Call it before allocating or computing the next graph.
        """
        _ggml.ggml_backend_sched_reset(self.ptr)
        self._current = None

    # -- queries ------------------------------------------------------------

    @property
    def backends(self) -> "tuple[Backend, ...]":
        """The backends this scheduler was built from, in priority order."""
        return self._backends

    @property
    def n_splits(self) -> int:
        return _ggml.ggml_backend_sched_get_n_splits(self.ptr)

    @property
    def n_copies(self) -> int:
        return _ggml.ggml_backend_sched_get_n_copies(self.ptr)

    # -- tensor backend assignment ------------------------------------------

    def set_tensor_backend(self, tensor: "Tensor", backend: "Backend") -> None:
        _ggml.ggml_backend_sched_set_tensor_backend(self.ptr, tensor.ptr, backend.ptr)

    def get_tensor_backend(self, tensor: "Tensor") -> "Optional[Backend]":
        """Return the :class:`Backend` assigned to ``tensor``, or ``None``.

        The answer is one of the backends this scheduler was built from; a
        tensor with no assignment yields ``None``.
        """
        ptr = _ggml.ggml_backend_sched_get_tensor_backend(self.ptr, tensor.ptr)
        if not ptr:
            return None
        address = ctypes.cast(ptr, ctypes.c_void_p).value
        for backend in self._backends:
            if ctypes.cast(backend.ptr, ctypes.c_void_p).value == address:
                return backend
        return None

    # -- lifecycle ----------------------------------------------------------

    def _free(self, ptr) -> None:
        super()._free(ptr)
        _ggml.ggml_backend_sched_free(ptr)
        self._backends = ()
        self._current = None

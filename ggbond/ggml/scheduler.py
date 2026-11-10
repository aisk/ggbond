"""Thin OO wrapper around ``ggml_backend_sched`` (multi-backend scheduler)."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Optional

import ggml as _ggml

from .buffer import BufferType
from .errors import GGMLError, check_status, nonnull
from .handle import Handle

if TYPE_CHECKING:
    from .backend import Backend
    from .graph import Graph
    from .tensor import Tensor


def _ptr_array(items):
    arr = (ctypes.c_void_p * len(items))()
    for i, p in enumerate(items):
        arr[i] = p
    return arr


class Scheduler(Handle):
    """Splits a computation graph across one or more backends, allocates its
    working memory, and computes it -- the multi-backend analog of
    :class:`GAllocr` + :meth:`Backend.compute`.

    ``Scheduler(gpu, cpu)`` takes its backends in priority order, as
    ``ggml_backend_sched_new`` does. The scheduler keeps them alive, and closing
    a backend closes the scheduler first. Closing the scheduler frees only the
    ``ggml_backend_sched``; the backends are owned by their own wrappers.
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
        for backend in backends:
            backend._track(self)

    # -- allocation ---------------------------------------------------------

    def reserve(self, graph: "Graph") -> None:
        """Pre-allocate backend buffers from a measure graph."""
        if not _ggml.ggml_backend_sched_reserve(self.ptr, graph.ptr):
            raise GGMLError("ggml_backend_sched_reserve failed")

    def alloc_graph(self, graph: "Graph") -> None:
        if not _ggml.ggml_backend_sched_alloc_graph(self.ptr, graph.ptr):
            raise GGMLError("ggml_backend_sched_alloc_graph failed")

    # -- compute ------------------------------------------------------------

    def compute(self, graph: "Graph") -> None:
        """Compute ``graph`` and wait for it (``ggml_backend_sched_graph_compute``).

        ggml synchronizes before returning, so results can be read right away.
        """
        status = _ggml.ggml_backend_sched_graph_compute(self.ptr, graph.ptr)
        check_status(status, "ggml_backend_sched_graph_compute")

    def compute_async(self, graph: "Graph") -> None:
        """Start computing ``graph``; call :meth:`synchronize` before reading results."""
        status = _ggml.ggml_backend_sched_graph_compute_async(self.ptr, graph.ptr)
        check_status(status, "ggml_backend_sched_graph_compute_async")

    def synchronize(self) -> None:
        _ggml.ggml_backend_sched_synchronize(self.ptr)

    def reset(self) -> None:
        """Reset all assignments and allocators (call before changing backends)."""
        _ggml.ggml_backend_sched_reset(self.ptr)

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
        _ggml.ggml_backend_sched_free(ptr)
        self._backends = ()

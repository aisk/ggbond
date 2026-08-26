"""Thin OO wrapper around ``ggml_backend_sched`` (multi-backend scheduler)."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Optional, Sequence

import ggml as _ggml

from .buffer import BufferType
from .errors import GGMLError, check_status, nonnull

if TYPE_CHECKING:
    from .backend import Backend
    from .graph import Graph
    from .tensor import Tensor


def _ptr_array(items):
    arr = (ctypes.c_void_p * len(items))()
    for i, p in enumerate(items):
        arr[i] = p
    return arr


class Scheduler:
    """Splits a computation graph across one or more backends, allocates its
    working memory, and computes it -- the multi-backend analog of
    :class:`GAllocr` + :meth:`Backend.compute`.

    The scheduler holds strong refs to its :class:`Backend` objects so they
    outlive it (a scheduler must be freed before its backends). Closing the
    scheduler frees only the ``ggml_backend_sched``; the backends are owned by
    their own wrappers.
    """

    def __init__(
        self,
        backends: "Sequence[Backend]",
        *,
        bufts=None,
        graph_size: Optional[int] = None,
        parallel: bool = False,
        op_offload: bool = True,
    ):
        backends = list(backends)
        if not backends:
            raise ValueError("Scheduler needs at least one backend")
        if any(backend.ptr is None for backend in backends):
            raise ValueError("Scheduler backends must be open")
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
        self.ptr = nonnull(
            _ggml.ggml_backend_sched_new(
                be_arr, buft_arr, n, graph_size, parallel, op_offload
            ),
            "ggml_backend_sched_new",
        )
        self._backends = backends
        self._closed = False

    @classmethod
    def from_backends(cls, *backends, **kw) -> "Scheduler":
        if len(backends) == 1 and not hasattr(backends[0], "ptr"):
            backends = tuple(backends[0])  # a single iterable was passed
        return cls(backends, **kw)

    # -- allocation ---------------------------------------------------------

    def reserve(self, graph: "Graph") -> None:
        """Pre-allocate backend buffers from a measure graph."""
        if not _ggml.ggml_backend_sched_reserve(self.ptr, graph.ptr):
            raise GGMLError("ggml_backend_sched_reserve failed")

    def alloc_graph(self, graph: "Graph") -> None:
        if not _ggml.ggml_backend_sched_alloc_graph(self.ptr, graph.ptr):
            raise GGMLError("ggml_backend_sched_alloc_graph failed")

    # -- compute ------------------------------------------------------------

    def graph_compute(self, graph: "Graph") -> None:
        status = _ggml.ggml_backend_sched_graph_compute(self.ptr, graph.ptr)
        check_status(status, "ggml_backend_sched_graph_compute")

    def graph_compute_async(self, graph: "Graph") -> None:
        status = _ggml.ggml_backend_sched_graph_compute_async(self.ptr, graph.ptr)
        check_status(status, "ggml_backend_sched_graph_compute_async")

    def synchronize(self) -> None:
        _ggml.ggml_backend_sched_synchronize(self.ptr)

    def reset(self) -> None:
        """Reset all assignments and allocators (call before changing backends)."""
        _ggml.ggml_backend_sched_reset(self.ptr)

    # -- queries ------------------------------------------------------------

    def get_n_splits(self) -> int:
        return _ggml.ggml_backend_sched_get_n_splits(self.ptr)

    def get_n_copies(self) -> int:
        return _ggml.ggml_backend_sched_get_n_copies(self.ptr)

    # -- tensor backend assignment ------------------------------------------

    def set_tensor_backend(self, tensor: "Tensor", backend: "Backend") -> None:
        _ggml.ggml_backend_sched_set_tensor_backend(self.ptr, tensor.ptr, backend.ptr)

    def get_tensor_backend(self, tensor: "Tensor") -> "Optional[Backend]":
        """Return the :class:`Backend` assigned to ``tensor``, or ``None``.

        The answer is one of the backends this scheduler was built from; a
        tensor with no assignment (or one assigned to a backend the scheduler
        does not own) yields ``None``.
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

    def close(self) -> None:
        if not self._closed and self.ptr is not None:
            _ggml.ggml_backend_sched_free(self.ptr)
            self._closed = True
            self.ptr = None

    def __enter__(self) -> "Scheduler":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

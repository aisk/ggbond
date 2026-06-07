"""GGBond: an object-oriented thin wrapper over ggml, backed by ggml-python."""

from . import ggml
from .ggml import Context, Tensor, Backend, Graph, GAllocr, Scheduler, DType

__version__ = "0.1.0"
__all__ = ["ggml", "Context", "Tensor", "Backend", "Graph", "GAllocr", "Scheduler", "DType"]

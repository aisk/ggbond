"""
Type stubs for ggbond.ggml module

This file contains type annotations for the GGML Python bindings implemented in src/ggmllib.cpp
"""

from __future__ import annotations
from typing import Optional, Union, Any
import numpy as np

# Type aliases for opaque pointers
ContextPtr = Any
TensorPtr = Any
GraphPtr = Any
BackendPtr = Any
BufferPtr = Any
AllocatorPtr = Any

class Type:
    """GGML data type enumeration"""
    F32: int
    F16: int
    Q4_0: int
    Q4_1: int
    Q5_0: int
    Q5_1: int
    Q8_0: int
    Q8_1: int
    Q2_K: int
    Q3_K: int
    Q4_K: int
    Q5_K: int
    Q6_K: int
    Q8_K: int
    IQ2_XXS: int
    IQ2_XS: int
    IQ3_XXS: int
    IQ1_S: int
    IQ4_NL: int
    IQ3_S: int
    IQ2_S: int
    IQ4_XS: int
    I8: int
    I16: int
    I32: int
    I64: int
    F64: int
    IQ1_M: int

class Status:
    """GGML status enumeration"""
    ALLOC_FAILED: int
    FAILED: int
    SUCCESS: int
    ABORTED: int

# Core functions
def log_set_default() -> None: ...

# Backend management
def backend_cpu_init() -> BackendPtr:
    """Initialize CPU backend"""
    ...

def backend_is_cpu(backend: BackendPtr) -> bool:
    """Check if backend is CPU backend"""
    ...

def backend_cpu_set_n_threads(backend: BackendPtr, n_threads: int) -> None:
    """Set number of threads for CPU backend"""
    ...

# Metal backend functions (macOS only)
def backend_metal_init() -> BackendPtr:
    """Initialize Metal backend (macOS only)"""
    ...

def backend_is_metal(backend: BackendPtr) -> bool:
    """Check if backend is Metal backend"""
    ...

def backend_metal_supports_family(backend: BackendPtr, family: int) -> bool:
    """Check if Metal device supports specific feature family"""
    ...

def backend_metal_capture_next_compute(backend: BackendPtr) -> None:
    """Capture next Metal compute for debugging"""
    ...

def backend_get_default_buffer_type(backend: BackendPtr) -> BufferPtr:
    """Get default buffer type for backend"""
    ...

def backend_alloc_ctx_tensors(ctx: ContextPtr, backend: BackendPtr) -> BufferPtr:
    """Allocate all tensors in a GGML context to a backend"""
    ...

def backend_tensor_set(tensor: TensorPtr, data: buffer, offset: int = 0, size: int = 0) -> None:
    """Set tensor data from Python buffer"""
    ...

def backend_tensor_get(tensor: TensorPtr, data: buffer, offset: int = 0, size: int = 0) -> None:
    """Get tensor data to Python buffer"""
    ...

def backend_graph_compute(backend: BackendPtr, cgraph: GraphPtr) -> None:
    """Compute graph using backend"""
    ...

def backend_buffer_free(buffer: BufferPtr) -> None:
    """Free backend buffer"""
    ...

def backend_free(backend: BackendPtr) -> None:
    """Free backend"""
    ...

# Memory management and context
def context_init(mem_size: int, mem_buffer: Optional[bytes] = None, no_alloc: bool = False) -> ContextPtr:
    """Initialize GGML context"""
    ...

def context_free(ctx: ContextPtr) -> None:
    """Free GGML context and all its allocated memory"""
    ...

def tensor_overhead() -> int:
    """Get the memory overhead of a tensor"""
    ...

def graph_overhead() -> int:
    """Get the memory overhead of a graph"""
    ...

def DEFAULT_GRAPH_SIZE() -> int:
    """Default graph size constant"""
    ...

def type_size(type: int) -> int:
    """Get size in bytes for all elements in a block of the given type"""
    ...

def blck_size(type: int) -> int:
    """Get block size (number of elements per block) for the given type"""
    ...

# Tensor operations
def new_tensor_1d(ctx: ContextPtr, type: int, ne0: int) -> TensorPtr:
    """Create a new 1D tensor"""
    ...

def new_tensor_2d(ctx: ContextPtr, type: int, ne0: int, ne1: int) -> TensorPtr:
    """Create a new 2D tensor"""
    ...

def new_tensor_3d(ctx: ContextPtr, type: int, ne0: int, ne1: int, ne2: int) -> TensorPtr:
    """Create a new 3D tensor"""
    ...

def new_tensor_4d(ctx: ContextPtr, type: int, ne0: int, ne1: int, ne2: int, ne3: int) -> TensorPtr:
    """Create a new 4D tensor"""
    ...

def get_data(tensor: TensorPtr) -> int:
    """Get data pointer from tensor (as uintptr_t)"""
    ...

def get_data_f32(tensor: TensorPtr) -> int:
    """Get float32 data pointer from tensor (as uintptr_t)"""
    ...

def tensor_ne(tensor: TensorPtr, dim: int) -> int:
    """Get tensor dimension size"""
    ...

def tensor_nb(tensor: TensorPtr, dim: int) -> int:
    """Get tensor stride in bytes"""
    ...

def tensor_type(tensor: TensorPtr) -> int:
    """Get tensor type"""
    ...

def nbytes(tensor: TensorPtr) -> int:
    """Get tensor size in bytes"""
    ...

def nelements(tensor: TensorPtr) -> int:
    """Get number of elements in tensor"""
    ...

def set_input(tensor: TensorPtr) -> None:
    """Mark tensor as graph input (allocated at graph start in non-overlapping addresses)"""
    ...

def set_output(tensor: TensorPtr) -> None:
    """Mark tensor as graph output (never freed or overwritten during computation)"""
    ...

# Computation graph
def new_graph(ctx: ContextPtr) -> GraphPtr:
    """Create a new computation graph"""
    ...

def new_graph_custom(ctx: ContextPtr, size: int, grads: bool) -> GraphPtr:
    """Create a new computation graph with custom size and gradient settings"""
    ...

def mul_mat(ctx: ContextPtr, a: TensorPtr, b: TensorPtr) -> TensorPtr:
    """Matrix multiplication: result = a * b^T"""
    ...

# Binary operations
def add(ctx: ContextPtr, a: TensorPtr, b: TensorPtr) -> TensorPtr:
    """Element-wise addition: result = a + b"""
    ...

def sub(ctx: ContextPtr, a: TensorPtr, b: TensorPtr) -> TensorPtr:
    """Element-wise subtraction: result = a - b"""
    ...

def mul(ctx: ContextPtr, a: TensorPtr, b: TensorPtr) -> TensorPtr:
    """Element-wise multiplication: result = a * b"""
    ...

def div(ctx: ContextPtr, a: TensorPtr, b: TensorPtr) -> TensorPtr:
    """Element-wise division: result = a / b"""
    ...

def add1(ctx: ContextPtr, a: TensorPtr, b: TensorPtr) -> TensorPtr:
    """Add scalar b to each row of matrix a"""
    ...

def out_prod(ctx: ContextPtr, a: TensorPtr, b: TensorPtr) -> TensorPtr:
    """Outer product: result = a @ b^T"""
    ...

def concat(ctx: ContextPtr, a: TensorPtr, b: TensorPtr, dim: int) -> TensorPtr:
    """Concatenate tensors a and b along dimension dim"""
    ...

def count_equal(ctx: ContextPtr, a: TensorPtr, b: TensorPtr) -> TensorPtr:
    """Count number of equal elements in a and b"""
    ...

# Unary operations
def abs(ctx: ContextPtr, a: TensorPtr) -> TensorPtr:
    """Element-wise absolute value: result = |a|"""
    ...

def neg(ctx: ContextPtr, a: TensorPtr) -> TensorPtr:
    """Element-wise negation: result = -a"""
    ...

def sqrt(ctx: ContextPtr, a: TensorPtr) -> TensorPtr:
    """Element-wise square root: result = sqrt(a)"""
    ...

def sqr(ctx: ContextPtr, a: TensorPtr) -> TensorPtr:
    """Element-wise square: result = a^2"""
    ...

def log(ctx: ContextPtr, a: TensorPtr) -> TensorPtr:
    """Element-wise natural logarithm: result = log(a)"""
    ...

def exp(ctx: ContextPtr, a: TensorPtr) -> TensorPtr:
    """Element-wise exponential: result = exp(a)"""
    ...

def tanh(ctx: ContextPtr, a: TensorPtr) -> TensorPtr:
    """Element-wise hyperbolic tangent: result = tanh(a)"""
    ...

def sigmoid(ctx: ContextPtr, a: TensorPtr) -> TensorPtr:
    """Element-wise sigmoid activation: result = 1 / (1 + exp(-a))"""
    ...

def relu(ctx: ContextPtr, a: TensorPtr) -> TensorPtr:
    """Element-wise ReLU activation: result = max(0, a)"""
    ...

def gelu(ctx: ContextPtr, a: TensorPtr) -> TensorPtr:
    """Element-wise GELU activation (Gaussian Error Linear Unit)"""
    ...

# Reduction operations
def sum(ctx: ContextPtr, a: TensorPtr) -> TensorPtr:
    """Sum all elements in tensor, returns scalar"""
    ...

def mean(ctx: ContextPtr, a: TensorPtr) -> TensorPtr:
    """Mean of all elements along rows"""
    ...

def build_forward_expand(cgraph: GraphPtr, tensor: TensorPtr) -> None:
    """Build forward computation graph from tensor"""
    ...

def graph_compute_with_ctx(ctx: ContextPtr, cgraph: GraphPtr, n_threads: int = 1) -> int:
    """Compute the graph with given context and thread count"""
    ...

def graph_node(cgraph: GraphPtr, i: int) -> TensorPtr:
    """Get node from graph by index"""
    ...

# Graph allocator
def gallocr_new(buffer_type: BufferPtr) -> AllocatorPtr:
    """Create new graph allocator"""
    ...

def gallocr_reserve(allocr: AllocatorPtr, cgraph: GraphPtr) -> bool:
    """Reserve memory for graph computation"""
    ...

def gallocr_alloc_graph(allocr: AllocatorPtr, cgraph: GraphPtr) -> bool:
    """Allocate tensors for graph computation"""
    ...

def gallocr_get_buffer_size(allocr: AllocatorPtr, buffer_index: int = 0) -> int:
    """Get buffer size for allocated graph"""
    ...

def gallocr_free(allocr: AllocatorPtr) -> None:
    """Free graph allocator"""
    ...

# Time utilities
def time_init() -> None:
    """Initialize time measurement - call this once at the beginning of the program"""
    ...
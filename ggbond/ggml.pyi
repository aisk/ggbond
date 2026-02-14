"""
Type stubs for ggbond.ggml module

This file contains type annotations for the GGML Python bindings implemented in src/ggmllib.cpp
"""

from __future__ import annotations
from collections import abc


# Opaque pointer types
class Context:
    """Opaque pointer to ggml_context"""
    def __repr__(self) -> str: ...
    def __bool__(self) -> bool: ...

class Tensor:
    """Opaque pointer to ggml_tensor"""
    def __repr__(self) -> str: ...
    def __bool__(self) -> bool: ...

class Graph:
    """Opaque pointer to ggml_cgraph"""
    def __repr__(self) -> str: ...
    def __bool__(self) -> bool: ...

class Backend:
    """Opaque pointer to ggml_backend"""
    def __repr__(self) -> str: ...
    def __bool__(self) -> bool: ...

class Buffer:
    """Opaque pointer to ggml_backend_buffer"""
    def __repr__(self) -> str: ...
    def __bool__(self) -> bool: ...

class BufferType:
    """Opaque pointer to ggml_backend_buffer_type"""
    def __repr__(self) -> str: ...
    def __bool__(self) -> bool: ...

class GAllocr:
    """Opaque pointer to ggml_gallocr"""
    def __repr__(self) -> str: ...
    def __bool__(self) -> bool: ...

class GGUFContext:
    """Opaque pointer to gguf_context"""
    def __repr__(self) -> str: ...
    def __bool__(self) -> bool: ...

class Type:
    """GGML data type enumeration"""

    F32: Type
    F16: Type
    Q4_0: Type
    Q4_1: Type
    Q5_0: Type
    Q5_1: Type
    Q8_0: Type
    Q8_1: Type
    Q2_K: Type
    Q3_K: Type
    Q4_K: Type
    Q5_K: Type
    Q6_K: Type
    Q8_K: Type
    IQ2_XXS: Type
    IQ2_XS: Type
    IQ3_XXS: Type
    IQ1_S: Type
    IQ4_NL: Type
    IQ3_S: Type
    IQ2_S: Type
    IQ4_XS: Type
    I8: Type
    I16: Type
    I32: Type
    I64: Type
    F64: Type
    IQ1_M: Type

class OpPool:
    """GGML pooling operation enumeration"""

    MAX: OpPool
    AVG: OpPool
    COUNT: OpPool

class Status:
    """GGML status enumeration"""

    ALLOC_FAILED: Status
    FAILED: Status
    SUCCESS: Status
    ABORTED: Status

# Core functions
def log_set_default() -> None: ...

# Backend management
def backend_cpu_init() -> Backend:
    """Initialize CPU backend"""
    ...

def backend_is_cpu(backend: Backend) -> bool:
    """Check if backend is CPU backend"""
    ...

def backend_cpu_set_n_threads(backend: Backend, n_threads: int) -> None:
    """Set number of threads for CPU backend"""
    ...

# Metal backend functions (macOS only)
def backend_metal_init() -> Backend:
    """Initialize Metal backend (macOS only)"""
    ...

def backend_is_metal(backend: Backend) -> bool:
    """Check if backend is Metal backend"""
    ...

def backend_metal_supports_family(backend: Backend, family: int) -> bool:
    """Check if Metal device supports specific feature family"""
    ...

def backend_metal_capture_next_compute(backend: Backend) -> None:
    """Capture next Metal compute for debugging"""
    ...

def backend_get_default_buffer_type(backend: Backend) -> BufferType:
    """Get default buffer type for backend"""
    ...

def backend_alloc_ctx_tensors(ctx: Context, backend: Backend) -> Buffer:
    """Allocate all tensors in a GGML context to a backend"""
    ...

def backend_tensor_set(
    tensor: Tensor, data: abc.Buffer, offset: int = 0, size: int = 0
) -> None:
    """Set tensor data from Python buffer"""
    ...

def backend_tensor_get(
    tensor: Tensor, data: abc.Buffer, offset: int = 0, size: int = 0
) -> None:
    """Get tensor data to Python buffer"""
    ...

def backend_graph_compute(backend: Backend, cgraph: Graph) -> None:
    """Compute graph using backend"""
    ...

def backend_buffer_free(buffer: Buffer) -> None:
    """Free backend buffer"""
    ...

def backend_buffer_is_host(buffer: Buffer) -> bool:
    """Check if buffer is in host memory"""
    ...

def backend_free(backend: Backend) -> None:
    """Free backend"""
    ...

# Memory management and context
def context_init(
    mem_size: int, mem_buffer: bytes | None = None, no_alloc: bool = False
) -> Context:
    """Initialize GGML context"""
    ...

def context_free(ctx: Context) -> None:
    """Free GGML context and all its allocated memory"""
    ...

def tensor_overhead() -> int:
    """Get the memory overhead of a tensor"""
    ...

def graph_overhead() -> int:
    """Get the memory overhead of a graph"""
    ...

def graph_overhead_custom(size: int, grads: bool) -> int:
    """Get memory overhead for custom graph size"""
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

def ftype_to_ggml_type(ftype: int) -> int:
    """Convert file type to tensor type"""
    ...

# Tensor operations
def new_tensor_1d(ctx: Context, type: int, ne0: int) -> Tensor:
    """Create a new 1D tensor"""
    ...

def new_tensor_2d(ctx: Context, type: int, ne0: int, ne1: int) -> Tensor:
    """Create a new 2D tensor"""
    ...

def new_tensor_3d(ctx: Context, type: int, ne0: int, ne1: int, ne2: int) -> Tensor:
    """Create a new 3D tensor"""
    ...

def new_tensor_4d(
    ctx: Context, type: int, ne0: int, ne1: int, ne2: int, ne3: int
) -> Tensor:
    """Create a new 4D tensor"""
    ...

def get_data(tensor: Tensor) -> int:
    """Get data pointer from tensor (as uintptr_t)"""
    ...

def get_data_f32(tensor: Tensor) -> int:
    """Get float32 data pointer from tensor (as uintptr_t)"""
    ...

def tensor_ne(tensor: Tensor, dim: int) -> int:
    """Get tensor dimension size"""
    ...

def tensor_nb(tensor: Tensor, dim: int) -> int:
    """Get tensor stride in bytes"""
    ...

def tensor_type(tensor: Tensor) -> int:
    """Get tensor type"""
    ...

def nbytes(tensor: Tensor) -> int:
    """Get tensor size in bytes"""
    ...

def nelements(tensor: Tensor) -> int:
    """Get number of elements in tensor"""
    ...

def set_input(tensor: Tensor) -> None:
    """Mark tensor as graph input (allocated at graph start in non-overlapping addresses)"""
    ...

def set_output(tensor: Tensor) -> None:
    """Mark tensor as graph output (never freed or overwritten during computation)"""
    ...

def get_tensor(ctx: Context, name: str) -> Tensor:
    """Get tensor from context by name"""
    ...

def set_name(tensor: Tensor, name: str) -> Tensor:
    """Set tensor name"""
    ...

# Computation graph
def new_graph(ctx: Context) -> Graph:
    """Create a new computation graph"""
    ...

def new_graph_custom(ctx: Context, size: int, grads: bool) -> Graph:
    """Create a new computation graph with custom size and gradient settings"""
    ...

def mul_mat(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication: result = a * b^T"""
    ...

# Binary operations
def add(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
    """Element-wise addition: result = a + b"""
    ...

def sub(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
    """Element-wise subtraction: result = a - b"""
    ...

def mul(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
    """Element-wise multiplication: result = a * b"""
    ...

def div(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
    """Element-wise division: result = a / b"""
    ...

def add1(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
    """Add scalar b to each row of matrix a"""
    ...

def out_prod(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
    """Outer product: result = a @ b^T"""
    ...

def concat(ctx: Context, a: Tensor, b: Tensor, dim: int) -> Tensor:
    """Concatenate tensors a and b along dimension dim"""
    ...

def count_equal(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
    """Count number of equal elements in a and b"""
    ...

# Unary operations
def abs(ctx: Context, a: Tensor) -> Tensor:
    """Element-wise absolute value: result = |a|"""
    ...

def neg(ctx: Context, a: Tensor) -> Tensor:
    """Element-wise negation: result = -a"""
    ...

def sqrt(ctx: Context, a: Tensor) -> Tensor:
    """Element-wise square root: result = sqrt(a)"""
    ...

def sqr(ctx: Context, a: Tensor) -> Tensor:
    """Element-wise square: result = a^2"""
    ...

def log(ctx: Context, a: Tensor) -> Tensor:
    """Element-wise natural logarithm: result = log(a)"""
    ...

def exp(ctx: Context, a: Tensor) -> Tensor:
    """Element-wise exponential: result = exp(a)"""
    ...

def tanh(ctx: Context, a: Tensor) -> Tensor:
    """Element-wise hyperbolic tangent: result = tanh(a)"""
    ...

def sigmoid(ctx: Context, a: Tensor) -> Tensor:
    """Element-wise sigmoid activation: result = 1 / (1 + exp(-a))"""
    ...

def relu(ctx: Context, a: Tensor) -> Tensor:
    """Element-wise ReLU activation: result = max(0, a)"""
    ...

def gelu(ctx: Context, a: Tensor) -> Tensor:
    """Element-wise GELU activation (Gaussian Error Linear Unit)"""
    ...

# Reduction operations
def sum(ctx: Context, a: Tensor) -> Tensor:
    """Sum all elements in tensor, returns scalar"""
    ...

def mean(ctx: Context, a: Tensor) -> Tensor:
    """Mean of all elements along rows"""
    ...

# Normalization and activation
def norm(ctx: Context, a: Tensor, eps: float) -> Tensor:
    """Normalize tensor along rows"""
    ...

def soft_max(ctx: Context, a: Tensor) -> Tensor:
    """Apply softmax activation"""
    ...

# Reshape operations
def reshape_2d(ctx: Context, a: Tensor, ne0: int, ne1: int) -> Tensor:
    """Reshape tensor to 2D"""
    ...

def reshape_3d(ctx: Context, a: Tensor, ne0: int, ne1: int, ne2: int) -> Tensor:
    """Reshape tensor to 3D"""
    ...

# Transpose and contiguous
def transpose(ctx: Context, a: Tensor) -> Tensor:
    """Transpose tensor (permute 1,0,2,3)"""
    ...

def cont(ctx: Context, a: Tensor) -> Tensor:
    """Make tensor contiguous in memory"""
    ...

def cont_2d(ctx: Context, a: Tensor, ne0: int, ne1: int) -> Tensor:
    """Make tensor contiguous with 2D shape"""
    ...

def cont_3d(ctx: Context, a: Tensor, ne0: int, ne1: int, ne2: int) -> Tensor:
    """Make tensor contiguous with 3D shape"""
    ...

# View operations (for KV cache and Q/K/V split)
def view_1d(ctx: Context, a: Tensor, ne0: int, offset: int) -> Tensor:
    """Create 1D view of tensor"""
    ...

def view_2d(
    ctx: Context, a: Tensor, ne0: int, ne1: int, nb1: int, offset: int
) -> Tensor:
    """Create 2D view of tensor"""
    ...

# Embedding lookup
def get_rows(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
    """Get rows by index (a: data, b: indices)"""
    ...

# Dimension operations
def permute(
    ctx: Context, a: Tensor, axis0: int, axis1: int, axis2: int, axis3: int
) -> Tensor:
    """Permute tensor dimensions"""
    ...

# Math operations
def scale(ctx: Context, a: Tensor, s: float) -> Tensor:
    """Scale tensor by scalar"""
    ...

def diag_mask_inf(ctx: Context, a: Tensor, n_past: int) -> Tensor:
    """Apply diagonal mask with -inf for causal attention"""
    ...

# Copy operations
def cpy(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
    """Copy tensor a to b"""
    ...

# Helper functions
def element_size(tensor: Tensor) -> int:
    """Get element size in bytes"""
    ...

def build_forward_expand(cgraph: Graph, tensor: Tensor) -> None:
    """Build forward computation graph from tensor"""
    ...

def graph_compute_with_ctx(ctx: Context, cgraph: Graph, n_threads: int = 1) -> int:
    """Compute the graph with given context and thread count"""
    ...

def graph_node(cgraph: Graph, i: int) -> Tensor:
    """Get node from graph by index"""
    ...

def graph_get_tensor(cgraph: Graph, name: str) -> Tensor:
    """Get tensor from graph by name"""
    ...

# Graph allocator
def gallocr_new(buffer_type: BufferType) -> GAllocr:
    """Create new graph allocator"""
    ...

def gallocr_reserve(allocr: GAllocr, cgraph: Graph) -> bool:
    """Reserve memory for graph computation"""
    ...

def gallocr_alloc_graph(allocr: GAllocr, cgraph: Graph) -> bool:
    """Allocate tensors for graph computation"""
    ...

def gallocr_get_buffer_size(allocr: GAllocr, buffer_index: int = 0) -> int:
    """Get buffer size for allocated graph"""
    ...

def gallocr_free(allocr: GAllocr) -> None:
    """Free graph allocator"""
    ...

# Pooling operations
def pool_1d(ctx: Context, a: Tensor, op: OpPool, k0: int, s0: int, p0: int) -> Tensor:
    """1D pooling (max or average)"""
    ...

# GGUF functions
def gguf_init_from_file(
    fname: str, no_alloc: bool = True
) -> tuple[GGUFContext, Context]:
    """Initialize GGUF context from file, returns (gguf_ctx, meta_ctx) tuple"""
    ...

def gguf_free(ctx: GGUFContext) -> None:
    """Free GGUF context"""
    ...

def gguf_get_n_tensors(ctx: GGUFContext) -> int:
    """Get number of tensors in GGUF file"""
    ...

def gguf_get_data_offset(ctx: GGUFContext) -> int:
    """Get data offset in GGUF file"""
    ...

def gguf_get_tensor_offset(ctx: GGUFContext, tensor_id: int) -> int:
    """Get tensor offset in GGUF file"""
    ...

def gguf_get_tensor_name(ctx: GGUFContext, tensor_id: int) -> str:
    """Get tensor name by index"""
    ...

# Time utilities
def time_init() -> None:
    """Initialize time measurement - call this once at the beginning of the program"""
    ...

def time_us() -> int:
    """Get current time in microseconds"""
    ...

# Type information
def type_name(type: int) -> str:
    """Get type name string"""
    ...

# Backend buffer
def backend_buffer_get_size(buffer: Buffer) -> int:
    """Get buffer size in bytes"""
    ...

# Constants
FILE_MAGIC: int
QNT_VERSION_FACTOR: int

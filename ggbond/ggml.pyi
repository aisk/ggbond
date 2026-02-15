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

class SortOrder:
    """GGML sort order enumeration"""

    ASC: SortOrder
    DESC: SortOrder

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

def n_dims(tensor: Tensor) -> int:
    """Get number of dimensions"""
    ...

def nrows(tensor: Tensor) -> int:
    """Get number of rows"""
    ...

def get_name(tensor: Tensor) -> str:
    """Get tensor name"""
    ...

def is_contiguous(tensor: Tensor) -> bool:
    """Check if tensor is contiguous"""
    ...

def is_transposed(tensor: Tensor) -> bool:
    """Check if tensor is transposed"""
    ...

def is_permuted(tensor: Tensor) -> bool:
    """Check if tensor is permuted"""
    ...

def is_quantized(type: Type) -> bool:
    """Check if type is quantized"""
    ...

def are_same_shape(a: Tensor, b: Tensor) -> bool:
    """Check if two tensors have the same shape"""
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

def sin(ctx: Context, a: Tensor) -> Tensor:
    """Element-wise sine"""
    ...

def cos(ctx: Context, a: Tensor) -> Tensor:
    """Element-wise cosine"""
    ...

def sgn(ctx: Context, a: Tensor) -> Tensor:
    """Element-wise sign function"""
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

def silu(ctx: Context, a: Tensor) -> Tensor:
    """SiLU activation (x * sigmoid(x))"""
    ...

def leaky_relu(ctx: Context, a: Tensor, negative_slope: float, inplace: bool) -> Tensor:
    """Leaky ReLU activation"""
    ...

def gelu_quick(ctx: Context, a: Tensor) -> Tensor:
    """Fast GELU approximation"""
    ...

def hardswish(ctx: Context, a: Tensor) -> Tensor:
    """Hard swish activation"""
    ...

def hardsigmoid(ctx: Context, a: Tensor) -> Tensor:
    """Hard sigmoid activation"""
    ...

def elu(ctx: Context, a: Tensor) -> Tensor:
    """ELU activation"""
    ...

# Reduction operations
def sum(ctx: Context, a: Tensor) -> Tensor:
    """Sum all elements in tensor, returns scalar"""
    ...

def sum_rows(ctx: Context, a: Tensor) -> Tensor:
    """Sum elements along rows"""
    ...

def mean(ctx: Context, a: Tensor) -> Tensor:
    """Mean of all elements along rows"""
    ...

def argmax(ctx: Context, a: Tensor) -> Tensor:
    """Argmax along rows"""
    ...

# Normalization and activation
def norm(ctx: Context, a: Tensor, eps: float) -> Tensor:
    """Normalize tensor along rows"""
    ...

def rms_norm(ctx: Context, a: Tensor, eps: float) -> Tensor:
    """RMS normalization"""
    ...

def group_norm(ctx: Context, a: Tensor, n_groups: int, eps: float) -> Tensor:
    """Group normalization"""
    ...

def l2_norm(ctx: Context, a: Tensor, eps: float) -> Tensor:
    """L2 normalization"""
    ...

def soft_max(ctx: Context, a: Tensor) -> Tensor:
    """Apply softmax activation"""
    ...

# Reshape operations
def reshape_1d(ctx: Context, a: Tensor, ne0: int) -> Tensor:
    """Reshape tensor to 1D"""
    ...

def reshape_2d(ctx: Context, a: Tensor, ne0: int, ne1: int) -> Tensor:
    """Reshape tensor to 2D"""
    ...

def reshape_3d(ctx: Context, a: Tensor, ne0: int, ne1: int, ne2: int) -> Tensor:
    """Reshape tensor to 3D"""
    ...

def reshape_4d(
    ctx: Context, a: Tensor, ne0: int, ne1: int, ne2: int, ne3: int
) -> Tensor:
    """Reshape tensor to 4D"""
    ...

# Transpose and contiguous
def transpose(ctx: Context, a: Tensor) -> Tensor:
    """Transpose tensor (permute 1,0,2,3)"""
    ...

def cont(ctx: Context, a: Tensor) -> Tensor:
    """Make tensor contiguous in memory"""
    ...

def cont_1d(ctx: Context, a: Tensor, ne0: int) -> Tensor:
    """Make tensor contiguous with 1D shape"""
    ...

def cont_2d(ctx: Context, a: Tensor, ne0: int, ne1: int) -> Tensor:
    """Make tensor contiguous with 2D shape"""
    ...

def cont_3d(ctx: Context, a: Tensor, ne0: int, ne1: int, ne2: int) -> Tensor:
    """Make tensor contiguous with 3D shape"""
    ...

def cont_4d(
    ctx: Context, a: Tensor, ne0: int, ne1: int, ne2: int, ne3: int
) -> Tensor:
    """Make tensor contiguous with 4D shape"""
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

def view_3d(
    ctx: Context,
    a: Tensor,
    ne0: int,
    ne1: int,
    ne2: int,
    nb1: int,
    nb2: int,
    offset: int,
) -> Tensor:
    """Create 3D view of tensor"""
    ...

def view_4d(
    ctx: Context,
    a: Tensor,
    ne0: int,
    ne1: int,
    ne2: int,
    ne3: int,
    nb1: int,
    nb2: int,
    nb3: int,
    offset: int,
) -> Tensor:
    """Create 4D view of tensor"""
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

def clamp(ctx: Context, a: Tensor, min: float, max: float) -> Tensor:
    """Clamp tensor values to [min, max]"""
    ...

def repeat(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
    """Repeat tensor a to match shape of b"""
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

def pool_2d(ctx: Context, a: Tensor, op: OpPool, k0: int, k1: int, s0: int, s1: int, p0: float, p1: float) -> Tensor:
    """2D pooling (max or average)"""
    ...

# Attention-related operations
def soft_max_ext(ctx: Context, a: Tensor, mask: Tensor, scale: float, max_bias: float) -> Tensor:
    """Softmax with mask, scale and ALiBi bias"""
    ...

def diag_mask_zero(ctx: Context, a: Tensor, n_past: int) -> Tensor:
    """Set elements above the diagonal to 0"""
    ...

def rope(ctx: Context, a: Tensor, b: Tensor, n_dims: int, mode: int) -> Tensor:
    """Rotary position embedding"""
    ...

def rope_ext(
    ctx: Context, a: Tensor, b: Tensor, c: Tensor,
    n_dims: int, mode: int, n_ctx_orig: int,
    freq_base: float, freq_scale: float,
    ext_factor: float, attn_factor: float,
    beta_fast: float, beta_slow: float,
) -> Tensor:
    """Extended rotary position embedding"""
    ...

def flash_attn_ext(
    ctx: Context, q: Tensor, k: Tensor, v: Tensor, mask: Tensor,
    scale: float, max_bias: float, logit_softcap: float,
) -> Tensor:
    """Flash Attention"""
    ...

# Convolution operations
def conv_1d(ctx: Context, a: Tensor, b: Tensor, s0: int, p0: int, d0: int) -> Tensor:
    """1D convolution"""
    ...

def conv_2d(
    ctx: Context, a: Tensor, b: Tensor,
    s0: int, s1: int, p0: int, p1: int, d0: int, d1: int,
) -> Tensor:
    """2D convolution"""
    ...

# Tensor operations
def dup(ctx: Context, a: Tensor) -> Tensor:
    """Duplicate tensor"""
    ...

def dup_tensor(ctx: Context, src: Tensor) -> Tensor:
    """Duplicate tensor structure (without data)"""
    ...

def view_tensor(ctx: Context, src: Tensor) -> Tensor:
    """Create a view of tensor"""
    ...

def cast(ctx: Context, a: Tensor, type: Type) -> Tensor:
    """Cast tensor to different type"""
    ...

def pad(ctx: Context, a: Tensor, p0: int, p1: int, p2: int, p3: int) -> Tensor:
    """Pad tensor with zeros"""
    ...

def arange(ctx: Context, start: float, stop: float, step: float) -> Tensor:
    """Create range tensor"""
    ...

def argsort(ctx: Context, a: Tensor, order: SortOrder) -> Tensor:
    """Get sorted indices"""
    ...

def top_k(ctx: Context, a: Tensor, k: int) -> Tensor:
    """Top-K elements per row"""
    ...

def diag(ctx: Context, a: Tensor) -> Tensor:
    """Create diagonal matrix"""
    ...

def repeat_4d(ctx: Context, a: Tensor, ne0: int, ne1: int, ne2: int, ne3: int) -> Tensor:
    """Repeat tensor to specified shape"""
    ...

def set_rows(ctx: Context, a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Set rows by index (a: dest, b: src rows, c: indices)"""
    ...

def scale_bias(ctx: Context, a: Tensor, s: float, b: float) -> Tensor:
    """Scale and bias: x = s*a + b"""
    ...

def cross_entropy_loss(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
    """Cross entropy loss (a: logits, b: labels)"""
    ...

# Tensor property queries
def is_scalar(tensor: Tensor) -> bool:
    """Check if tensor is scalar"""
    ...

def is_vector(tensor: Tensor) -> bool:
    """Check if tensor is vector"""
    ...

def is_matrix(tensor: Tensor) -> bool:
    """Check if tensor is matrix"""
    ...

def is_3d(tensor: Tensor) -> bool:
    """Check if tensor is 3D"""
    ...

def is_empty(tensor: Tensor) -> bool:
    """Check if tensor is empty"""
    ...

def are_same_stride(a: Tensor, b: Tensor) -> bool:
    """Check if two tensors have the same stride"""
    ...

def can_repeat(a: Tensor, b: Tensor) -> bool:
    """Check if tensor a can be repeated to shape of b"""
    ...

def nbytes_pad(tensor: Tensor) -> int:
    """Get padded tensor size in bytes"""
    ...

def row_size(type: Type, ne: int) -> int:
    """Get row size in bytes"""
    ...

def op_desc(tensor: Tensor) -> str:
    """Get operation description"""
    ...

# Tensor value operations (from ggml-cpu.h)
def new_f32(ctx: Context, value: float) -> Tensor:
    """Create scalar f32 tensor"""
    ...

def new_i32(ctx: Context, value: int) -> Tensor:
    """Create scalar i32 tensor"""
    ...

def set_f32(tensor: Tensor, value: float) -> Tensor:
    """Set all elements to f32 value"""
    ...

def set_i32(tensor: Tensor, value: int) -> Tensor:
    """Set all elements to i32 value"""
    ...

def set_zero(tensor: Tensor) -> Tensor:
    """Set all elements to zero"""
    ...

def get_f32_1d(tensor: Tensor, i: int) -> float:
    """Get single f32 value"""
    ...

def set_f32_1d(tensor: Tensor, i: int, value: float) -> None:
    """Set single f32 value"""
    ...

def get_i32_1d(tensor: Tensor, i: int) -> int:
    """Get single i32 value"""
    ...

def set_i32_1d(tensor: Tensor, i: int, value: int) -> None:
    """Set single i32 value"""
    ...

# Graph operations
def graph_n_nodes(cgraph: Graph) -> int:
    """Get number of nodes in graph"""
    ...

def graph_size(cgraph: Graph) -> int:
    """Get graph capacity"""
    ...

def graph_print(cgraph: Graph) -> None:
    """Print graph info"""
    ...

def graph_dump_dot(gb: Graph, gf: Graph, filename: str) -> None:
    """Export graph to dot file"""
    ...

# Context info
def used_mem(ctx: Context) -> int:
    """Get used memory in context"""
    ...

def set_no_alloc(ctx: Context, no_alloc: bool) -> None:
    """Set no_alloc flag"""
    ...

def get_no_alloc(ctx: Context) -> bool:
    """Get no_alloc flag"""
    ...

def get_first_tensor(ctx: Context) -> Tensor:
    """Get first tensor in context"""
    ...

def get_next_tensor(ctx: Context, tensor: Tensor) -> Tensor:
    """Get next tensor in context"""
    ...

# GLU activations
def swiglu(ctx: Context, a: Tensor) -> Tensor:
    """SwiGLU activation"""
    ...

def reglu(ctx: Context, a: Tensor) -> Tensor:
    """ReGLU activation"""
    ...

def geglu(ctx: Context, a: Tensor) -> Tensor:
    """GEGLU activation"""
    ...

# Parameter and loss marking
def set_param(tensor: Tensor) -> None:
    """Mark tensor as trainable parameter"""
    ...

def set_loss(tensor: Tensor) -> None:
    """Mark tensor as loss"""
    ...

# Backend tensor operations
def backend_tensor_copy(src: Tensor, dst: Tensor) -> None:
    """Copy tensor between backends"""
    ...

def backend_tensor_memset(tensor: Tensor, value: int, offset: int, size: int) -> None:
    """Fill tensor memory with value"""
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

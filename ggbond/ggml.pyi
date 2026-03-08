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

class GluOp:
    """GGML GLU operation enumeration"""

    REGLU: GluOp
    GEGLU: GluOp
    SWIGLU: GluOp
    SWIGLU_OAI: GluOp
    GEGLU_ERF: GluOp
    GEGLU_QUICK: GluOp

class ScaleMode:
    """GGML interpolation mode enumeration"""

    NEAREST: ScaleMode
    BILINEAR: ScaleMode

class ScaleFlag:
    """GGML interpolation flag enumeration"""

    ALIGN_CORNERS: ScaleFlag

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

# CUDA-like backend functions (CUDA/HIP/MUSA, when enabled at build time)
def backend_cuda_init(device: int = 0) -> Backend:
    """Initialize CUDA-like backend (CUDA/HIP/MUSA)"""
    ...

def backend_is_cuda(backend: Backend) -> bool:
    """Check if backend is CUDA-like backend"""
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
    """Compute graph using backend. Raises RuntimeError if compute fails."""
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
    """Initialize GGML context. Raises MemoryError if allocation fails."""
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
    """Create a new 1D tensor. Raises MemoryError if allocation fails."""
    ...

def new_tensor_2d(ctx: Context, type: int, ne0: int, ne1: int) -> Tensor:
    """Create a new 2D tensor. Raises MemoryError if allocation fails."""
    ...

def new_tensor_3d(ctx: Context, type: int, ne0: int, ne1: int, ne2: int) -> Tensor:
    """Create a new 3D tensor. Raises MemoryError if allocation fails."""
    ...

def new_tensor_4d(
    ctx: Context, type: int, ne0: int, ne1: int, ne2: int, ne3: int
) -> Tensor:
    """Create a new 4D tensor. Raises MemoryError if allocation fails."""
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

def gguf_get_n_kv(ctx: GGUFContext) -> int:
    """Get number of KV pairs in GGUF file"""
    ...

def gguf_find_key(ctx: GGUFContext, key: str) -> int:
    """Find key index by name, returns -1 if not found"""
    ...

def gguf_get_key(ctx: GGUFContext, key_id: int) -> str:
    """Get key name by index"""
    ...

def gguf_get_kv_type(ctx: GGUFContext, key_id: int) -> int:
    """Get KV pair type by key index"""
    ...

def gguf_get_val_str(ctx: GGUFContext, key_id: int) -> str:
    """Get string value by key index"""
    ...

def gguf_get_val_u32(ctx: GGUFContext, key_id: int) -> int:
    """Get uint32 value by key index"""
    ...

def gguf_get_val_i32(ctx: GGUFContext, key_id: int) -> int:
    """Get int32 value by key index"""
    ...

def gguf_get_val_f32(ctx: GGUFContext, key_id: int) -> float:
    """Get float32 value by key index"""
    ...

def gguf_get_val_i64(ctx: GGUFContext, key_id: int) -> int:
    """Get int64 value by key index"""
    ...

def gguf_get_val_i64(ctx: GGUFContext, key_id: int) -> int:
    """Get int64 value by key index"""
    ...

def gguf_get_arr_n(ctx: GGUFContext, key_id: int) -> int:
    """Get array length by key index"""
    ...

def gguf_get_arr_type(ctx: GGUFContext, key_id: int) -> int:
    """Get array element type by key index"""
    ...

def gguf_get_arr_str(ctx: GGUFContext, key_id: int, i: int) -> str:
    """Get i-th string from array by key index"""
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

def is_contiguous_rows(tensor: Tensor) -> bool:
    """Check if tensor has contiguous rows"""
    ...

def step(ctx: Context, a: Tensor) -> Tensor:
    """Step activation: result = (a > 0) ? 1 : 0"""
    ...

def step_inplace(ctx: Context, a: Tensor) -> Tensor:
    """Step activation (in-place)"""
    ...

def gelu_erf(ctx: Context, a: Tensor) -> Tensor:
    """GELU activation using ERF"""
    ...

def gelu_erf_inplace(ctx: Context, a: Tensor) -> Tensor:
    """GELU activation using ERF (in-place)"""
    ...

def add_cast(ctx: Context, a: Tensor, b: Tensor, type: Type) -> Tensor:
    """Element-wise addition with type casting"""
    ...

def acc(ctx: Context, a: Tensor, b: Tensor, nb1: int, nb2: int, nb3: int, offset: int) -> Tensor:
    """Accumulate b into a view of a at offset"""
    ...

def acc_inplace(ctx: Context, a: Tensor, b: Tensor, nb1: int, nb2: int, nb3: int, offset: int) -> Tensor:
    """Accumulate b into a view of a at offset (in-place)"""
    ...

def mul_mat_id(ctx: Context, as: Tensor, b: Tensor, ids: Tensor) -> Tensor:
    """MoE indirect matrix multiplication"""
    ...

def geglu_erf(ctx: Context, a: Tensor) -> Tensor:
    """GEGLU with ERF gating (fused, gate in second half)"""
    ...

def geglu_erf_swapped(ctx: Context, a: Tensor) -> Tensor:
    """GEGLU with ERF gating (fused, gate in first half)"""
    ...

def geglu_quick(ctx: Context, a: Tensor) -> Tensor:
    """GEGLU with quick GELU gating (fused, gate in second half)"""
    ...

def geglu_quick_swapped(ctx: Context, a: Tensor) -> Tensor:
    """GEGLU with quick GELU gating (fused, gate in first half)"""
    ...

def reglu_split(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
    """ReGLU (split: a=input, b=gate)"""
    ...

def geglu_split(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
    """GEGLU (split: a=input, b=gate)"""
    ...

def swiglu_split(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
    """SwiGLU (split: a=input, b=gate)"""
    ...

def geglu_erf_split(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
    """GEGLU-ERF (split: a=input, b=gate)"""
    ...

def geglu_quick_split(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
    """GEGLU-quick (split: a=input, b=gate)"""
    ...

def swiglu_oai(ctx: Context, a: Tensor, b: Tensor, alpha: float, limit: float) -> Tensor:
    """OpenAI SwiGLU with alpha/limit"""
    ...

def interpolate(ctx: Context, a: Tensor, ne0: int, ne1: int, ne2: int, ne3: int, mode: int) -> Tensor:
    """Interpolate (resize) tensor to new spatial dimensions"""
    ...

def pad_ext(ctx: Context, a: Tensor, lp0: int, rp0: int, lp1: int, rp1: int, lp2: int, rp2: int, lp3: int, rp3: int) -> Tensor:
    """Pad tensor with independent left/right padding per dimension"""
    ...

def timestep_embedding(ctx: Context, timesteps: Tensor, dim: int, max_period: int) -> Tensor:
    """Sinusoidal timestep embedding for diffusion models"""
    ...

def im2col(ctx: Context, a: Tensor, b: Tensor, s0: int, s1: int, p0: int, p1: int, d0: int, d1: int, is_2D: bool, dst_type: Type) -> Tensor:
    """Image-to-column transform (underlying primitive for conv)"""
    ...

def conv_1d_ph(ctx: Context, a: Tensor, b: Tensor, s: int, d: int) -> Tensor:
    """1D convolution with half-padding"""
    ...

def conv_1d_dw(ctx: Context, a: Tensor, b: Tensor, s0: int, p0: int, d0: int) -> Tensor:
    """Depthwise 1D convolution"""
    ...

def conv_1d_dw_ph(ctx: Context, a: Tensor, b: Tensor, s0: int, d0: int) -> Tensor:
    """Depthwise 1D convolution with half-padding"""
    ...

def conv_transpose_1d(ctx: Context, a: Tensor, b: Tensor, s0: int, p0: int, d0: int) -> Tensor:
    """1D transposed convolution (deconvolution)"""
    ...

def conv_2d_sk_p0(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
    """2D convolution with stride=kernel, padding=0 (e.g. SAM patch embed)"""
    ...

def conv_2d_s1_ph(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
    """2D convolution with stride=1, half-padding (e.g. SAM encoder)"""
    ...

def conv_2d_dw(ctx: Context, a: Tensor, b: Tensor, s0: int, s1: int, p0: int, p1: int, d0: int, d1: int) -> Tensor:
    """Depthwise 2D convolution"""
    ...

def conv_3d(ctx: Context, a: Tensor, b: Tensor, IC: int, s0: int, s1: int, s2: int, p0: int, p1: int, p2: int, d0: int, d1: int, d2: int) -> Tensor:
    """3D convolution"""
    ...

def win_part(ctx: Context, a: Tensor, w: int) -> Tensor:
    """Partition tensor into non-overlapping windows of size w"""
    ...

def win_unpart(ctx: Context, a: Tensor, w0: int, h0: int, w: int) -> Tensor:
    """Reverse window partition (reconstruct spatial tensor)"""
    ...

def get_rel_pos(ctx: Context, a: Tensor, qh: int, kh: int) -> Tensor:
    """Slice relative position bias for given query/key heights"""
    ...

def add_rel_pos(ctx: Context, a: Tensor, pw: Tensor, ph: Tensor) -> Tensor:
    """Add relative position bias to attention scores"""
    ...

def add_rel_pos_inplace(ctx: Context, a: Tensor, pw: Tensor, ph: Tensor) -> Tensor:
    """Add relative position bias to attention scores (in-place)"""
    ...

def ssm_conv(ctx: Context, sx: Tensor, c: Tensor) -> Tensor:
    """Mamba SSM convolution step"""
    ...

def ssm_scan(ctx: Context, s: Tensor, x: Tensor, dt: Tensor, A: Tensor, B: Tensor, C: Tensor, ids: Tensor) -> Tensor:
    """Mamba SSM selective scan"""
    ...

def rwkv_wkv6(ctx: Context, k: Tensor, v: Tensor, r: Tensor, tf: Tensor, td: Tensor, state: Tensor) -> Tensor:
    """RWKV WKV6 recurrent step"""
    ...

def rwkv_wkv7(ctx: Context, r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor, state: Tensor) -> Tensor:
    """RWKV WKV7 recurrent step"""
    ...

def gated_linear_attn(ctx: Context, k: Tensor, v: Tensor, q: Tensor, g: Tensor, state: Tensor, scale: float) -> Tensor:
    """Gated Linear Attention step"""
    ...

def opt_step_adamw(ctx: Context, a: Tensor, grad: Tensor, m: Tensor, v: Tensor, adamw_params: Tensor) -> Tensor:
    """AdamW optimizer step"""
    ...

def opt_step_sgd(ctx: Context, a: Tensor, grad: Tensor, sgd_params: Tensor) -> Tensor:
    """SGD optimizer step"""
    ...

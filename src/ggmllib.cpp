#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>
#include <gguf.h>
#ifdef __APPLE__
#include <ggml-metal.h>
#endif

#include <sstream>

namespace py = pybind11;

// Typed pointer wrapper - each Tag creates a distinct Python type
template<typename Tag>
struct TypedPtr {
    void* ptr;
    TypedPtr() : ptr(nullptr) {}
    explicit TypedPtr(void* p) : ptr(p) {}
};

// Tag types
struct ggml_context_tag {};
struct ggml_tensor_tag {};
struct ggml_cgraph_tag {};
struct ggml_backend_tag {};
struct ggml_backend_buffer_tag {};
struct ggml_backend_buffer_type_tag {};
struct ggml_gallocr_tag {};
struct gguf_context_tag {};

// Aliases
using ContextPtr = TypedPtr<ggml_context_tag>;
using TensorPtr = TypedPtr<ggml_tensor_tag>;
using GraphPtr = TypedPtr<ggml_cgraph_tag>;
using BackendPtr = TypedPtr<ggml_backend_tag>;
using BufferPtr = TypedPtr<ggml_backend_buffer_tag>;
using BufferTypePtr = TypedPtr<ggml_backend_buffer_type_tag>;
using GAllocrPtr = TypedPtr<ggml_gallocr_tag>;
using GGUFContextPtr = TypedPtr<gguf_context_tag>;

// Type extraction helper for high-frequency struct pointer types
template<typename T, typename Tag>
inline T* as(const TypedPtr<Tag>& p) { return static_cast<T*>(p.ptr); }

#define REGISTER_PTR(Alias, Name) \
    py::class_<Alias>(m, Name) \
        .def("__repr__", [](const Alias& self) { \
            std::ostringstream oss; \
            oss << "<ggml." Name " at 0x" \
                << std::hex << reinterpret_cast<uintptr_t>(self.ptr) << ">"; \
            return oss.str(); \
        }) \
        .def("__bool__", [](const Alias& self) { \
            return self.ptr != nullptr; \
        })

PYBIND11_MODULE(ggml, m) {
    m.doc() = "Simple and naive GGML binding";

    // Register typed pointer classes
    REGISTER_PTR(ContextPtr, "Context");
    REGISTER_PTR(TensorPtr, "Tensor");
    REGISTER_PTR(GraphPtr, "Graph");
    REGISTER_PTR(BackendPtr, "Backend");
    REGISTER_PTR(BufferPtr, "Buffer");
    REGISTER_PTR(BufferTypePtr, "BufferType");
    REGISTER_PTR(GAllocrPtr, "GAllocr");
    REGISTER_PTR(GGUFContextPtr, "GGUFContext");

    // Bind ggml_type enum
    py::enum_<ggml_type>(m, "Type")
        .value("F32", GGML_TYPE_F32)
        .value("F16", GGML_TYPE_F16)
        .value("Q4_0", GGML_TYPE_Q4_0)
        .value("Q4_1", GGML_TYPE_Q4_1)
        .value("Q5_0", GGML_TYPE_Q5_0)
        .value("Q5_1", GGML_TYPE_Q5_1)
        .value("Q8_0", GGML_TYPE_Q8_0)
        .value("Q8_1", GGML_TYPE_Q8_1)
        .value("Q2_K", GGML_TYPE_Q2_K)
        .value("Q3_K", GGML_TYPE_Q3_K)
        .value("Q4_K", GGML_TYPE_Q4_K)
        .value("Q5_K", GGML_TYPE_Q5_K)
        .value("Q6_K", GGML_TYPE_Q6_K)
        .value("Q8_K", GGML_TYPE_Q8_K)
        .value("IQ2_XXS", GGML_TYPE_IQ2_XXS)
        .value("IQ2_XS", GGML_TYPE_IQ2_XS)
        .value("IQ3_XXS", GGML_TYPE_IQ3_XXS)
        .value("IQ1_S", GGML_TYPE_IQ1_S)
        .value("IQ4_NL", GGML_TYPE_IQ4_NL)
        .value("IQ3_S", GGML_TYPE_IQ3_S)
        .value("IQ2_S", GGML_TYPE_IQ2_S)
        .value("IQ4_XS", GGML_TYPE_IQ4_XS)
        .value("I8", GGML_TYPE_I8)
        .value("I16", GGML_TYPE_I16)
        .value("I32", GGML_TYPE_I32)
        .value("I64", GGML_TYPE_I64)
        .value("F64", GGML_TYPE_F64)
        .value("IQ1_M", GGML_TYPE_IQ1_M)
        .export_values();

    // Bind ggml_status enum
    py::enum_<ggml_status>(m, "Status")
        .value("ALLOC_FAILED", GGML_STATUS_ALLOC_FAILED)
        .value("FAILED", GGML_STATUS_FAILED)
        .value("SUCCESS", GGML_STATUS_SUCCESS)
        .value("ABORTED", GGML_STATUS_ABORTED)
        .export_values();

    m.def("log_set_default", []() {  }, "");
    m.def("backend_cpu_init", []() { return BackendPtr(ggml_backend_cpu_init()); }, "Initialize CPU backend");
    m.def("backend_is_cpu", [](BackendPtr backend) { return ggml_backend_is_cpu(static_cast<ggml_backend_t>(backend.ptr)); }, "Check if backend is CPU backend", py::arg("backend"));
    m.def("backend_cpu_set_n_threads", [](BackendPtr backend, int n_threads) { ggml_backend_cpu_set_n_threads(static_cast<ggml_backend_t>(backend.ptr), n_threads); }, "Set number of threads for CPU backend", py::arg("backend"), py::arg("n_threads"));

#ifdef __APPLE__
    // Metal backend functions (macOS only)
    m.def("backend_metal_init", []() { return BackendPtr(ggml_backend_metal_init()); }, "Initialize Metal backend (macOS only)");
    m.def("backend_is_metal", [](BackendPtr backend) { return ggml_backend_is_metal(static_cast<ggml_backend_t>(backend.ptr)); }, "Check if backend is Metal backend", py::arg("backend"));
    m.def("backend_metal_supports_family", [](BackendPtr backend, int family) { return ggml_backend_metal_supports_family(static_cast<ggml_backend_t>(backend.ptr), family); }, "Check if Metal device supports specific feature family", py::arg("backend"), py::arg("family"));
    m.def("backend_metal_capture_next_compute", [](BackendPtr backend) { ggml_backend_metal_capture_next_compute(static_cast<ggml_backend_t>(backend.ptr)); }, "Capture next Metal compute for debugging", py::arg("backend"));
#endif
    m.def("tensor_overhead", &ggml_tensor_overhead, "Get the memory overhead of a tensor");
    m.def("graph_overhead", &ggml_graph_overhead, "Get the memory overhead of a graph");
    m.def("graph_overhead_custom", [](size_t size, bool grads) {
        return ggml_graph_overhead_custom(size, grads);
    }, "Get memory overhead for custom graph size", py::arg("size"), py::arg("grads"));
    m.def("DEFAULT_GRAPH_SIZE", []() { return GGML_DEFAULT_GRAPH_SIZE; }, "Default graph size constant");
    m.attr("FILE_MAGIC") = GGML_FILE_MAGIC;
    m.attr("QNT_VERSION_FACTOR") = GGML_QNT_VERSION_FACTOR;
    m.def("context_init", [](size_t mem_size, void* mem_buffer = nullptr, bool no_alloc = false) {
        ggml_init_params params = {mem_size, mem_buffer, no_alloc};
        return ContextPtr(ggml_init(params));
    }, "Initialize GGML context", py::arg("mem_size"), py::arg("mem_buffer") = nullptr, py::arg("no_alloc") = false);
    m.def("new_tensor_1d", [](ContextPtr ctx, ggml_type type, int64_t ne0) {
        return TensorPtr(ggml_new_tensor_1d(as<ggml_context>(ctx), type, ne0));
    }, "Create a new 1D tensor", py::arg("ctx"), py::arg("type"), py::arg("ne0"));
    m.def("new_tensor_2d", [](ContextPtr ctx, ggml_type type, int64_t ne0, int64_t ne1) {
        return TensorPtr(ggml_new_tensor_2d(as<ggml_context>(ctx), type, ne0, ne1));
    }, "Create a new 2D tensor", py::arg("ctx"), py::arg("type"), py::arg("ne0"), py::arg("ne1"));
    m.def("new_tensor_3d", [](ContextPtr ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
        return TensorPtr(ggml_new_tensor_3d(as<ggml_context>(ctx), type, ne0, ne1, ne2));
    }, "Create a new 3D tensor", py::arg("ctx"), py::arg("type"), py::arg("ne0"), py::arg("ne1"), py::arg("ne2"));
    m.def("new_tensor_4d", [](ContextPtr ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
        return TensorPtr(ggml_new_tensor_4d(as<ggml_context>(ctx), type, ne0, ne1, ne2, ne3));
    }, "Create a new 4D tensor", py::arg("ctx"), py::arg("type"), py::arg("ne0"), py::arg("ne1"), py::arg("ne2"), py::arg("ne3"));
    m.def("type_size", &ggml_type_size, "Get size in bytes for all elements in a block of the given type", py::arg("type"));
    m.def("blck_size", &ggml_blck_size, "Get block size (number of elements per block) for the given type", py::arg("type"));
    m.def("type_name", [](int type) {
        return std::string(ggml_type_name(static_cast<ggml_type>(type)));
    }, "Get type name string", py::arg("type"));
    m.def("ftype_to_ggml_type", [](int ftype) {
        return static_cast<int>(ggml_ftype_to_ggml_type(static_cast<ggml_ftype>(ftype)));
    }, "Convert file type to tensor type", py::arg("ftype"));
    m.def("backend_alloc_ctx_tensors", [](ContextPtr ctx, BackendPtr backend) {
        return BufferPtr(ggml_backend_alloc_ctx_tensors(as<ggml_context>(ctx), static_cast<ggml_backend_t>(backend.ptr)));
    }, "Allocate all tensors in a GGML context to a backend", py::arg("ctx"), py::arg("backend"));
    m.def("backend_buffer_free", [](BufferPtr buffer) {
        ggml_backend_buffer_free(static_cast<ggml_backend_buffer_t>(buffer.ptr));
    }, "Free backend buffer", py::arg("buffer"));
    m.def("backend_buffer_is_host", [](BufferPtr buffer) {
        return ggml_backend_buffer_is_host(static_cast<ggml_backend_buffer_t>(buffer.ptr));
    }, "Check if buffer is in host memory", py::arg("buffer"));
    m.def("backend_buffer_get_size", [](BufferPtr buffer) {
        return ggml_backend_buffer_get_size(static_cast<ggml_backend_buffer_t>(buffer.ptr));
    }, "Get buffer size in bytes", py::arg("buffer"));
    m.def("backend_tensor_set", [](TensorPtr tensor, py::buffer data, size_t offset, size_t size) {
        py::buffer_info info = data.request();
        size_t data_size = info.itemsize * info.size;
        if (size == 0) {
            size = data_size;
        }
        if (data_size < size) {
            throw std::runtime_error("Data buffer is too small");
        }
        ggml_backend_tensor_set(
            as<ggml_tensor>(tensor),
            static_cast<const void*>(info.ptr),
            offset,
            size
        );
    }, "Set tensor data from Python buffer", py::arg("tensor"), py::arg("data"), py::arg("offset") = 0, py::arg("size") = 0);
    m.def("new_graph", [](ContextPtr ctx) {
        return GraphPtr(ggml_new_graph(as<ggml_context>(ctx)));
    }, "Create a new computation graph", py::arg("ctx"));
    m.def("new_graph_custom", [](ContextPtr ctx, size_t size, bool grads) {
        return GraphPtr(ggml_new_graph_custom(as<ggml_context>(ctx), size, grads));
    }, "Create a new computation graph with custom size and gradient settings", py::arg("ctx"), py::arg("size"), py::arg("grads"));
    m.def("mul_mat", [](ContextPtr ctx, TensorPtr a, TensorPtr b) {
        return TensorPtr(ggml_mul_mat(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b)));
    }, "Matrix multiplication: result = a * b", py::arg("ctx"), py::arg("a"), py::arg("b"));

    // Binary operations
    m.def("add", [](ContextPtr ctx, TensorPtr a, TensorPtr b) {
        return TensorPtr(ggml_add(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b)));
    }, "Element-wise addition: result = a + b", py::arg("ctx"), py::arg("a"), py::arg("b"));

    m.def("sub", [](ContextPtr ctx, TensorPtr a, TensorPtr b) {
        return TensorPtr(ggml_sub(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b)));
    }, "Element-wise subtraction: result = a - b", py::arg("ctx"), py::arg("a"), py::arg("b"));

    m.def("mul", [](ContextPtr ctx, TensorPtr a, TensorPtr b) {
        return TensorPtr(ggml_mul(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b)));
    }, "Element-wise multiplication: result = a * b", py::arg("ctx"), py::arg("a"), py::arg("b"));

    m.def("div", [](ContextPtr ctx, TensorPtr a, TensorPtr b) {
        return TensorPtr(ggml_div(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b)));
    }, "Element-wise division: result = a / b", py::arg("ctx"), py::arg("a"), py::arg("b"));

    // Additional binary operations
    m.def("add1", [](ContextPtr ctx, TensorPtr a, TensorPtr b) {
        return TensorPtr(ggml_add1(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b)));
    }, "Add scalar b to each row of matrix a", py::arg("ctx"), py::arg("a"), py::arg("b"));

    m.def("out_prod", [](ContextPtr ctx, TensorPtr a, TensorPtr b) {
        return TensorPtr(ggml_out_prod(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b)));
    }, "Outer product: result = a @ b^T", py::arg("ctx"), py::arg("a"), py::arg("b"));

    m.def("concat", [](ContextPtr ctx, TensorPtr a, TensorPtr b, int dim) {
        return TensorPtr(ggml_concat(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b), dim));
    }, "Concatenate tensors a and b along dimension dim", py::arg("ctx"), py::arg("a"), py::arg("b"), py::arg("dim"));

    m.def("count_equal", [](ContextPtr ctx, TensorPtr a, TensorPtr b) {
        return TensorPtr(ggml_count_equal(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b)));
    }, "Count number of equal elements in a and b", py::arg("ctx"), py::arg("a"), py::arg("b"));

    // Unary operations
    m.def("abs", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_abs(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Element-wise absolute value: result = |a|", py::arg("ctx"), py::arg("a"));

    m.def("neg", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_neg(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Element-wise negation: result = -a", py::arg("ctx"), py::arg("a"));

    m.def("sqrt", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_sqrt(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Element-wise square root: result = sqrt(a)", py::arg("ctx"), py::arg("a"));

    m.def("sqr", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_sqr(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Element-wise square: result = a^2", py::arg("ctx"), py::arg("a"));

    // Additional unary operations
    m.def("log", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_log(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Element-wise natural logarithm: result = log(a)", py::arg("ctx"), py::arg("a"));

    m.def("exp", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_exp(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Element-wise exponential: result = exp(a)", py::arg("ctx"), py::arg("a"));

    m.def("tanh", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_tanh(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Element-wise hyperbolic tangent: result = tanh(a)", py::arg("ctx"), py::arg("a"));

    m.def("sigmoid", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_sigmoid(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Element-wise sigmoid activation: result = 1 / (1 + exp(-a))", py::arg("ctx"), py::arg("a"));

    m.def("relu", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_relu(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Element-wise ReLU activation: result = max(0, a)", py::arg("ctx"), py::arg("a"));

    m.def("gelu", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_gelu(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Element-wise GELU activation (Gaussian Error Linear Unit)", py::arg("ctx"), py::arg("a"));

    // Reduction operations
    m.def("sum", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_sum(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Sum all elements in tensor, returns scalar", py::arg("ctx"), py::arg("a"));

    m.def("mean", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_mean(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Mean of all elements along rows", py::arg("ctx"), py::arg("a"));

    // Normalization and activation
    m.def("norm", [](ContextPtr ctx, TensorPtr a, float eps) {
        return TensorPtr(ggml_norm(as<ggml_context>(ctx), as<ggml_tensor>(a), eps));
    }, "Normalize tensor along rows", py::arg("ctx"), py::arg("a"), py::arg("eps"));

    m.def("soft_max", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_soft_max(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Apply softmax activation", py::arg("ctx"), py::arg("a"));

    // Reshape operations
    m.def("reshape_2d", [](ContextPtr ctx, TensorPtr a, int64_t ne0, int64_t ne1) {
        return TensorPtr(ggml_reshape_2d(as<ggml_context>(ctx), as<ggml_tensor>(a), ne0, ne1));
    }, "Reshape tensor to 2D", py::arg("ctx"), py::arg("a"), py::arg("ne0"), py::arg("ne1"));

    m.def("reshape_3d", [](ContextPtr ctx, TensorPtr a, int64_t ne0, int64_t ne1, int64_t ne2) {
        return TensorPtr(ggml_reshape_3d(as<ggml_context>(ctx), as<ggml_tensor>(a), ne0, ne1, ne2));
    }, "Reshape tensor to 3D", py::arg("ctx"), py::arg("a"), py::arg("ne0"), py::arg("ne1"), py::arg("ne2"));

    // Transpose and contiguous
    m.def("transpose", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_transpose(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Transpose tensor (permute 1,0,2,3)", py::arg("ctx"), py::arg("a"));

    m.def("cont", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_cont(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Make tensor contiguous in memory", py::arg("ctx"), py::arg("a"));

    m.def("cont_2d", [](ContextPtr ctx, TensorPtr a, int64_t ne0, int64_t ne1) {
        return TensorPtr(ggml_cont_2d(as<ggml_context>(ctx), as<ggml_tensor>(a), ne0, ne1));
    }, "Make tensor contiguous with 2D shape", py::arg("ctx"), py::arg("a"), py::arg("ne0"), py::arg("ne1"));

    m.def("cont_3d", [](ContextPtr ctx, TensorPtr a, int64_t ne0, int64_t ne1, int64_t ne2) {
        return TensorPtr(ggml_cont_3d(as<ggml_context>(ctx), as<ggml_tensor>(a), ne0, ne1, ne2));
    }, "Make tensor contiguous with 3D shape", py::arg("ctx"), py::arg("a"), py::arg("ne0"), py::arg("ne1"), py::arg("ne2"));

    // View operations (for KV cache and Q/K/V split)
    m.def("view_1d", [](ContextPtr ctx, TensorPtr a, int64_t ne0, size_t offset) {
        return TensorPtr(ggml_view_1d(as<ggml_context>(ctx), as<ggml_tensor>(a), ne0, offset));
    }, "Create 1D view of tensor", py::arg("ctx"), py::arg("a"), py::arg("ne0"), py::arg("offset"));

    m.def("view_2d", [](ContextPtr ctx, TensorPtr a, int64_t ne0, int64_t ne1, size_t nb1, size_t offset) {
        return TensorPtr(ggml_view_2d(as<ggml_context>(ctx), as<ggml_tensor>(a), ne0, ne1, nb1, offset));
    }, "Create 2D view of tensor", py::arg("ctx"), py::arg("a"), py::arg("ne0"), py::arg("ne1"), py::arg("nb1"), py::arg("offset"));

    // Embedding lookup
    m.def("get_rows", [](ContextPtr ctx, TensorPtr a, TensorPtr b) {
        return TensorPtr(ggml_get_rows(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b)));
    }, "Get rows by index (a: data, b: indices)", py::arg("ctx"), py::arg("a"), py::arg("b"));

    // Permute dimensions (for attention)
    m.def("permute", [](ContextPtr ctx, TensorPtr a, int axis0, int axis1, int axis2, int axis3) {
        return TensorPtr(ggml_permute(as<ggml_context>(ctx), as<ggml_tensor>(a), axis0, axis1, axis2, axis3));
    }, "Permute tensor dimensions", py::arg("ctx"), py::arg("a"), py::arg("axis0"), py::arg("axis1"), py::arg("axis2"), py::arg("axis3"));

    // Scale operation (for attention scaling)
    m.def("scale", [](ContextPtr ctx, TensorPtr a, float s) {
        return TensorPtr(ggml_scale(as<ggml_context>(ctx), as<ggml_tensor>(a), s));
    }, "Scale tensor by scalar", py::arg("ctx"), py::arg("a"), py::arg("s"));

    // Diagonal mask (for causal attention)
    m.def("diag_mask_inf", [](ContextPtr ctx, TensorPtr a, int n_past) {
        return TensorPtr(ggml_diag_mask_inf(as<ggml_context>(ctx), as<ggml_tensor>(a), n_past));
    }, "Apply diagonal mask with -inf for causal attention", py::arg("ctx"), py::arg("a"), py::arg("n_past"));

    // Copy tensor (for KV cache storage)
    m.def("cpy", [](ContextPtr ctx, TensorPtr a, TensorPtr b) {
        return TensorPtr(ggml_cpy(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b)));
    }, "Copy tensor a to b", py::arg("ctx"), py::arg("a"), py::arg("b"));

    // Element size helper
    m.def("element_size", [](TensorPtr tensor) {
        return ggml_element_size(as<ggml_tensor>(tensor));
    }, "Get element size in bytes", py::arg("tensor"));

    m.def("build_forward_expand", [](GraphPtr cgraph, TensorPtr tensor) {
        ggml_build_forward_expand(as<ggml_cgraph>(cgraph), as<ggml_tensor>(tensor));
    }, "Build forward computation graph from tensor", py::arg("cgraph"), py::arg("tensor"));
    m.def("graph_compute_with_ctx", [](ContextPtr ctx, GraphPtr cgraph, int n_threads) {
        return ggml_graph_compute_with_ctx(as<ggml_context>(ctx), as<ggml_cgraph>(cgraph), n_threads);
    }, "Compute the graph with given context and thread count", py::arg("ctx"), py::arg("cgraph"), py::arg("n_threads") = 1);
    m.def("context_free", [](ContextPtr ctx) {
        ggml_free(as<ggml_context>(ctx));
    }, "Free GGML context and all its allocated memory", py::arg("ctx"));
    m.def("get_data", [](TensorPtr tensor) {
        return reinterpret_cast<uintptr_t>(ggml_get_data(as<ggml_tensor>(tensor)));
    }, "Get data pointer from tensor (as uintptr_t)", py::arg("tensor"));
    m.def("get_data_f32", [](TensorPtr tensor) {
        return reinterpret_cast<uintptr_t>(ggml_get_data_f32(as<ggml_tensor>(tensor)));
    }, "Get float32 data pointer from tensor (as uintptr_t)", py::arg("tensor"));
    m.def("time_init", []() {
        ggml_time_init();
    }, "Initialize time measurement - call this once at the beginning of the program");

    m.def("time_us", []() {
        return ggml_time_us();
    }, "Get current time in microseconds");

    // Graph allocator functions
    m.def("gallocr_new", [](BufferTypePtr buffer_type) {
        return GAllocrPtr(ggml_gallocr_new(static_cast<ggml_backend_buffer_type_t>(buffer_type.ptr)));
    }, "Create new graph allocator", py::arg("buffer_type"));

    m.def("gallocr_reserve", [](GAllocrPtr allocr, GraphPtr cgraph) {
        return ggml_gallocr_reserve(static_cast<ggml_gallocr_t>(allocr.ptr), as<ggml_cgraph>(cgraph));
    }, " reserve memory for graph computation", py::arg("allocr"), py::arg("cgraph"));

    m.def("gallocr_alloc_graph", [](GAllocrPtr allocr, GraphPtr cgraph) {
        return ggml_gallocr_alloc_graph(static_cast<ggml_gallocr_t>(allocr.ptr), as<ggml_cgraph>(cgraph));
    }, "Allocate tensors for graph computation", py::arg("allocr"), py::arg("cgraph"));

    m.def("gallocr_get_buffer_size", [](GAllocrPtr allocr, int buffer_index) {
        return ggml_gallocr_get_buffer_size(static_cast<ggml_gallocr_t>(allocr.ptr), buffer_index);
    }, "Get buffer size for allocated graph", py::arg("allocr"), py::arg("buffer_index") = 0);

    m.def("gallocr_free", [](GAllocrPtr allocr) {
        ggml_gallocr_free(static_cast<ggml_gallocr_t>(allocr.ptr));
    }, "Free graph allocator", py::arg("allocr"));

    // Backend buffer type
    m.def("backend_get_default_buffer_type", [](BackendPtr backend) {
        return BufferTypePtr(ggml_backend_get_default_buffer_type(static_cast<ggml_backend_t>(backend.ptr)));
    }, "Get default buffer type for backend", py::arg("backend"));

    // Graph node access
    m.def("graph_node", [](GraphPtr cgraph, int i) {
        return TensorPtr(ggml_graph_node(as<ggml_cgraph>(cgraph), i));
    }, "Get node from graph by index", py::arg("cgraph"), py::arg("i"));

    m.def("graph_get_tensor", [](GraphPtr cgraph, const char* name) {
        return TensorPtr(ggml_graph_get_tensor(as<ggml_cgraph>(cgraph), name));
    }, "Get tensor from graph by name", py::arg("cgraph"), py::arg("name"));

    // Backend graph computation
    m.def("backend_graph_compute", [](BackendPtr backend, GraphPtr cgraph) {
        ggml_backend_graph_compute(static_cast<ggml_backend_t>(backend.ptr), as<ggml_cgraph>(cgraph));
    }, "Compute graph using backend", py::arg("backend"), py::arg("cgraph"));

    // Backend tensor get
    m.def("backend_tensor_get", [](TensorPtr tensor, py::buffer data, size_t offset, size_t size) {
        py::buffer_info info = data.request();
        size_t data_size = info.itemsize * info.size;
        if (size == 0) {
            size = data_size;
        }
        if (data_size < size) {
            throw std::runtime_error("Data buffer is too small");
        }
        ggml_backend_tensor_get(
            as<ggml_tensor>(tensor),
            static_cast<void*>(info.ptr),
            offset,
            size
        );
    }, "Get tensor data to Python buffer", py::arg("tensor"), py::arg("data"), py::arg("offset") = 0, py::arg("size") = 0);

    // Backend management
    m.def("backend_free", [](BackendPtr backend) {
        ggml_backend_free(static_cast<ggml_backend_t>(backend.ptr));
    }, "Free backend", py::arg("backend"));

    // Tensor helper functions
    m.def("tensor_ne", [](TensorPtr tensor, int dim) {
        return as<ggml_tensor>(tensor)->ne[dim];
    }, "Get tensor dimension size", py::arg("tensor"), py::arg("dim"));

    m.def("tensor_nb", [](TensorPtr tensor, int dim) {
        return as<ggml_tensor>(tensor)->nb[dim];
    }, "Get tensor stride in bytes", py::arg("tensor"), py::arg("dim"));

    m.def("tensor_type", [](TensorPtr tensor) {
        return static_cast<ggml_type>(as<ggml_tensor>(tensor)->type);
    }, "Get tensor type", py::arg("tensor"));

    m.def("nbytes", [](TensorPtr tensor) {
        return ggml_nbytes(as<ggml_tensor>(tensor));
    }, "Get tensor size in bytes", py::arg("tensor"));

    m.def("nelements", [](TensorPtr tensor) {
        return ggml_nelements(as<ggml_tensor>(tensor));
    }, "Get number of elements in tensor", py::arg("tensor"));

    // Graph input/output marking
    m.def("set_input", [](TensorPtr tensor) {
        ggml_set_input(as<ggml_tensor>(tensor));
    }, "Mark tensor as graph input (allocated at graph start in non-overlapping addresses)", py::arg("tensor"));

    m.def("set_output", [](TensorPtr tensor) {
        ggml_set_output(as<ggml_tensor>(tensor));
    }, "Mark tensor as graph output (never freed or overwritten during computation)", py::arg("tensor"));

    m.def("get_tensor", [](ContextPtr ctx, const char* name) {
        return TensorPtr(ggml_get_tensor(as<ggml_context>(ctx), name));
    }, "Get tensor from context by name", py::arg("ctx"), py::arg("name"));

    m.def("set_name", [](TensorPtr tensor, const char* name) {
        return TensorPtr(ggml_set_name(as<ggml_tensor>(tensor), name));
    }, "Set tensor name", py::arg("tensor"), py::arg("name"));

    // Bind ggml_op_pool enum
    py::enum_<ggml_op_pool>(m, "OpPool")
        .value("MAX", GGML_OP_POOL_MAX)
        .value("AVG", GGML_OP_POOL_AVG)
        .value("COUNT", GGML_OP_POOL_COUNT)
        .export_values();

    // Pooling operations
    m.def("pool_1d", [](ContextPtr ctx, TensorPtr a, ggml_op_pool op, int k0, int s0, int p0) {
        return TensorPtr(ggml_pool_1d(as<ggml_context>(ctx), as<ggml_tensor>(a), op, k0, s0, p0));
    }, "1D pooling (max or average)", py::arg("ctx"), py::arg("a"), py::arg("op"), py::arg("k0"), py::arg("s0"), py::arg("p0"));

    // GGUF functions
    m.def("gguf_init_from_file", [](const char* fname, bool no_alloc) {
        struct ggml_context* meta_ctx = nullptr;
        struct gguf_init_params params = { no_alloc, &meta_ctx };
        struct gguf_context* gguf_ctx = gguf_init_from_file(fname, params);
        return py::make_tuple(GGUFContextPtr(gguf_ctx), ContextPtr(meta_ctx));
    }, "Initialize GGUF context from file, returns (gguf_ctx, meta_ctx) tuple", py::arg("fname"), py::arg("no_alloc") = true);

    m.def("gguf_free", [](GGUFContextPtr ctx) {
        gguf_free(static_cast<struct gguf_context*>(ctx.ptr));
    }, "Free GGUF context", py::arg("ctx"));

    m.def("gguf_get_n_tensors", [](GGUFContextPtr ctx) {
        return gguf_get_n_tensors(static_cast<const struct gguf_context*>(ctx.ptr));
    }, "Get number of tensors in GGUF file", py::arg("ctx"));

    m.def("gguf_get_data_offset", [](GGUFContextPtr ctx) {
        return gguf_get_data_offset(static_cast<const struct gguf_context*>(ctx.ptr));
    }, "Get data offset in GGUF file", py::arg("ctx"));

    m.def("gguf_get_tensor_offset", [](GGUFContextPtr ctx, int64_t tensor_id) {
        return gguf_get_tensor_offset(static_cast<const struct gguf_context*>(ctx.ptr), tensor_id);
    }, "Get tensor offset in GGUF file", py::arg("ctx"), py::arg("tensor_id"));

    m.def("gguf_get_tensor_name", [](GGUFContextPtr ctx, int64_t tensor_id) {
        return gguf_get_tensor_name(static_cast<const struct gguf_context*>(ctx.ptr), tensor_id);
    }, "Get tensor name by index", py::arg("ctx"), py::arg("tensor_id"));
}

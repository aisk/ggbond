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
#include <stdexcept>
#include <string>

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

inline void* require_non_null(void* ptr, const char* api_name) {
    if (ptr == nullptr) {
        PyErr_Format(PyExc_MemoryError, "%s returned nullptr", api_name);
        throw py::error_already_set();
    }
    return ptr;
}

inline void throw_if_compute_failed(ggml_status status, const char* api_name) {
    if (status != GGML_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << api_name << " failed: " << ggml_status_to_string(status)
            << " (status=" << static_cast<int>(status) << ")";
        throw std::runtime_error(oss.str());
    }
}

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
        return ContextPtr(require_non_null(ggml_init(params), "ggml_init"));
    }, "Initialize GGML context", py::arg("mem_size"), py::arg("mem_buffer") = nullptr, py::arg("no_alloc") = false);
    m.def("new_tensor_1d", [](ContextPtr ctx, ggml_type type, int64_t ne0) {
        return TensorPtr(require_non_null(ggml_new_tensor_1d(as<ggml_context>(ctx), type, ne0), "ggml_new_tensor_1d"));
    }, "Create a new 1D tensor", py::arg("ctx"), py::arg("type"), py::arg("ne0"));
    m.def("new_tensor_2d", [](ContextPtr ctx, ggml_type type, int64_t ne0, int64_t ne1) {
        return TensorPtr(require_non_null(ggml_new_tensor_2d(as<ggml_context>(ctx), type, ne0, ne1), "ggml_new_tensor_2d"));
    }, "Create a new 2D tensor", py::arg("ctx"), py::arg("type"), py::arg("ne0"), py::arg("ne1"));
    m.def("new_tensor_3d", [](ContextPtr ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
        return TensorPtr(require_non_null(ggml_new_tensor_3d(as<ggml_context>(ctx), type, ne0, ne1, ne2), "ggml_new_tensor_3d"));
    }, "Create a new 3D tensor", py::arg("ctx"), py::arg("type"), py::arg("ne0"), py::arg("ne1"), py::arg("ne2"));
    m.def("new_tensor_4d", [](ContextPtr ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
        return TensorPtr(require_non_null(ggml_new_tensor_4d(as<ggml_context>(ctx), type, ne0, ne1, ne2, ne3), "ggml_new_tensor_4d"));
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

    m.def("sin", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_sin(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Element-wise sine", py::arg("ctx"), py::arg("a"));

    m.def("cos", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_cos(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Element-wise cosine", py::arg("ctx"), py::arg("a"));

    m.def("sgn", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_sgn(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Element-wise sign function", py::arg("ctx"), py::arg("a"));

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

    m.def("silu", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_silu(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "SiLU activation (x * sigmoid(x))", py::arg("ctx"), py::arg("a"));

    m.def("leaky_relu", [](ContextPtr ctx, TensorPtr a, float negative_slope, bool inplace) {
        return TensorPtr(ggml_leaky_relu(as<ggml_context>(ctx), as<ggml_tensor>(a), negative_slope, inplace));
    }, "Leaky ReLU activation", py::arg("ctx"), py::arg("a"), py::arg("negative_slope"), py::arg("inplace"));

    m.def("gelu_quick", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_gelu_quick(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Fast GELU approximation", py::arg("ctx"), py::arg("a"));

    m.def("hardswish", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_hardswish(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Hard swish activation", py::arg("ctx"), py::arg("a"));

    m.def("hardsigmoid", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_hardsigmoid(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Hard sigmoid activation", py::arg("ctx"), py::arg("a"));

    m.def("elu", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_elu(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "ELU activation", py::arg("ctx"), py::arg("a"));

    // Reduction operations
    m.def("sum", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_sum(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Sum all elements in tensor, returns scalar", py::arg("ctx"), py::arg("a"));

    m.def("sum_rows", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_sum_rows(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Sum elements along rows", py::arg("ctx"), py::arg("a"));

    m.def("mean", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_mean(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Mean of all elements along rows", py::arg("ctx"), py::arg("a"));

    m.def("argmax", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_argmax(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Argmax along rows", py::arg("ctx"), py::arg("a"));

    // Normalization and activation
    m.def("norm", [](ContextPtr ctx, TensorPtr a, float eps) {
        return TensorPtr(ggml_norm(as<ggml_context>(ctx), as<ggml_tensor>(a), eps));
    }, "Normalize tensor along rows", py::arg("ctx"), py::arg("a"), py::arg("eps"));

    m.def("rms_norm", [](ContextPtr ctx, TensorPtr a, float eps) {
        return TensorPtr(ggml_rms_norm(as<ggml_context>(ctx), as<ggml_tensor>(a), eps));
    }, "RMS normalization", py::arg("ctx"), py::arg("a"), py::arg("eps"));

    m.def("group_norm", [](ContextPtr ctx, TensorPtr a, int n_groups, float eps) {
        return TensorPtr(ggml_group_norm(as<ggml_context>(ctx), as<ggml_tensor>(a), n_groups, eps));
    }, "Group normalization", py::arg("ctx"), py::arg("a"), py::arg("n_groups"), py::arg("eps"));

    m.def("l2_norm", [](ContextPtr ctx, TensorPtr a, float eps) {
        return TensorPtr(ggml_l2_norm(as<ggml_context>(ctx), as<ggml_tensor>(a), eps));
    }, "L2 normalization", py::arg("ctx"), py::arg("a"), py::arg("eps"));

    m.def("soft_max", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_soft_max(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Apply softmax activation", py::arg("ctx"), py::arg("a"));

    // Reshape operations
    m.def("reshape_1d", [](ContextPtr ctx, TensorPtr a, int64_t ne0) {
        return TensorPtr(ggml_reshape_1d(as<ggml_context>(ctx), as<ggml_tensor>(a), ne0));
    }, "Reshape tensor to 1D", py::arg("ctx"), py::arg("a"), py::arg("ne0"));

    m.def("reshape_2d", [](ContextPtr ctx, TensorPtr a, int64_t ne0, int64_t ne1) {
        return TensorPtr(ggml_reshape_2d(as<ggml_context>(ctx), as<ggml_tensor>(a), ne0, ne1));
    }, "Reshape tensor to 2D", py::arg("ctx"), py::arg("a"), py::arg("ne0"), py::arg("ne1"));

    m.def("reshape_3d", [](ContextPtr ctx, TensorPtr a, int64_t ne0, int64_t ne1, int64_t ne2) {
        return TensorPtr(ggml_reshape_3d(as<ggml_context>(ctx), as<ggml_tensor>(a), ne0, ne1, ne2));
    }, "Reshape tensor to 3D", py::arg("ctx"), py::arg("a"), py::arg("ne0"), py::arg("ne1"), py::arg("ne2"));

    m.def("reshape_4d", [](ContextPtr ctx, TensorPtr a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
        return TensorPtr(ggml_reshape_4d(as<ggml_context>(ctx), as<ggml_tensor>(a), ne0, ne1, ne2, ne3));
    }, "Reshape tensor to 4D", py::arg("ctx"), py::arg("a"), py::arg("ne0"), py::arg("ne1"), py::arg("ne2"), py::arg("ne3"));

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

    m.def("cont_1d", [](ContextPtr ctx, TensorPtr a, int64_t ne0) {
        return TensorPtr(ggml_cont_1d(as<ggml_context>(ctx), as<ggml_tensor>(a), ne0));
    }, "Make tensor contiguous with 1D shape", py::arg("ctx"), py::arg("a"), py::arg("ne0"));

    m.def("cont_3d", [](ContextPtr ctx, TensorPtr a, int64_t ne0, int64_t ne1, int64_t ne2) {
        return TensorPtr(ggml_cont_3d(as<ggml_context>(ctx), as<ggml_tensor>(a), ne0, ne1, ne2));
    }, "Make tensor contiguous with 3D shape", py::arg("ctx"), py::arg("a"), py::arg("ne0"), py::arg("ne1"), py::arg("ne2"));

    m.def("cont_4d", [](ContextPtr ctx, TensorPtr a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
        return TensorPtr(ggml_cont_4d(as<ggml_context>(ctx), as<ggml_tensor>(a), ne0, ne1, ne2, ne3));
    }, "Make tensor contiguous with 4D shape", py::arg("ctx"), py::arg("a"), py::arg("ne0"), py::arg("ne1"), py::arg("ne2"), py::arg("ne3"));

    // View operations (for KV cache and Q/K/V split)
    m.def("view_1d", [](ContextPtr ctx, TensorPtr a, int64_t ne0, size_t offset) {
        return TensorPtr(ggml_view_1d(as<ggml_context>(ctx), as<ggml_tensor>(a), ne0, offset));
    }, "Create 1D view of tensor", py::arg("ctx"), py::arg("a"), py::arg("ne0"), py::arg("offset"));

    m.def("view_2d", [](ContextPtr ctx, TensorPtr a, int64_t ne0, int64_t ne1, size_t nb1, size_t offset) {
        return TensorPtr(ggml_view_2d(as<ggml_context>(ctx), as<ggml_tensor>(a), ne0, ne1, nb1, offset));
    }, "Create 2D view of tensor", py::arg("ctx"), py::arg("a"), py::arg("ne0"), py::arg("ne1"), py::arg("nb1"), py::arg("offset"));

    m.def("view_3d", [](ContextPtr ctx, TensorPtr a, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2, size_t offset) {
        return TensorPtr(ggml_view_3d(as<ggml_context>(ctx), as<ggml_tensor>(a), ne0, ne1, ne2, nb1, nb2, offset));
    }, "Create 3D view of tensor", py::arg("ctx"), py::arg("a"), py::arg("ne0"), py::arg("ne1"), py::arg("ne2"), py::arg("nb1"), py::arg("nb2"), py::arg("offset"));

    m.def("view_4d", [](ContextPtr ctx, TensorPtr a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2, size_t nb3, size_t offset) {
        return TensorPtr(ggml_view_4d(as<ggml_context>(ctx), as<ggml_tensor>(a), ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset));
    }, "Create 4D view of tensor", py::arg("ctx"), py::arg("a"), py::arg("ne0"), py::arg("ne1"), py::arg("ne2"), py::arg("ne3"), py::arg("nb1"), py::arg("nb2"), py::arg("nb3"), py::arg("offset"));

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

    m.def("clamp", [](ContextPtr ctx, TensorPtr a, float min, float max) {
        return TensorPtr(ggml_clamp(as<ggml_context>(ctx), as<ggml_tensor>(a), min, max));
    }, "Clamp tensor values to [min, max]", py::arg("ctx"), py::arg("a"), py::arg("min"), py::arg("max"));

    m.def("repeat", [](ContextPtr ctx, TensorPtr a, TensorPtr b) {
        return TensorPtr(ggml_repeat(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b)));
    }, "Repeat tensor a to match shape of b", py::arg("ctx"), py::arg("a"), py::arg("b"));

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
        const ggml_status status = ggml_backend_graph_compute(
            static_cast<ggml_backend_t>(backend.ptr),
            as<ggml_cgraph>(cgraph)
        );
        throw_if_compute_failed(status, "ggml_backend_graph_compute");
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

    m.def("n_dims", [](TensorPtr tensor) {
        return ggml_n_dims(as<ggml_tensor>(tensor));
    }, "Get number of dimensions", py::arg("tensor"));

    m.def("nrows", [](TensorPtr tensor) {
        return ggml_nrows(as<ggml_tensor>(tensor));
    }, "Get number of rows", py::arg("tensor"));

    m.def("get_name", [](TensorPtr tensor) {
        return std::string(ggml_get_name(as<ggml_tensor>(tensor)));
    }, "Get tensor name", py::arg("tensor"));

    m.def("is_contiguous", [](TensorPtr tensor) {
        return ggml_is_contiguous(as<ggml_tensor>(tensor));
    }, "Check if tensor is contiguous", py::arg("tensor"));

    m.def("is_transposed", [](TensorPtr tensor) {
        return ggml_is_transposed(as<ggml_tensor>(tensor));
    }, "Check if tensor is transposed", py::arg("tensor"));

    m.def("is_permuted", [](TensorPtr tensor) {
        return ggml_is_permuted(as<ggml_tensor>(tensor));
    }, "Check if tensor is permuted", py::arg("tensor"));

    m.def("is_quantized", [](ggml_type type) {
        return ggml_is_quantized(type);
    }, "Check if type is quantized", py::arg("type"));

    m.def("are_same_shape", [](TensorPtr a, TensorPtr b) {
        return ggml_are_same_shape(as<ggml_tensor>(a), as<ggml_tensor>(b));
    }, "Check if two tensors have the same shape", py::arg("a"), py::arg("b"));

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

    // Bind ggml_sort_order enum
    py::enum_<ggml_sort_order>(m, "SortOrder")
        .value("ASC", GGML_SORT_ORDER_ASC)
        .value("DESC", GGML_SORT_ORDER_DESC)
        .export_values();

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

    m.def("pool_2d", [](ContextPtr ctx, TensorPtr a, ggml_op_pool op, int k0, int k1, int s0, int s1, float p0, float p1) {
        return TensorPtr(ggml_pool_2d(as<ggml_context>(ctx), as<ggml_tensor>(a), op, k0, k1, s0, s1, p0, p1));
    }, "2D pooling (max or average)", py::arg("ctx"), py::arg("a"), py::arg("op"), py::arg("k0"), py::arg("k1"), py::arg("s0"), py::arg("s1"), py::arg("p0"), py::arg("p1"));

    // Attention-related operations
    m.def("soft_max_ext", [](ContextPtr ctx, TensorPtr a, TensorPtr mask, float scale, float max_bias) {
        return TensorPtr(ggml_soft_max_ext(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(mask), scale, max_bias));
    }, "Softmax with mask, scale and ALiBi bias", py::arg("ctx"), py::arg("a"), py::arg("mask"), py::arg("scale"), py::arg("max_bias"));

    m.def("diag_mask_zero", [](ContextPtr ctx, TensorPtr a, int n_past) {
        return TensorPtr(ggml_diag_mask_zero(as<ggml_context>(ctx), as<ggml_tensor>(a), n_past));
    }, "Set elements above the diagonal to 0", py::arg("ctx"), py::arg("a"), py::arg("n_past"));

    m.def("rope", [](ContextPtr ctx, TensorPtr a, TensorPtr b, int n_dims, int mode) {
        return TensorPtr(ggml_rope(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b), n_dims, mode));
    }, "Rotary position embedding", py::arg("ctx"), py::arg("a"), py::arg("b"), py::arg("n_dims"), py::arg("mode"));

    m.def("rope_ext", [](ContextPtr ctx, TensorPtr a, TensorPtr b, TensorPtr c, int n_dims, int mode, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
        return TensorPtr(ggml_rope_ext(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b), as<ggml_tensor>(c), n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow));
    }, "Extended rotary position embedding", py::arg("ctx"), py::arg("a"), py::arg("b"), py::arg("c"), py::arg("n_dims"), py::arg("mode"), py::arg("n_ctx_orig"), py::arg("freq_base"), py::arg("freq_scale"), py::arg("ext_factor"), py::arg("attn_factor"), py::arg("beta_fast"), py::arg("beta_slow"));

    m.def("flash_attn_ext", [](ContextPtr ctx, TensorPtr q, TensorPtr k, TensorPtr v, TensorPtr mask, float scale, float max_bias, float logit_softcap) {
        return TensorPtr(ggml_flash_attn_ext(as<ggml_context>(ctx), as<ggml_tensor>(q), as<ggml_tensor>(k), as<ggml_tensor>(v), as<ggml_tensor>(mask), scale, max_bias, logit_softcap));
    }, "Flash Attention", py::arg("ctx"), py::arg("q"), py::arg("k"), py::arg("v"), py::arg("mask"), py::arg("scale"), py::arg("max_bias"), py::arg("logit_softcap"));

    // Convolution operations
    m.def("conv_1d", [](ContextPtr ctx, TensorPtr a, TensorPtr b, int s0, int p0, int d0) {
        return TensorPtr(ggml_conv_1d(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b), s0, p0, d0));
    }, "1D convolution", py::arg("ctx"), py::arg("a"), py::arg("b"), py::arg("s0"), py::arg("p0"), py::arg("d0"));

    m.def("conv_2d", [](ContextPtr ctx, TensorPtr a, TensorPtr b, int s0, int s1, int p0, int p1, int d0, int d1) {
        return TensorPtr(ggml_conv_2d(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b), s0, s1, p0, p1, d0, d1));
    }, "2D convolution", py::arg("ctx"), py::arg("a"), py::arg("b"), py::arg("s0"), py::arg("s1"), py::arg("p0"), py::arg("p1"), py::arg("d0"), py::arg("d1"));

    // Tensor operations
    m.def("dup", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_dup(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Duplicate tensor", py::arg("ctx"), py::arg("a"));

    m.def("dup_tensor", [](ContextPtr ctx, TensorPtr src) {
        return TensorPtr(ggml_dup_tensor(as<ggml_context>(ctx), as<ggml_tensor>(src)));
    }, "Duplicate tensor structure (without data)", py::arg("ctx"), py::arg("src"));

    m.def("view_tensor", [](ContextPtr ctx, TensorPtr src) {
        return TensorPtr(ggml_view_tensor(as<ggml_context>(ctx), as<ggml_tensor>(src)));
    }, "Create a view of tensor", py::arg("ctx"), py::arg("src"));

    m.def("cast", [](ContextPtr ctx, TensorPtr a, ggml_type type) {
        return TensorPtr(ggml_cast(as<ggml_context>(ctx), as<ggml_tensor>(a), type));
    }, "Cast tensor to different type", py::arg("ctx"), py::arg("a"), py::arg("type"));

    m.def("pad", [](ContextPtr ctx, TensorPtr a, int p0, int p1, int p2, int p3) {
        return TensorPtr(ggml_pad(as<ggml_context>(ctx), as<ggml_tensor>(a), p0, p1, p2, p3));
    }, "Pad tensor with zeros", py::arg("ctx"), py::arg("a"), py::arg("p0"), py::arg("p1"), py::arg("p2"), py::arg("p3"));

    m.def("arange", [](ContextPtr ctx, float start, float stop, float step) {
        return TensorPtr(ggml_arange(as<ggml_context>(ctx), start, stop, step));
    }, "Create range tensor", py::arg("ctx"), py::arg("start"), py::arg("stop"), py::arg("step"));

    m.def("argsort", [](ContextPtr ctx, TensorPtr a, ggml_sort_order order) {
        return TensorPtr(ggml_argsort(as<ggml_context>(ctx), as<ggml_tensor>(a), order));
    }, "Get sorted indices", py::arg("ctx"), py::arg("a"), py::arg("order"));

    m.def("top_k", [](ContextPtr ctx, TensorPtr a, int k) {
        return TensorPtr(ggml_top_k(as<ggml_context>(ctx), as<ggml_tensor>(a), k));
    }, "Top-K elements per row", py::arg("ctx"), py::arg("a"), py::arg("k"));

    m.def("diag", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_diag(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Create diagonal matrix", py::arg("ctx"), py::arg("a"));

    m.def("repeat_4d", [](ContextPtr ctx, TensorPtr a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
        return TensorPtr(ggml_repeat_4d(as<ggml_context>(ctx), as<ggml_tensor>(a), ne0, ne1, ne2, ne3));
    }, "Repeat tensor to specified shape", py::arg("ctx"), py::arg("a"), py::arg("ne0"), py::arg("ne1"), py::arg("ne2"), py::arg("ne3"));

    m.def("set_rows", [](ContextPtr ctx, TensorPtr a, TensorPtr b, TensorPtr c) {
        return TensorPtr(ggml_set_rows(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b), as<ggml_tensor>(c)));
    }, "Set rows by index (a: dest, b: src rows, c: indices)", py::arg("ctx"), py::arg("a"), py::arg("b"), py::arg("c"));

    m.def("scale_bias", [](ContextPtr ctx, TensorPtr a, float s, float b) {
        return TensorPtr(ggml_scale_bias(as<ggml_context>(ctx), as<ggml_tensor>(a), s, b));
    }, "Scale and bias: x = s*a + b", py::arg("ctx"), py::arg("a"), py::arg("s"), py::arg("b"));

    m.def("cross_entropy_loss", [](ContextPtr ctx, TensorPtr a, TensorPtr b) {
        return TensorPtr(ggml_cross_entropy_loss(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b)));
    }, "Cross entropy loss (a: logits, b: labels)", py::arg("ctx"), py::arg("a"), py::arg("b"));

    // Tensor property queries
    m.def("is_scalar", [](TensorPtr tensor) {
        return ggml_is_scalar(as<ggml_tensor>(tensor));
    }, "Check if tensor is scalar", py::arg("tensor"));

    m.def("is_vector", [](TensorPtr tensor) {
        return ggml_is_vector(as<ggml_tensor>(tensor));
    }, "Check if tensor is vector", py::arg("tensor"));

    m.def("is_matrix", [](TensorPtr tensor) {
        return ggml_is_matrix(as<ggml_tensor>(tensor));
    }, "Check if tensor is matrix", py::arg("tensor"));

    m.def("is_3d", [](TensorPtr tensor) {
        return ggml_is_3d(as<ggml_tensor>(tensor));
    }, "Check if tensor is 3D", py::arg("tensor"));

    m.def("is_empty", [](TensorPtr tensor) {
        return ggml_is_empty(as<ggml_tensor>(tensor));
    }, "Check if tensor is empty", py::arg("tensor"));

    m.def("are_same_stride", [](TensorPtr a, TensorPtr b) {
        return ggml_are_same_stride(as<ggml_tensor>(a), as<ggml_tensor>(b));
    }, "Check if two tensors have the same stride", py::arg("a"), py::arg("b"));

    m.def("can_repeat", [](TensorPtr a, TensorPtr b) {
        return ggml_can_repeat(as<ggml_tensor>(a), as<ggml_tensor>(b));
    }, "Check if tensor a can be repeated to shape of b", py::arg("a"), py::arg("b"));

    m.def("nbytes_pad", [](TensorPtr tensor) {
        return ggml_nbytes_pad(as<ggml_tensor>(tensor));
    }, "Get padded tensor size in bytes", py::arg("tensor"));

    m.def("row_size", [](ggml_type type, int64_t ne) {
        return ggml_row_size(type, ne);
    }, "Get row size in bytes", py::arg("type"), py::arg("ne"));

    m.def("op_desc", [](TensorPtr tensor) {
        return std::string(ggml_op_desc(as<ggml_tensor>(tensor)));
    }, "Get operation description", py::arg("tensor"));

    // Tensor value operations (from ggml-cpu.h)
    m.def("new_f32", [](ContextPtr ctx, float value) {
        return TensorPtr(ggml_new_f32(as<ggml_context>(ctx), value));
    }, "Create scalar f32 tensor", py::arg("ctx"), py::arg("value"));

    m.def("new_i32", [](ContextPtr ctx, int32_t value) {
        return TensorPtr(ggml_new_i32(as<ggml_context>(ctx), value));
    }, "Create scalar i32 tensor", py::arg("ctx"), py::arg("value"));

    m.def("set_f32", [](TensorPtr tensor, float value) {
        return TensorPtr(ggml_set_f32(as<ggml_tensor>(tensor), value));
    }, "Set all elements to f32 value", py::arg("tensor"), py::arg("value"));

    m.def("set_i32", [](TensorPtr tensor, int32_t value) {
        return TensorPtr(ggml_set_i32(as<ggml_tensor>(tensor), value));
    }, "Set all elements to i32 value", py::arg("tensor"), py::arg("value"));

    m.def("set_zero", [](TensorPtr tensor) {
        return TensorPtr(ggml_set_zero(as<ggml_tensor>(tensor)));
    }, "Set all elements to zero", py::arg("tensor"));

    m.def("get_f32_1d", [](TensorPtr tensor, int i) {
        return ggml_get_f32_1d(as<ggml_tensor>(tensor), i);
    }, "Get single f32 value", py::arg("tensor"), py::arg("i"));

    m.def("set_f32_1d", [](TensorPtr tensor, int i, float value) {
        ggml_set_f32_1d(as<ggml_tensor>(tensor), i, value);
    }, "Set single f32 value", py::arg("tensor"), py::arg("i"), py::arg("value"));

    m.def("get_i32_1d", [](TensorPtr tensor, int i) {
        return ggml_get_i32_1d(as<ggml_tensor>(tensor), i);
    }, "Get single i32 value", py::arg("tensor"), py::arg("i"));

    m.def("set_i32_1d", [](TensorPtr tensor, int i, int32_t value) {
        ggml_set_i32_1d(as<ggml_tensor>(tensor), i, value);
    }, "Set single i32 value", py::arg("tensor"), py::arg("i"), py::arg("value"));

    // Graph operations
    m.def("graph_n_nodes", [](GraphPtr cgraph) {
        return ggml_graph_n_nodes(as<ggml_cgraph>(cgraph));
    }, "Get number of nodes in graph", py::arg("cgraph"));

    m.def("graph_size", [](GraphPtr cgraph) {
        return ggml_graph_size(as<ggml_cgraph>(cgraph));
    }, "Get graph capacity", py::arg("cgraph"));

    m.def("graph_print", [](GraphPtr cgraph) {
        ggml_graph_print(as<ggml_cgraph>(cgraph));
    }, "Print graph info", py::arg("cgraph"));

    m.def("graph_dump_dot", [](GraphPtr gb, GraphPtr gf, const char* filename) {
        ggml_graph_dump_dot(as<ggml_cgraph>(gb), as<ggml_cgraph>(gf), filename);
    }, "Export graph to dot file", py::arg("gb"), py::arg("gf"), py::arg("filename"));

    // Context info
    m.def("used_mem", [](ContextPtr ctx) {
        return ggml_used_mem(as<ggml_context>(ctx));
    }, "Get used memory in context", py::arg("ctx"));

    m.def("set_no_alloc", [](ContextPtr ctx, bool no_alloc) {
        ggml_set_no_alloc(as<ggml_context>(ctx), no_alloc);
    }, "Set no_alloc flag", py::arg("ctx"), py::arg("no_alloc"));

    m.def("get_no_alloc", [](ContextPtr ctx) {
        return ggml_get_no_alloc(as<ggml_context>(ctx));
    }, "Get no_alloc flag", py::arg("ctx"));

    m.def("get_first_tensor", [](ContextPtr ctx) {
        return TensorPtr(ggml_get_first_tensor(as<ggml_context>(ctx)));
    }, "Get first tensor in context", py::arg("ctx"));

    m.def("get_next_tensor", [](ContextPtr ctx, TensorPtr tensor) {
        return TensorPtr(ggml_get_next_tensor(as<ggml_context>(ctx), as<ggml_tensor>(tensor)));
    }, "Get next tensor in context", py::arg("ctx"), py::arg("tensor"));

    // GLU activations
    m.def("swiglu", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_swiglu(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "SwiGLU activation", py::arg("ctx"), py::arg("a"));

    m.def("reglu", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_reglu(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "ReGLU activation", py::arg("ctx"), py::arg("a"));

    m.def("geglu", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_geglu(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "GEGLU activation", py::arg("ctx"), py::arg("a"));

    // Parameter and loss marking
    m.def("set_param", [](TensorPtr tensor) {
        ggml_set_param(as<ggml_tensor>(tensor));
    }, "Mark tensor as trainable parameter", py::arg("tensor"));

    m.def("set_loss", [](TensorPtr tensor) {
        ggml_set_loss(as<ggml_tensor>(tensor));
    }, "Mark tensor as loss", py::arg("tensor"));

    // Backend tensor operations
    m.def("backend_tensor_copy", [](TensorPtr src, TensorPtr dst) {
        ggml_backend_tensor_copy(as<ggml_tensor>(src), as<ggml_tensor>(dst));
    }, "Copy tensor between backends", py::arg("src"), py::arg("dst"));

    m.def("backend_tensor_memset", [](TensorPtr tensor, uint8_t value, size_t offset, size_t size) {
        ggml_backend_tensor_memset(as<ggml_tensor>(tensor), value, offset, size);
    }, "Fill tensor memory with value", py::arg("tensor"), py::arg("value"), py::arg("offset"), py::arg("size"));

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

    // GGUF KV metadata functions
    m.def("gguf_get_n_kv", [](GGUFContextPtr ctx) {
        return gguf_get_n_kv(static_cast<const struct gguf_context*>(ctx.ptr));
    }, "Get number of KV pairs in GGUF file", py::arg("ctx"));

    m.def("gguf_find_key", [](GGUFContextPtr ctx, const char* key) {
        return gguf_find_key(static_cast<const struct gguf_context*>(ctx.ptr), key);
    }, "Find key index by name, returns -1 if not found", py::arg("ctx"), py::arg("key"));

    m.def("gguf_get_key", [](GGUFContextPtr ctx, int64_t key_id) {
        return gguf_get_key(static_cast<const struct gguf_context*>(ctx.ptr), key_id);
    }, "Get key name by index", py::arg("ctx"), py::arg("key_id"));

    m.def("gguf_get_kv_type", [](GGUFContextPtr ctx, int64_t key_id) {
        return static_cast<int>(gguf_get_kv_type(static_cast<const struct gguf_context*>(ctx.ptr), key_id));
    }, "Get KV pair type by key index", py::arg("ctx"), py::arg("key_id"));

    m.def("gguf_get_val_str", [](GGUFContextPtr ctx, int64_t key_id) {
        return gguf_get_val_str(static_cast<const struct gguf_context*>(ctx.ptr), key_id);
    }, "Get string value by key index", py::arg("ctx"), py::arg("key_id"));

    m.def("gguf_get_val_u32", [](GGUFContextPtr ctx, int64_t key_id) {
        return gguf_get_val_u32(static_cast<const struct gguf_context*>(ctx.ptr), key_id);
    }, "Get uint32 value by key index", py::arg("ctx"), py::arg("key_id"));

    m.def("gguf_get_val_i32", [](GGUFContextPtr ctx, int64_t key_id) {
        return gguf_get_val_i32(static_cast<const struct gguf_context*>(ctx.ptr), key_id);
    }, "Get int32 value by key index", py::arg("ctx"), py::arg("key_id"));

    m.def("gguf_get_val_f32", [](GGUFContextPtr ctx, int64_t key_id) {
        return gguf_get_val_f32(static_cast<const struct gguf_context*>(ctx.ptr), key_id);
    }, "Get float32 value by key index", py::arg("ctx"), py::arg("key_id"));

    m.def("gguf_get_val_i64", [](GGUFContextPtr ctx, int64_t key_id) {
        return gguf_get_val_i64(static_cast<const struct gguf_context*>(ctx.ptr), key_id);
    }, "Get int64 value by key index", py::arg("ctx"), py::arg("key_id"));

    m.def("gguf_get_arr_n", [](GGUFContextPtr ctx, int64_t key_id) {
        return gguf_get_arr_n(static_cast<const struct gguf_context*>(ctx.ptr), key_id);
    }, "Get array length by key index", py::arg("ctx"), py::arg("key_id"));

    m.def("gguf_get_arr_type", [](GGUFContextPtr ctx, int64_t key_id) {
        return static_cast<int>(gguf_get_arr_type(static_cast<const struct gguf_context*>(ctx.ptr), key_id));
    }, "Get array element type by key index", py::arg("ctx"), py::arg("key_id"));

    m.def("gguf_get_arr_str", [](GGUFContextPtr ctx, int64_t key_id, size_t i) {
        return gguf_get_arr_str(static_cast<const struct gguf_context*>(ctx.ptr), key_id, i);
    }, "Get i-th string from array by key index", py::arg("ctx"), py::arg("key_id"), py::arg("i"));

    m.def("is_contiguous_rows", [](TensorPtr tensor) {
        return ggml_is_contiguous_rows(as<ggml_tensor>(tensor));
    }, "Check if tensor has contiguous rows", py::arg("tensor"));

    m.def("step", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_step(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Step activation: result = (a > 0) ? 1 : 0", py::arg("ctx"), py::arg("a"));

    m.def("step_inplace", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_step_inplace(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "Step activation (in-place)", py::arg("ctx"), py::arg("a"));

    m.def("gelu_erf", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_gelu_erf(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "GELU activation using ERF", py::arg("ctx"), py::arg("a"));

    m.def("gelu_erf_inplace", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_gelu_erf_inplace(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "GELU activation using ERF (in-place)", py::arg("ctx"), py::arg("a"));

    m.def("add_cast", [](ContextPtr ctx, TensorPtr a, TensorPtr b, ggml_type type) {
        return TensorPtr(ggml_add_cast(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b), type));
    }, "Element-wise addition with type casting", py::arg("ctx"), py::arg("a"), py::arg("b"), py::arg("type"));

    m.def("acc", [](ContextPtr ctx, TensorPtr a, TensorPtr b,
                    size_t nb1, size_t nb2, size_t nb3, size_t offset) {
        return TensorPtr(ggml_acc(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b),
                                  nb1, nb2, nb3, offset));
    }, "Accumulate b into a view of a at offset", py::arg("ctx"), py::arg("a"), py::arg("b"),
       py::arg("nb1"), py::arg("nb2"), py::arg("nb3"), py::arg("offset"));

    m.def("acc_inplace", [](ContextPtr ctx, TensorPtr a, TensorPtr b,
                             size_t nb1, size_t nb2, size_t nb3, size_t offset) {
        return TensorPtr(ggml_acc_inplace(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b),
                                          nb1, nb2, nb3, offset));
    }, "Accumulate b into a view of a at offset (in-place)", py::arg("ctx"), py::arg("a"), py::arg("b"),
       py::arg("nb1"), py::arg("nb2"), py::arg("nb3"), py::arg("offset"));

    m.def("mul_mat_id", [](ContextPtr ctx, TensorPtr as_t, TensorPtr b, TensorPtr ids) {
        return TensorPtr(ggml_mul_mat_id(as<ggml_context>(ctx), as<ggml_tensor>(as_t),
                                         as<ggml_tensor>(b), as<ggml_tensor>(ids)));
    }, "MoE indirect matrix multiplication", py::arg("ctx"), py::arg("as"), py::arg("b"), py::arg("ids"));

    // GLU variants
    py::enum_<ggml_glu_op>(m, "GluOp")
        .value("REGLU", GGML_GLU_OP_REGLU)
        .value("GEGLU", GGML_GLU_OP_GEGLU)
        .value("SWIGLU", GGML_GLU_OP_SWIGLU)
        .value("SWIGLU_OAI", GGML_GLU_OP_SWIGLU_OAI)
        .value("GEGLU_ERF", GGML_GLU_OP_GEGLU_ERF)
        .value("GEGLU_QUICK", GGML_GLU_OP_GEGLU_QUICK)
        .export_values();

    m.def("geglu_erf", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_geglu_erf(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "GEGLU with ERF gating (fused, gate in second half)", py::arg("ctx"), py::arg("a"));

    m.def("geglu_erf_swapped", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_geglu_erf_swapped(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "GEGLU with ERF gating (fused, gate in first half)", py::arg("ctx"), py::arg("a"));

    m.def("geglu_quick", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_geglu_quick(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "GEGLU with quick GELU gating (fused, gate in second half)", py::arg("ctx"), py::arg("a"));

    m.def("geglu_quick_swapped", [](ContextPtr ctx, TensorPtr a) {
        return TensorPtr(ggml_geglu_quick_swapped(as<ggml_context>(ctx), as<ggml_tensor>(a)));
    }, "GEGLU with quick GELU gating (fused, gate in first half)", py::arg("ctx"), py::arg("a"));

    m.def("reglu_split", [](ContextPtr ctx, TensorPtr a, TensorPtr b) {
        return TensorPtr(ggml_reglu_split(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b)));
    }, "ReGLU (split: a=input, b=gate)", py::arg("ctx"), py::arg("a"), py::arg("b"));

    m.def("geglu_split", [](ContextPtr ctx, TensorPtr a, TensorPtr b) {
        return TensorPtr(ggml_geglu_split(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b)));
    }, "GEGLU (split: a=input, b=gate)", py::arg("ctx"), py::arg("a"), py::arg("b"));

    m.def("swiglu_split", [](ContextPtr ctx, TensorPtr a, TensorPtr b) {
        return TensorPtr(ggml_swiglu_split(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b)));
    }, "SwiGLU (split: a=input, b=gate)", py::arg("ctx"), py::arg("a"), py::arg("b"));

    m.def("geglu_erf_split", [](ContextPtr ctx, TensorPtr a, TensorPtr b) {
        return TensorPtr(ggml_geglu_erf_split(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b)));
    }, "GEGLU-ERF (split: a=input, b=gate)", py::arg("ctx"), py::arg("a"), py::arg("b"));

    m.def("geglu_quick_split", [](ContextPtr ctx, TensorPtr a, TensorPtr b) {
        return TensorPtr(ggml_geglu_quick_split(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b)));
    }, "GEGLU-quick (split: a=input, b=gate)", py::arg("ctx"), py::arg("a"), py::arg("b"));

    m.def("swiglu_oai", [](ContextPtr ctx, TensorPtr a, TensorPtr b, float alpha, float limit) {
        return TensorPtr(ggml_swiglu_oai(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b),
                                          alpha, limit));
    }, "OpenAI SwiGLU with alpha/limit", py::arg("ctx"), py::arg("a"), py::arg("b"),
       py::arg("alpha"), py::arg("limit"));

    // Interpolation and spatial ops
    py::enum_<ggml_scale_mode>(m, "ScaleMode")
        .value("NEAREST", GGML_SCALE_MODE_NEAREST)
        .value("BILINEAR", GGML_SCALE_MODE_BILINEAR)
        .export_values();

    py::enum_<ggml_scale_flag>(m, "ScaleFlag")
        .value("ALIGN_CORNERS", GGML_SCALE_FLAG_ALIGN_CORNERS)
        .export_values();

    m.def("interpolate", [](ContextPtr ctx, TensorPtr a,
                             int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                             uint32_t mode) {
        return TensorPtr(ggml_interpolate(as<ggml_context>(ctx), as<ggml_tensor>(a),
                                          ne0, ne1, ne2, ne3, mode));
    }, "Interpolate (resize) tensor to new spatial dimensions",
       py::arg("ctx"), py::arg("a"),
       py::arg("ne0"), py::arg("ne1"), py::arg("ne2"), py::arg("ne3"), py::arg("mode"));

    m.def("pad_ext", [](ContextPtr ctx, TensorPtr a,
                        int lp0, int rp0, int lp1, int rp1,
                        int lp2, int rp2, int lp3, int rp3) {
        return TensorPtr(ggml_pad_ext(as<ggml_context>(ctx), as<ggml_tensor>(a),
                                      lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3));
    }, "Pad tensor with independent left/right padding per dimension",
       py::arg("ctx"), py::arg("a"),
       py::arg("lp0"), py::arg("rp0"), py::arg("lp1"), py::arg("rp1"),
       py::arg("lp2"), py::arg("rp2"), py::arg("lp3"), py::arg("rp3"));

    m.def("timestep_embedding", [](ContextPtr ctx, TensorPtr timesteps, int dim, int max_period) {
        return TensorPtr(ggml_timestep_embedding(as<ggml_context>(ctx), as<ggml_tensor>(timesteps),
                                                  dim, max_period));
    }, "Sinusoidal timestep embedding for diffusion models",
       py::arg("ctx"), py::arg("timesteps"), py::arg("dim"), py::arg("max_period"));

    // Additional convolution operations
    m.def("im2col", [](ContextPtr ctx, TensorPtr a, TensorPtr b,
                       int s0, int s1, int p0, int p1, int d0, int d1,
                       bool is_2D, ggml_type dst_type) {
        return TensorPtr(ggml_im2col(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b),
                                     s0, s1, p0, p1, d0, d1, is_2D, dst_type));
    }, "Image-to-column transform (underlying primitive for conv)",
       py::arg("ctx"), py::arg("a"), py::arg("b"),
       py::arg("s0"), py::arg("s1"), py::arg("p0"), py::arg("p1"),
       py::arg("d0"), py::arg("d1"), py::arg("is_2D"), py::arg("dst_type"));

    m.def("conv_1d_ph", [](ContextPtr ctx, TensorPtr a, TensorPtr b, int s, int d) {
        return TensorPtr(ggml_conv_1d_ph(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b),
                                          s, d));
    }, "1D convolution with half-padding", py::arg("ctx"), py::arg("a"), py::arg("b"),
       py::arg("s"), py::arg("d"));

    m.def("conv_1d_dw", [](ContextPtr ctx, TensorPtr a, TensorPtr b, int s0, int p0, int d0) {
        return TensorPtr(ggml_conv_1d_dw(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b),
                                          s0, p0, d0));
    }, "Depthwise 1D convolution", py::arg("ctx"), py::arg("a"), py::arg("b"),
       py::arg("s0"), py::arg("p0"), py::arg("d0"));

    m.def("conv_1d_dw_ph", [](ContextPtr ctx, TensorPtr a, TensorPtr b, int s0, int d0) {
        return TensorPtr(ggml_conv_1d_dw_ph(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b),
                                             s0, d0));
    }, "Depthwise 1D convolution with half-padding", py::arg("ctx"), py::arg("a"), py::arg("b"),
       py::arg("s0"), py::arg("d0"));

    m.def("conv_transpose_1d", [](ContextPtr ctx, TensorPtr a, TensorPtr b, int s0, int p0, int d0) {
        return TensorPtr(ggml_conv_transpose_1d(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b),
                                                 s0, p0, d0));
    }, "1D transposed convolution (deconvolution)", py::arg("ctx"), py::arg("a"), py::arg("b"),
       py::arg("s0"), py::arg("p0"), py::arg("d0"));

    m.def("conv_2d_sk_p0", [](ContextPtr ctx, TensorPtr a, TensorPtr b) {
        return TensorPtr(ggml_conv_2d_sk_p0(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b)));
    }, "2D convolution with stride=kernel, padding=0 (e.g. SAM patch embed)",
       py::arg("ctx"), py::arg("a"), py::arg("b"));

    m.def("conv_2d_s1_ph", [](ContextPtr ctx, TensorPtr a, TensorPtr b) {
        return TensorPtr(ggml_conv_2d_s1_ph(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b)));
    }, "2D convolution with stride=1, half-padding (e.g. SAM encoder)",
       py::arg("ctx"), py::arg("a"), py::arg("b"));

    m.def("conv_2d_dw", [](ContextPtr ctx, TensorPtr a, TensorPtr b,
                            int s0, int s1, int p0, int p1, int d0, int d1) {
        return TensorPtr(ggml_conv_2d_dw(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b),
                                          s0, s1, p0, p1, d0, d1));
    }, "Depthwise 2D convolution",
       py::arg("ctx"), py::arg("a"), py::arg("b"),
       py::arg("s0"), py::arg("s1"), py::arg("p0"), py::arg("p1"), py::arg("d0"), py::arg("d1"));

    m.def("conv_3d", [](ContextPtr ctx, TensorPtr a, TensorPtr b,
                        int64_t IC,
                        int s0, int s1, int s2,
                        int p0, int p1, int p2,
                        int d0, int d1, int d2) {
        return TensorPtr(ggml_conv_3d(as<ggml_context>(ctx), as<ggml_tensor>(a), as<ggml_tensor>(b),
                                      IC, s0, s1, s2, p0, p1, p2, d0, d1, d2));
    }, "3D convolution",
       py::arg("ctx"), py::arg("a"), py::arg("b"), py::arg("IC"),
       py::arg("s0"), py::arg("s1"), py::arg("s2"),
       py::arg("p0"), py::arg("p1"), py::arg("p2"),
       py::arg("d0"), py::arg("d1"), py::arg("d2"));

    // Vision ops (SAM)
    m.def("win_part", [](ContextPtr ctx, TensorPtr a, int w) {
        return TensorPtr(ggml_win_part(as<ggml_context>(ctx), as<ggml_tensor>(a), w));
    }, "Partition tensor into non-overlapping windows of size w",
       py::arg("ctx"), py::arg("a"), py::arg("w"));

    m.def("win_unpart", [](ContextPtr ctx, TensorPtr a, int w0, int h0, int w) {
        return TensorPtr(ggml_win_unpart(as<ggml_context>(ctx), as<ggml_tensor>(a), w0, h0, w));
    }, "Reverse window partition (reconstruct spatial tensor)",
       py::arg("ctx"), py::arg("a"), py::arg("w0"), py::arg("h0"), py::arg("w"));

    m.def("get_rel_pos", [](ContextPtr ctx, TensorPtr a, int qh, int kh) {
        return TensorPtr(ggml_get_rel_pos(as<ggml_context>(ctx), as<ggml_tensor>(a), qh, kh));
    }, "Slice relative position bias for given query/key heights",
       py::arg("ctx"), py::arg("a"), py::arg("qh"), py::arg("kh"));

    m.def("add_rel_pos", [](ContextPtr ctx, TensorPtr a, TensorPtr pw, TensorPtr ph) {
        return TensorPtr(ggml_add_rel_pos(as<ggml_context>(ctx), as<ggml_tensor>(a),
                                          as<ggml_tensor>(pw), as<ggml_tensor>(ph)));
    }, "Add relative position bias to attention scores",
       py::arg("ctx"), py::arg("a"), py::arg("pw"), py::arg("ph"));

    m.def("add_rel_pos_inplace", [](ContextPtr ctx, TensorPtr a, TensorPtr pw, TensorPtr ph) {
        return TensorPtr(ggml_add_rel_pos_inplace(as<ggml_context>(ctx), as<ggml_tensor>(a),
                                                   as<ggml_tensor>(pw), as<ggml_tensor>(ph)));
    }, "Add relative position bias to attention scores (in-place)",
       py::arg("ctx"), py::arg("a"), py::arg("pw"), py::arg("ph"));

    // Sequence and recurrent model ops
    m.def("ssm_conv", [](ContextPtr ctx, TensorPtr sx, TensorPtr c) {
        return TensorPtr(ggml_ssm_conv(as<ggml_context>(ctx), as<ggml_tensor>(sx), as<ggml_tensor>(c)));
    }, "Mamba SSM convolution step", py::arg("ctx"), py::arg("sx"), py::arg("c"));

    m.def("ssm_scan", [](ContextPtr ctx, TensorPtr s, TensorPtr x,
                         TensorPtr dt, TensorPtr A, TensorPtr B, TensorPtr C, TensorPtr ids) {
        return TensorPtr(ggml_ssm_scan(as<ggml_context>(ctx),
                                        as<ggml_tensor>(s), as<ggml_tensor>(x),
                                        as<ggml_tensor>(dt), as<ggml_tensor>(A),
                                        as<ggml_tensor>(B), as<ggml_tensor>(C),
                                        as<ggml_tensor>(ids)));
    }, "Mamba SSM selective scan",
       py::arg("ctx"), py::arg("s"), py::arg("x"),
       py::arg("dt"), py::arg("A"), py::arg("B"), py::arg("C"), py::arg("ids"));

    m.def("rwkv_wkv6", [](ContextPtr ctx, TensorPtr k, TensorPtr v,
                           TensorPtr r, TensorPtr tf, TensorPtr td, TensorPtr state) {
        return TensorPtr(ggml_rwkv_wkv6(as<ggml_context>(ctx),
                                         as<ggml_tensor>(k), as<ggml_tensor>(v),
                                         as<ggml_tensor>(r), as<ggml_tensor>(tf),
                                         as<ggml_tensor>(td), as<ggml_tensor>(state)));
    }, "RWKV WKV6 recurrent step",
       py::arg("ctx"), py::arg("k"), py::arg("v"),
       py::arg("r"), py::arg("tf"), py::arg("td"), py::arg("state"));

    m.def("rwkv_wkv7", [](ContextPtr ctx, TensorPtr r, TensorPtr w,
                           TensorPtr k, TensorPtr v, TensorPtr a, TensorPtr b, TensorPtr state) {
        return TensorPtr(ggml_rwkv_wkv7(as<ggml_context>(ctx),
                                         as<ggml_tensor>(r), as<ggml_tensor>(w),
                                         as<ggml_tensor>(k), as<ggml_tensor>(v),
                                         as<ggml_tensor>(a), as<ggml_tensor>(b),
                                         as<ggml_tensor>(state)));
    }, "RWKV WKV7 recurrent step",
       py::arg("ctx"), py::arg("r"), py::arg("w"),
       py::arg("k"), py::arg("v"), py::arg("a"), py::arg("b"), py::arg("state"));

    m.def("gated_linear_attn", [](ContextPtr ctx, TensorPtr k, TensorPtr v,
                                   TensorPtr q, TensorPtr g, TensorPtr state, float scale) {
        return TensorPtr(ggml_gated_linear_attn(as<ggml_context>(ctx),
                                                 as<ggml_tensor>(k), as<ggml_tensor>(v),
                                                 as<ggml_tensor>(q), as<ggml_tensor>(g),
                                                 as<ggml_tensor>(state), scale));
    }, "Gated Linear Attention step",
       py::arg("ctx"), py::arg("k"), py::arg("v"),
       py::arg("q"), py::arg("g"), py::arg("state"), py::arg("scale"));

    // Optimizer ops
    m.def("opt_step_adamw", [](ContextPtr ctx, TensorPtr a, TensorPtr grad,
                                TensorPtr m, TensorPtr v, TensorPtr adamw_params) {
        return TensorPtr(ggml_opt_step_adamw(as<ggml_context>(ctx),
                                              as<ggml_tensor>(a), as<ggml_tensor>(grad),
                                              as<ggml_tensor>(m), as<ggml_tensor>(v),
                                              as<ggml_tensor>(adamw_params)));
    }, "AdamW optimizer step",
       py::arg("ctx"), py::arg("a"), py::arg("grad"),
       py::arg("m"), py::arg("v"), py::arg("adamw_params"));

    m.def("opt_step_sgd", [](ContextPtr ctx, TensorPtr a, TensorPtr grad, TensorPtr sgd_params) {
        return TensorPtr(ggml_opt_step_sgd(as<ggml_context>(ctx),
                                            as<ggml_tensor>(a), as<ggml_tensor>(grad),
                                            as<ggml_tensor>(sgd_params)));
    }, "SGD optimizer step",
       py::arg("ctx"), py::arg("a"), py::arg("grad"), py::arg("sgd_params"));
}

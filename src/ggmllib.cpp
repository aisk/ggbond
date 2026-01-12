#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>
#ifdef __APPLE__
#include <ggml-metal.h>
#endif

namespace py = pybind11;

PYBIND11_MODULE(ggml, m) {
    m.doc() = "Simple and naive GGML binding";

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
    m.def("backend_cpu_init", []() { return static_cast<void*>(ggml_backend_cpu_init()); }, "Initialize CPU backend");
    m.def("backend_is_cpu", [](void* backend) { return ggml_backend_is_cpu(static_cast<ggml_backend_t>(backend)); }, "Check if backend is CPU backend", py::arg("backend"));
    m.def("backend_cpu_set_n_threads", [](void* backend, int n_threads) { ggml_backend_cpu_set_n_threads(static_cast<ggml_backend_t>(backend), n_threads); }, "Set number of threads for CPU backend", py::arg("backend"), py::arg("n_threads"));

#ifdef __APPLE__
    // Metal backend functions (macOS only)
    m.def("backend_metal_init", []() { return static_cast<void*>(ggml_backend_metal_init()); }, "Initialize Metal backend (macOS only)");
    m.def("backend_is_metal", [](void* backend) { return ggml_backend_is_metal(static_cast<ggml_backend_t>(backend)); }, "Check if backend is Metal backend", py::arg("backend"));
    m.def("backend_metal_supports_family", [](void* backend, int family) { return ggml_backend_metal_supports_family(static_cast<ggml_backend_t>(backend), family); }, "Check if Metal device supports specific feature family", py::arg("backend"), py::arg("family"));
    m.def("backend_metal_capture_next_compute", [](void* backend) { ggml_backend_metal_capture_next_compute(static_cast<ggml_backend_t>(backend)); }, "Capture next Metal compute for debugging", py::arg("backend"));
#endif
    m.def("tensor_overhead", &ggml_tensor_overhead, "Get the memory overhead of a tensor");
    m.def("graph_overhead", &ggml_graph_overhead, "Get the memory overhead of a graph");
    m.def("DEFAULT_GRAPH_SIZE", []() { return GGML_DEFAULT_GRAPH_SIZE; }, "Default graph size constant");
    m.def("context_init", [](size_t mem_size, void* mem_buffer = nullptr, bool no_alloc = false) {
        ggml_init_params params = {mem_size, mem_buffer, no_alloc};
        return static_cast<void*>(ggml_init(params));
    }, "Initialize GGML context", py::arg("mem_size"), py::arg("mem_buffer") = nullptr, py::arg("no_alloc") = false);
    m.def("new_tensor_1d", [](void* ctx, ggml_type type, int64_t ne0) {
        return static_cast<void*>(ggml_new_tensor_1d(
            static_cast<ggml_context*>(ctx),
            type,
            ne0
        ));
    }, "Create a new 1D tensor", py::arg("ctx"), py::arg("type"), py::arg("ne0"));
    m.def("new_tensor_2d", [](void* ctx, ggml_type type, int64_t ne0, int64_t ne1) {
        return static_cast<void*>(ggml_new_tensor_2d(
            static_cast<ggml_context*>(ctx),
            type,
            ne0,
            ne1
        ));
    }, "Create a new 2D tensor", py::arg("ctx"), py::arg("type"), py::arg("ne0"), py::arg("ne1"));
    m.def("new_tensor_3d", [](void* ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
        return static_cast<void*>(ggml_new_tensor_3d(
            static_cast<ggml_context*>(ctx),
            type,
            ne0,
            ne1,
            ne2
        ));
    }, "Create a new 3D tensor", py::arg("ctx"), py::arg("type"), py::arg("ne0"), py::arg("ne1"), py::arg("ne2"));
    m.def("new_tensor_4d", [](void* ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
        return static_cast<void*>(ggml_new_tensor_4d(
            static_cast<ggml_context*>(ctx),
            type,
            ne0,
            ne1,
            ne2,
            ne3
        ));
    }, "Create a new 4D tensor", py::arg("ctx"), py::arg("type"), py::arg("ne0"), py::arg("ne1"), py::arg("ne2"), py::arg("ne3"));
    m.def("type_size", &ggml_type_size, "Get size in bytes for all elements in a block of the given type", py::arg("type"));
    m.def("blck_size", &ggml_blck_size, "Get block size (number of elements per block) for the given type", py::arg("type"));
    m.def("backend_alloc_ctx_tensors", [](void* ctx, void* backend) {
        return static_cast<void*>(ggml_backend_alloc_ctx_tensors(
            static_cast<ggml_context*>(ctx),
            static_cast<ggml_backend_t>(backend)
        ));
    }, "Allocate all tensors in a GGML context to a backend", py::arg("ctx"), py::arg("backend"));
    m.def("backend_buffer_free", [](void* buffer) {
        ggml_backend_buffer_free(static_cast<ggml_backend_buffer_t>(buffer));
    }, "Free backend buffer", py::arg("buffer"));
    m.def("backend_tensor_set", [](void* tensor, py::buffer data, size_t offset, size_t size) {
        py::buffer_info info = data.request();
        size_t data_size = info.itemsize * info.size;
        if (size == 0) {
            size = data_size;
        }
        if (data_size < size) {
            throw std::runtime_error("Data buffer is too small");
        }
        ggml_backend_tensor_set(
            static_cast<ggml_tensor*>(tensor),
            static_cast<const void*>(info.ptr),
            offset,
            size
        );
    }, "Set tensor data from Python buffer", py::arg("tensor"), py::arg("data"), py::arg("offset") = 0, py::arg("size") = 0);
    m.def("new_graph", [](void* ctx) {
        return static_cast<void*>(ggml_new_graph(static_cast<ggml_context*>(ctx)));
    }, "Create a new computation graph", py::arg("ctx"));
    m.def("new_graph_custom", [](void* ctx, size_t size, bool grads) {
        return static_cast<void*>(ggml_new_graph_custom(static_cast<ggml_context*>(ctx), size, grads));
    }, "Create a new computation graph with custom size and gradient settings", py::arg("ctx"), py::arg("size"), py::arg("grads"));
    m.def("mul_mat", [](void* ctx, void* a, void* b) {
        return static_cast<void*>(ggml_mul_mat(
            static_cast<ggml_context*>(ctx),
            static_cast<ggml_tensor*>(a),
            static_cast<ggml_tensor*>(b)
        ));
    }, "Matrix multiplication: result = a * b", py::arg("ctx"), py::arg("a"), py::arg("b"));

    // Binary operations
    m.def("add", [](void* ctx, void* a, void* b) {
        return static_cast<void*>(ggml_add(
            static_cast<ggml_context*>(ctx),
            static_cast<ggml_tensor*>(a),
            static_cast<ggml_tensor*>(b)
        ));
    }, "Element-wise addition: result = a + b", py::arg("ctx"), py::arg("a"), py::arg("b"));

    m.def("sub", [](void* ctx, void* a, void* b) {
        return static_cast<void*>(ggml_sub(
            static_cast<ggml_context*>(ctx),
            static_cast<ggml_tensor*>(a),
            static_cast<ggml_tensor*>(b)
        ));
    }, "Element-wise subtraction: result = a - b", py::arg("ctx"), py::arg("a"), py::arg("b"));

    m.def("mul", [](void* ctx, void* a, void* b) {
        return static_cast<void*>(ggml_mul(
            static_cast<ggml_context*>(ctx),
            static_cast<ggml_tensor*>(a),
            static_cast<ggml_tensor*>(b)
        ));
    }, "Element-wise multiplication: result = a * b", py::arg("ctx"), py::arg("a"), py::arg("b"));

    m.def("div", [](void* ctx, void* a, void* b) {
        return static_cast<void*>(ggml_div(
            static_cast<ggml_context*>(ctx),
            static_cast<ggml_tensor*>(a),
            static_cast<ggml_tensor*>(b)
        ));
    }, "Element-wise division: result = a / b", py::arg("ctx"), py::arg("a"), py::arg("b"));

    // Additional binary operations
    m.def("add1", [](void* ctx, void* a, void* b) {
        return static_cast<void*>(ggml_add1(
            static_cast<ggml_context*>(ctx),
            static_cast<ggml_tensor*>(a),
            static_cast<ggml_tensor*>(b)
        ));
    }, "Add scalar b to each row of matrix a", py::arg("ctx"), py::arg("a"), py::arg("b"));

    m.def("out_prod", [](void* ctx, void* a, void* b) {
        return static_cast<void*>(ggml_out_prod(
            static_cast<ggml_context*>(ctx),
            static_cast<ggml_tensor*>(a),
            static_cast<ggml_tensor*>(b)
        ));
    }, "Outer product: result = a @ b^T", py::arg("ctx"), py::arg("a"), py::arg("b"));

    m.def("concat", [](void* ctx, void* a, void* b, int dim) {
        return static_cast<void*>(ggml_concat(
            static_cast<ggml_context*>(ctx),
            static_cast<ggml_tensor*>(a),
            static_cast<ggml_tensor*>(b),
            dim
        ));
    }, "Concatenate tensors a and b along dimension dim", py::arg("ctx"), py::arg("a"), py::arg("b"), py::arg("dim"));

    m.def("count_equal", [](void* ctx, void* a, void* b) {
        return static_cast<void*>(ggml_count_equal(
            static_cast<ggml_context*>(ctx),
            static_cast<ggml_tensor*>(a),
            static_cast<ggml_tensor*>(b)
        ));
    }, "Count number of equal elements in a and b", py::arg("ctx"), py::arg("a"), py::arg("b"));

    // Unary operations
    m.def("abs", [](void* ctx, void* a) {
        return static_cast<void*>(ggml_abs(
            static_cast<ggml_context*>(ctx),
            static_cast<ggml_tensor*>(a)
        ));
    }, "Element-wise absolute value: result = |a|", py::arg("ctx"), py::arg("a"));

    m.def("neg", [](void* ctx, void* a) {
        return static_cast<void*>(ggml_neg(
            static_cast<ggml_context*>(ctx),
            static_cast<ggml_tensor*>(a)
        ));
    }, "Element-wise negation: result = -a", py::arg("ctx"), py::arg("a"));

    m.def("sqrt", [](void* ctx, void* a) {
        return static_cast<void*>(ggml_sqrt(
            static_cast<ggml_context*>(ctx),
            static_cast<ggml_tensor*>(a)
        ));
    }, "Element-wise square root: result = sqrt(a)", py::arg("ctx"), py::arg("a"));

    m.def("sqr", [](void* ctx, void* a) {
        return static_cast<void*>(ggml_sqr(
            static_cast<ggml_context*>(ctx),
            static_cast<ggml_tensor*>(a)
        ));
    }, "Element-wise square: result = a^2", py::arg("ctx"), py::arg("a"));

    // Additional unary operations
    m.def("log", [](void* ctx, void* a) {
        return static_cast<void*>(ggml_log(
            static_cast<ggml_context*>(ctx),
            static_cast<ggml_tensor*>(a)
        ));
    }, "Element-wise natural logarithm: result = log(a)", py::arg("ctx"), py::arg("a"));

    m.def("exp", [](void* ctx, void* a) {
        return static_cast<void*>(ggml_exp(
            static_cast<ggml_context*>(ctx),
            static_cast<ggml_tensor*>(a)
        ));
    }, "Element-wise exponential: result = exp(a)", py::arg("ctx"), py::arg("a"));

    m.def("tanh", [](void* ctx, void* a) {
        return static_cast<void*>(ggml_tanh(
            static_cast<ggml_context*>(ctx),
            static_cast<ggml_tensor*>(a)
        ));
    }, "Element-wise hyperbolic tangent: result = tanh(a)", py::arg("ctx"), py::arg("a"));

    m.def("sigmoid", [](void* ctx, void* a) {
        return static_cast<void*>(ggml_sigmoid(
            static_cast<ggml_context*>(ctx),
            static_cast<ggml_tensor*>(a)
        ));
    }, "Element-wise sigmoid activation: result = 1 / (1 + exp(-a))", py::arg("ctx"), py::arg("a"));

    m.def("relu", [](void* ctx, void* a) {
        return static_cast<void*>(ggml_relu(
            static_cast<ggml_context*>(ctx),
            static_cast<ggml_tensor*>(a)
        ));
    }, "Element-wise ReLU activation: result = max(0, a)", py::arg("ctx"), py::arg("a"));

    m.def("gelu", [](void* ctx, void* a) {
        return static_cast<void*>(ggml_gelu(
            static_cast<ggml_context*>(ctx),
            static_cast<ggml_tensor*>(a)
        ));
    }, "Element-wise GELU activation (Gaussian Error Linear Unit)", py::arg("ctx"), py::arg("a"));

    // Reduction operations
    m.def("sum", [](void* ctx, void* a) {
        return static_cast<void*>(ggml_sum(
            static_cast<ggml_context*>(ctx),
            static_cast<ggml_tensor*>(a)
        ));
    }, "Sum all elements in tensor, returns scalar", py::arg("ctx"), py::arg("a"));

    m.def("mean", [](void* ctx, void* a) {
        return static_cast<void*>(ggml_mean(
            static_cast<ggml_context*>(ctx),
            static_cast<ggml_tensor*>(a)
        ));
    }, "Mean of all elements along rows", py::arg("ctx"), py::arg("a"));
    m.def("build_forward_expand", [](void* cgraph, void* tensor) {
        ggml_build_forward_expand(
            static_cast<ggml_cgraph*>(cgraph),
            static_cast<ggml_tensor*>(tensor)
        );
    }, "Build forward computation graph from tensor", py::arg("cgraph"), py::arg("tensor"));
    m.def("graph_compute_with_ctx", [](void* ctx, void* cgraph, int n_threads) {
        return ggml_graph_compute_with_ctx(
            static_cast<ggml_context*>(ctx),
            static_cast<ggml_cgraph*>(cgraph),
            n_threads
        );
    }, "Compute the graph with given context and thread count", py::arg("ctx"), py::arg("cgraph"), py::arg("n_threads") = 1);
    m.def("context_free", [](void* ctx) {
        ggml_free(static_cast<ggml_context*>(ctx));
    }, "Free GGML context and all its allocated memory", py::arg("ctx"));
    m.def("get_data", [](void* tensor) {
        return reinterpret_cast<uintptr_t>(ggml_get_data(static_cast<ggml_tensor*>(tensor)));
    }, "Get data pointer from tensor (as uintptr_t)", py::arg("tensor"));
    m.def("get_data_f32", [](void* tensor) {
        return reinterpret_cast<uintptr_t>(ggml_get_data_f32(static_cast<ggml_tensor*>(tensor)));
    }, "Get float32 data pointer from tensor (as uintptr_t)", py::arg("tensor"));
    m.def("time_init", []() {
        ggml_time_init();
    }, "Initialize time measurement - call this once at the beginning of the program");

    // Graph allocator functions
    m.def("gallocr_new", [](void* buffer_type) {
        return static_cast<void*>(ggml_gallocr_new(static_cast<ggml_backend_buffer_type_t>(buffer_type)));
    }, "Create new graph allocator", py::arg("buffer_type"));

    m.def("gallocr_reserve", [](void* allocr, void* cgraph) {
        return ggml_gallocr_reserve(
            static_cast<ggml_gallocr_t>(allocr),
            static_cast<ggml_cgraph*>(cgraph)
        );
    }, " reserve memory for graph computation", py::arg("allocr"), py::arg("cgraph"));

    m.def("gallocr_alloc_graph", [](void* allocr, void* cgraph) {
        return ggml_gallocr_alloc_graph(
            static_cast<ggml_gallocr_t>(allocr),
            static_cast<ggml_cgraph*>(cgraph)
        );
    }, "Allocate tensors for graph computation", py::arg("allocr"), py::arg("cgraph"));

    m.def("gallocr_get_buffer_size", [](void* allocr, int buffer_index) {
        return ggml_gallocr_get_buffer_size(
            static_cast<ggml_gallocr_t>(allocr),
            buffer_index
        );
    }, "Get buffer size for allocated graph", py::arg("allocr"), py::arg("buffer_index") = 0);

    m.def("gallocr_free", [](void* allocr) {
        ggml_gallocr_free(static_cast<ggml_gallocr_t>(allocr));
    }, "Free graph allocator", py::arg("allocr"));

    // Backend buffer type
    m.def("backend_get_default_buffer_type", [](void* backend) {
        return static_cast<void*>(ggml_backend_get_default_buffer_type(static_cast<ggml_backend_t>(backend)));
    }, "Get default buffer type for backend", py::arg("backend"));

    // Graph node access
    m.def("graph_node", [](void* cgraph, int i) {
        return static_cast<void*>(ggml_graph_node(static_cast<ggml_cgraph*>(cgraph), i));
    }, "Get node from graph by index", py::arg("cgraph"), py::arg("i"));

    // Backend graph computation
    m.def("backend_graph_compute", [](void* backend, void* cgraph) {
        ggml_backend_graph_compute(
            static_cast<ggml_backend_t>(backend),
            static_cast<ggml_cgraph*>(cgraph)
        );
    }, "Compute graph using backend", py::arg("backend"), py::arg("cgraph"));

    // Backend tensor get
    m.def("backend_tensor_get", [](void* tensor, py::buffer data, size_t offset, size_t size) {
        py::buffer_info info = data.request();
        size_t data_size = info.itemsize * info.size;
        if (size == 0) {
            size = data_size;
        }
        if (data_size < size) {
            throw std::runtime_error("Data buffer is too small");
        }
        ggml_backend_tensor_get(
            static_cast<ggml_tensor*>(tensor),
            static_cast<void*>(info.ptr),
            offset,
            size
        );
    }, "Get tensor data to Python buffer", py::arg("tensor"), py::arg("data"), py::arg("offset") = 0, py::arg("size") = 0);

    // Backend management
    m.def("backend_free", [](void* backend) {
        ggml_backend_free(static_cast<ggml_backend_t>(backend));
    }, "Free backend", py::arg("backend"));

    // Tensor helper functions
    m.def("tensor_ne", [](void* tensor, int dim) {
        return static_cast<ggml_tensor*>(tensor)->ne[dim];
    }, "Get tensor dimension size", py::arg("tensor"), py::arg("dim"));

    m.def("tensor_nb", [](void* tensor, int dim) {
        return static_cast<ggml_tensor*>(tensor)->nb[dim];
    }, "Get tensor stride in bytes", py::arg("tensor"), py::arg("dim"));

    m.def("tensor_type", [](void* tensor) {
        return static_cast<ggml_type>(static_cast<ggml_tensor*>(tensor)->type);
    }, "Get tensor type", py::arg("tensor"));

    m.def("nbytes", [](void* tensor) {
        return ggml_nbytes(static_cast<ggml_tensor*>(tensor));
    }, "Get tensor size in bytes", py::arg("tensor"));

    m.def("nelements", [](void* tensor) {
        return ggml_nelements(static_cast<ggml_tensor*>(tensor));
    }, "Get number of elements in tensor", py::arg("tensor"));

    // Graph input/output marking
    m.def("set_input", [](void* tensor) {
        ggml_set_input(static_cast<ggml_tensor*>(tensor));
    }, "Mark tensor as graph input (allocated at graph start in non-overlapping addresses)", py::arg("tensor"));

    m.def("set_output", [](void* tensor) {
        ggml_set_output(static_cast<ggml_tensor*>(tensor));
    }, "Mark tensor as graph output (never freed or overwritten during computation)", py::arg("tensor"));
}
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>

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

    m.def("backend_cpu_init", []() { return static_cast<void*>(ggml_backend_cpu_init()); }, "Initialize CPU backend");
    m.def("backend_is_cpu", [](void* backend) { return ggml_backend_is_cpu(static_cast<ggml_backend_t>(backend)); }, "Check if backend is CPU backend", py::arg("backend"));
    m.def("tensor_overhead", &ggml_tensor_overhead, "Get the memory overhead of a tensor");
    m.def("graph_overhead", &ggml_graph_overhead, "Get the memory overhead of a graph");
    m.def("context_init", [](size_t mem_size, void* mem_buffer = nullptr, bool no_alloc = false) {
        ggml_init_params params = {mem_size, mem_buffer, no_alloc};
        return static_cast<void*>(ggml_init(params));
    }, "Initialize GGML context", py::arg("mem_size"), py::arg("mem_buffer") = nullptr, py::arg("no_alloc") = false);
    m.def("new_tensor_2d", [](void* ctx, ggml_type type, int64_t ne0, int64_t ne1) {
        return static_cast<void*>(ggml_new_tensor_2d(
            static_cast<ggml_context*>(ctx),
            type,
            ne0,
            ne1
        ));
    }, "Create a new 2D tensor", py::arg("ctx"), py::arg("type"), py::arg("ne0"), py::arg("ne1"));
    m.def("type_size", &ggml_type_size, "Get size in bytes for all elements in a block of the given type", py::arg("type"));
    m.def("blck_size", &ggml_blck_size, "Get block size (number of elements per block) for the given type", py::arg("type"));
    m.def("backend_alloc_ctx_tensors", [](void* ctx, void* backend) {
        return static_cast<void*>(ggml_backend_alloc_ctx_tensors(
            static_cast<ggml_context*>(ctx),
            static_cast<ggml_backend_t>(backend)
        ));
    }, "Allocate all tensors in a GGML context to a backend", py::arg("ctx"), py::arg("backend"));
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
}
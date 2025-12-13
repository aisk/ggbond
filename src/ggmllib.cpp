#include <pybind11/pybind11.h>
#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-backend.h>

namespace py = pybind11;

void* backend_cpu_init() {
    return ggml_backend_cpu_init();
}

PYBIND11_MODULE(ggml, m) {
    m.doc() = "Simple and naive GGML binding";

    m.def("backend_cpu_init", &backend_cpu_init, "Initialize CPU backend");
}
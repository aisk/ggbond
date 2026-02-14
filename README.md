# GGBond

![logo](https://www.totallicensing.com/wp-content/uploads/2023/07/Winsing-to-use-1280x640.jpg)

Simple and naive GGML Python binding via pybind11. Working in progress.

## Installation

```bash
pip install -e .
```

## Requirements

- Python 3.10+
- CMake 3.15+
- C++17 compatible compiler

## Quick Start

```python
import numpy as np
from ggbond import ggml

# Phase 1: Define tensor metadata and build computation graph
ctx_model = ggml.context_init(ggml.tensor_overhead() * 2, no_alloc=True)
ctx_graph = ggml.context_init(
    ggml.tensor_overhead() * ggml.DEFAULT_GRAPH_SIZE() + ggml.graph_overhead(),
    no_alloc=True,
)

a = ggml.new_tensor_2d(ctx_model, ggml.Type.F32, 2, 4)
b = ggml.new_tensor_2d(ctx_model, ggml.Type.F32, 2, 3)

graph = ggml.new_graph(ctx_graph)
result = ggml.mul_mat(ctx_graph, a, b)
ggml.build_forward_expand(graph, result)

# Phase 2: Allocate memory on backend
backend = ggml.backend_cpu_init()
buffer = ggml.backend_alloc_ctx_tensors(ctx_model, backend)
allocr = ggml.gallocr_new(ggml.backend_get_default_buffer_type(backend))
ggml.gallocr_reserve(allocr, graph)
ggml.gallocr_alloc_graph(allocr, graph)

# Phase 3: Set data, compute, and get result
matrix_a = np.array([[2, 8], [5, 1], [4, 2], [8, 6]], dtype=np.float32)
matrix_b = np.array([[10, 5], [9, 9], [5, 4]], dtype=np.float32)
ggml.backend_tensor_set(a, matrix_a.flatten(), 0, ggml.nbytes(a))
ggml.backend_tensor_set(b, matrix_b.flatten(), 0, ggml.nbytes(b))

ggml.backend_graph_compute(backend, graph)

out = np.empty(ggml.nelements(result), dtype=np.float32)
ggml.backend_tensor_get(result, out, 0, ggml.nbytes(result))
print(out.reshape(ggml.tensor_ne(result, 1), ggml.tensor_ne(result, 0)))

# Cleanup
ggml.gallocr_free(allocr)
ggml.context_free(ctx_model)
ggml.backend_buffer_free(buffer)
ggml.backend_free(backend)
ggml.context_free(ctx_graph)
```

## Examples

See `examples/simple_backend.py` for a complete working example.

## License

MIT

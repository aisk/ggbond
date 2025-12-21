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
from ggbond import ggml

# Create context
ctx = ggml.context_init(mem_size=10*1024*1024)

# Create tensors
a = ggml.new_tensor_2d(ctx, ggml.Type.F32, 4, 2)
b = ggml.new_tensor_2d(ctx, ggml.Type.F32, 3, 4)

# Create graph and compute matrix multiplication
graph = ggml.new_graph(ctx)
result = ggml.mul_mat(ctx, a, b)
ggml.build_forward_expand(graph, result)
ggml.graph_compute_with_ctx(ctx, graph, 1)

# Cleanup
ggml.context_free(ctx)
```

## Examples

See `examples/simple_backend.py` for a complete working example.

## License

MIT

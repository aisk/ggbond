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
import ggbond
from ggbond import ggml

matrix_a = np.array([[2, 8], [5, 1], [4, 2], [8, 6]], dtype=np.float32)
matrix_b = np.array([[10, 5], [9, 9], [5, 4]], dtype=np.float32)
rows_a, cols_a = matrix_a.shape
rows_b, cols_b = matrix_b.shape

with ggbond.Session("cpu") as s:
    ctx_model = s.context(n_tensors=2)
    a = ctx_model.new_tensor(ggml.Type.F32, cols_a, rows_a)
    b = ctx_model.new_tensor(ggml.Type.F32, cols_b, rows_b)
    s.alloc(ctx_model)
    s.set(a, matrix_a)
    s.set(b, matrix_b)

    ctx_graph = s.graph_context()
    result = ggml.mul_mat(ctx_graph.raw, a, b)
    graph = ctx_graph.new_graph()
    ggml.build_forward_expand(graph, result)

    s.reserve(graph)
    s.run(graph)
    print(s.get(graph, result).reshape(3, 4))
```

## Examples

- [`examples/simple.py`](examples/simple.py) — Matrix multiplication
- [`examples/magika.py`](examples/magika.py) — File type detection with Magika
- [`examples/gpt2.py`](examples/gpt2.py) — GPT-2 text generation

## License

MIT

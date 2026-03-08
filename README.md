# GGBond

![logo](https://www.totallicensing.com/wp-content/uploads/2023/07/Winsing-to-use-1280x640.jpg)

*GGBond* stands for **GG**ML **Bo**u**nd**ing, is a simple and naive GGML Python binding via pybind11. Working in progress.

## Installation

```bash
pip install -e .
```

Enable HIP backend at build time:

```bash
CMAKE_ARGS="-DGGML_HIP=ON" pip install -e .
```

## Requirements

- Python 3.10+
- CMake 3.15+
- C++17 compatible compiler
- ROCm/HIP toolchain (only for `hip` backend)

## Architecture

GGBond provides three layers of abstraction:

### Layer 1: `ggbond.ggml`, Low-level bindings

A near 1:1 mapping of the GGML C API. Functions follow GGML naming conventions (`new_tensor_2d`, `mul_mat`, `backend_graph_compute`, etc.). All GGML opaque pointers are wrapped as distinct Python types (`ggml.Context`, `ggml.Tensor`, `ggml.Backend`, etc.) for type safety. Use this layer when you need full control over the GGML computation model.

### Layer 2: OO wrappers, `Backend`, `Context`, `Graph`, `GAllocr`

Object-oriented wrappers around GGML primitives with lifecycle management (`close()` / context manager). They simplify common patterns but are **not** a complete 1:1 equivalent of the raw API — for example, `Graph` internally owns its own `Context` for graph operations, and `Context` computes memory size from `n_tensors` automatically. These are primarily used as building blocks for the higher-level API.

### Layer 3: `Session` + `Tensor`, High-level API

`Session` owns a backend and manages all resource lifetimes. `Tensor` is a lazy-evaluated tensor bound to a session — operations build a computation graph, which is materialized on `compute()` or `numpy()`. GGUF model weights are loaded directly onto the target backend (CPU/Metal/HIP) without intermediate copies.

Supported backends:
- `cpu` (always available)
- `metal` (macOS only)
- `hip` (requires build with `GGML_HIP=ON`; alias: `rocm`)

## Quick Start

```python
import numpy as np
import ggbond

matrix_a = np.array([[2, 8], [5, 1], [4, 2], [8, 6]], dtype=np.float32)
matrix_b = np.array([[10, 5], [9, 9], [5, 4]], dtype=np.float32)

s = ggbond.Session("cpu")
a = s.tensor(matrix_a)
b = s.tensor(matrix_b)
print((a @ b).numpy())
s.close()
```

## Examples

- [`examples/simple.py`](examples/simple.py) — Matrix multiplication
- [`examples/magika.py`](examples/magika.py) — File type detection with Magika
- [`examples/gpt2.py`](examples/gpt2.py) — GPT-2 text generation

## License

MIT

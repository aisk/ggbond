# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GGBond is a Python binding library for GGML (Georgi Gerganov's Machine Learning library), built using pybind11. It provides a Pythonic interface to GGML's tensor operations and neural network computation capabilities.

- **Languages**: Python 3.10+ with C++17 backend
- **Build System**: CMake with scikit-build-core as Python build backend
- **Core Dependency**: vendor/ggml (GGML v0.9.4 submodule)

## Development Commands

### Building and Installation
```bash
# Development install (rebuilds C++ extension on changes)
pip install -e .

# Standard install
pip install .
```

### Testing
```bash
# Quick smoke test
python examples/simple.py

# GPT-2 inference (requires model file)
python examples/gpt2.py
```

## Architecture

GGBond provides three layers of abstraction:

### Layer 1: `ggbond.ggml` — Low-level C bindings

A near 1:1 mapping of the GGML C API exposed via pybind11 (`src/ggmllib.cpp`).

- GGML opaque pointers are wrapped in `TypedPtr<Tag>` template (e.g., `ContextPtr`, `TensorPtr`, `BackendPtr`) for type safety — each becomes a distinct Python type (`ggml.Context`, `ggml.Tensor`, etc.)
- Use `as<T>(ptr)` helper to extract the underlying C pointer from a `TypedPtr`
- `REGISTER_PTR` macro registers each pointer type with `__repr__` and `__bool__`
- Enums (`ggml_type`, `ggml_status`) are bound as Python enums
- Metal backend functions are behind `#ifdef __APPLE__` — not available on Linux
- Functions follow GGML naming conventions (e.g., `new_tensor_2d`, `mul_mat`)
- **ggbond/ggml.pyi**: Type stubs — **must be updated** whenever new bindings are added

### Layer 2: OO wrappers — `Backend`, `Context`, `Graph`, `GAllocr`

Object-oriented wrappers (`ggbond/backend.py`, `context.py`, `graph.py`) around GGML primitives with lifecycle management (`close()` / context manager). They simplify common patterns but are **not** a complete 1:1 equivalent of the raw API:

- **`Backend`** (`backend.py`): Wraps CPU/Metal/CUDA/HIP backend init, provides `alloc_ctx()`, `compute()`, `tensor_set()`, `tensor_get()`
- **`Context`** (`context.py`): Wraps `ggml_context` with auto-sized memory from `n_tensors`, provides `new_tensor()` dispatching to `new_tensor_{1..4}d`
- **`Graph`** (`graph.py`): Owns its own `Context` for graph ops, exposes convenience methods (`g.add(a, b)`) and direct access via `g.ctx`
- **`GAllocr`** (`graph.py`): Wraps `ggml_gallocr` for graph memory allocation

These are primarily building blocks for the higher-level API.

### Layer 3: `Session` + `Tensor` — High-level API

The recommended entry point for most use cases.

- **`Session`** (`session.py`): Owns a backend and manages all resource lifetimes (contexts, buffers). Creates tensors via `s.tensor(data)` and loads GGUF models via `s.load_gguf(path)`.
- **`Tensor`** (`tensor.py`): Lazy-evaluated tensor bound to a session. Operations (`+`, `-`, `*`, `/`, `@`, `.relu()`, `.softmax()`, etc.) build a computation graph. Materialized on `.compute()` or `.numpy()`.

```python
s = ggbond.Session("cpu")
a = s.tensor(np.array([[1, 2], [3, 4]], dtype=np.float32))
b = s.tensor(np.array([[5, 6], [7, 8]], dtype=np.float32))
print((a @ b).numpy())
s.close()
```

Key implementation details:
- `Tensor._from_ggml()`: Creates leaf nodes backed by backend-resident ggml tensors
- `Tensor._from_op()`: Creates lazy op nodes that record the operation + inputs
- `Tensor.compute()`: Topological sorts the DAG, builds a `Graph`, allocates via `GAllocr`, executes on the backend
- Shape is stored in GGML order `(ne0, ne1, ...)`, `.numpy()` reverses it automatically
- GGUF model weights are loaded directly onto the target backend without intermediate copies

### Key Files

| File | Purpose |
|------|---------|
| `src/ggmllib.cpp` | pybind11 bindings (C++ → Python) |
| `ggbond/__init__.py` | Package init, exports `Session`, `Tensor`, `ggml` |
| `ggbond/session.py` | High-level `Session` class |
| `ggbond/tensor.py` | Lazy `Tensor` with operator overloading |
| `ggbond/backend.py` | `Backend` wrapper (CPU/Metal/CUDA/HIP) |
| `ggbond/context.py` | `Context` wrapper |
| `ggbond/graph.py` | `Graph` and `GAllocr` wrappers |
| `ggbond/gguf.py` | GGUF model file loading |
| `ggbond/ggml.pyi` | Type stubs for the C extension |
| `vendor/ggml/` | GGML library source (git submodule) |

### GGML Computation Model

GGML uses a **two-phase computation model** (relevant for Layer 1 & 2 usage):

1. **Metadata Definition**: Create contexts with `no_alloc=True`, define tensor shapes and build computation graph
2. **Allocation & Execution**: Init backend → allocate memory via `GAllocr` or `backend_alloc_ctx_tensors` → set data → compute → get results

The `Tensor` high-level API (Layer 3) handles both phases automatically.

## Adding New GGML Bindings

1. **Add binding in src/ggmllib.cpp**
2. **Update ggbond/ggml.pyi**: Add corresponding type stub
3. **Rebuild**: `pip install -e .`
4. If the op should be available on `Tensor`, add it to `tensor.py` (`_UNARY_OPS`, `_BINARY_OPS`, or as a method)

## Adding New Tensor Operations

1. Add the ggml dispatch in `_materialize_op()` in `tensor.py`
2. Add shape inference in `_infer_shape()` if the output shape differs from input
3. Add a method on `Tensor` class (or add to `_UNARY_OPS`/`_BINARY_OPS`/`_REDUCTION_OPS` tuples for simple ops)

## Backends

- **CPU**: `Backend("cpu")` or `Session("cpu")` — always available
- **Metal**: `Backend("metal")` or `Session("metal")` — macOS only, GPU acceleration
- **CUDA**: `Backend("cuda")` or `Session("cuda")` — requires build with `GGML_CUDA=ON`, NVIDIA GPU
- **HIP**: `Backend("hip")` / `Backend("rocm")` or `Session("hip")` — requires build with `GGML_HIP=ON`

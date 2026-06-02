# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GGBond is an object-oriented Python wrapper over GGML (Georgi Gerganov's Machine Learning library). The OO layer (`ggbond.ggml`) is implemented in **pure Python on top of [`ggml-python`](https://github.com/abetlen/ggml-python)** (a ctypes binding to GGML) — there is no C/C++ source, no compiled extension, and no GGML submodule in this repo. The actual GGML shared library ships inside the `ggml-python` dependency.

- **Language**: Pure Python 3.10+
- **Build backend**: `hatchling` (pure-Python wheel, no compilation step)
- **Core dependency**: `ggml-python>=0.0.38` (provides the `ggml` ctypes module and `ggml.utils`)

> ⚠️ **Branch state — read this first.** The current branch (`refactor/oo-ggml-wrapper`) replaced the previous design. The commit `♻️ Reimplement OO ggml wrapper on top of ggml-python` **deleted** `src/ggmllib.cpp`, `CMakeLists.txt`, `ggbond/ggml.pyi`, `.gitmodules`, and the `vendor/ggml` submodule. Several things in the tree are now **stale and broken**:
> - The top-level modules `ggbond/session.py`, `ggbond/tensor.py`, `ggbond/backend.py`, `ggbond/context.py`, `ggbond/graph.py`, `ggbond/gguf.py` are leftovers from the old pybind11-based `Session`/`Tensor` design. They reference a removed C-mapping API (`ggml.Type`, `ggml.n_dims`, `ggml.time_init`, …) and **fail to import**. There is no `Session` class anymore.
> - `README.md` still describes the old "three-layer / pybind11 / CMake" design and a `ggbond.Session` API — it is **out of date**.
> - The examples `simple.py`, `gpt2.py`, `magika.py`, `clip.py`, `tensor_demo.py` use the old `Session` API and **do not run**. Only `examples/simple_ggbond.py` uses the current API.
>
> When asked to "fix" or "continue" work here, the likely intent is to either delete the stale modules/examples or port them onto the new `ggbond.ggml` OO API. Confirm before assuming.

## Development Commands

```bash
# Editable install (no compile step — pure Python). uv is the project's tool of choice (uv.lock present).
uv pip install -e .          # or: pip install -e .

# Smoke test the current API
.venv/bin/python examples/simple_ggbond.py            # or: uv run examples/simple_ggbond.py
.venv/bin/python examples/simple_ggbond.py -b cuda    # select backend (cpu|cuda|metal|hip)
```

There is no test suite, linter config, or CI in the repo. `examples/simple_ggbond.py` is the de-facto smoke test.

### Backend availability
Backend init dispatches in `ggbond/ggml/backend.py` (`_init_backend`). Whether `cuda`/`metal`/`hip` actually work depends on **how the installed `ggml-python` wheel was built** — this repo no longer builds GGML, so the old `CMAKE_ARGS="-DGGML_CUDA=ON"` instructions in the README no longer apply here. `cpu` is always available. HIP is guarded with a `hasattr` check and raises a clear error if absent.

## Architecture

Everything live lives under **`ggbond/ggml/`** — a thin OO wrapper whose classes map to GGML concepts and whose methods map ~1:1 to `ggml_*` C functions. Computation is **explicit and two-phase; nothing is lazy** (unlike the old `Tensor` design).

| Module | Class | Wraps |
|--------|-------|-------|
| `ggbond/ggml/_ffi.py` | — | **The only module that imports the top-level `ggml` (ggml-python) package**, re-exported as `_ggml` / `_utils`. Also defines `GGMLError`, `check_status`, `nonnull`. All other modules import the bindings *from here*, never `import ggml` directly — this avoids the name clash between the site-packages `ggml` and the `ggbond.ggml` subpackage. |
| `ggbond/ggml/types.py` | `DType` | Reuses `ggml.utils.GGML_TYPE` directly (so values never drift). Aliases `F32`, `F16`, `I32`, … and `dtype_from_numpy` / `dtype_to_numpy`. |
| `ggbond/ggml/context.py` | `Context` | `ggml_context`. Either `mem_size` (bytes) or `n_tensors`/`n_graphs` (auto-sizes metadata overhead; requires `no_alloc=True`). Factories: `new_tensor(dtype, *ne_dims)`, `tensor_from_numpy(x)` (needs `no_alloc=False`), `new_graph()`. |
| `ggbond/ggml/tensor.py` | `Tensor` | A `(Context, ggml_tensor_p)` pair. Op methods return new `Tensor`s in the same context. Simple ops are table-generated from `_UNARY_OPS` / `_BINARY_OPS`; scalar/shape ops (`scale`, `norm`, `reshape`, `permute`, `view_*`) are explicit methods. `numpy()` / `set_numpy()` take an optional `backend` (CPU → zero-copy view; non-CPU → copy via `ggml_backend_tensor_get/set`). |
| `ggbond/ggml/graph.py` | `Graph` | `ggml_cgraph`. Holds a strong ref to its backing `Context`; no independent `close()` (freed with the context). `build_forward_expand()`, `compute_with_ctx()` (CPU-only convenience), node iteration via struct fields. |
| `ggbond/ggml/gallocr.py` | `GAllocr` | `ggml_gallocr`. `from_backend(be)`, `reserve()`, `alloc_graph()`. |
| `ggbond/ggml/backend.py` | `Backend` | CPU/CUDA/Metal/HIP handle. Tracks buffers it allocates and frees them on `close()`. `alloc_ctx_tensors()`, `tensor_set/get()`, `compute(graph)`, `synchronize()`. |

`ggbond/__init__.py` re-exports the subpackage as `ggbond.ggml` plus `Context, Tensor, Backend, Graph, GAllocr, DType`.

### Canonical computation flow

GGML is two-phase. The pattern (see `examples/simple_ggbond.py`):

```python
from ggbond.ggml import Context, Backend, GAllocr, F32

with Backend("cpu", n_threads=4) as be:
    with Context(n_tensors=16, n_graphs=1, no_alloc=True) as ctx:
        ta = ctx.new_tensor(F32, *reversed(a.shape), name="a")  # ggml order = reversed numpy shape
        tc = tb.mul_mat(ta)                                     # build metadata graph
        be.alloc_ctx_tensors(ctx)                               # allocate backend memory
        ta.set_numpy(a, backend=be)                             # upload
        graph = ctx.new_graph().build_forward_expand(tc)
        with GAllocr.from_backend(be) as alloc:
            alloc.alloc_graph(graph)                            # allocate working memory
            be.compute(graph); be.synchronize()
        result = tc.numpy(backend=be)                           # download
```

### Shape convention (frequent source of bugs)
- `Tensor.ne` is **GGML order** `(ne0, ne1, …)` — column-major-ish, the reverse of numpy.
- `Tensor.shape` is **numpy order** (reverse of `ne`), matching `.numpy()`.
- `new_tensor`/`reshape` take dims in **GGML order**, so construct from numpy with `*reversed(arr.shape)`.
- `a.mul_mat(b)` requires `a.ne[0] == b.ne[0]` (shared inner dim). To compute numpy `a @ b`, call `b.mul_mat(a)` (see the example's comment).

### Lifecycle rules
- A `Context` must outlive every `Tensor`/`Graph` it produced — `ggml_free` releases all their metadata at once. `Tensor` and `Graph` hold strong refs back to their `Context` to enforce this; don't break those refs.
- All wrappers are context managers and have idempotent `close()` + `__del__`. Prefer `with` blocks.

## Adding / changing ops

1. Find the `ggml_*` function name in the `ggml-python` bindings (it must already be exposed by ggml-python — this repo cannot add new C bindings).
2. For a simple element-wise unary or binary op, add an entry to `_UNARY_OPS` / `_BINARY_OPS` in `ggbond/ggml/tensor.py` (`name -> "ggml_funcname"`); it is generated onto the class automatically.
3. For ops with scalar params or non-trivial signatures, add an explicit method that calls `_ggml.ggml_xxx(self._c, self.ptr, …)` and returns `self._wrap(ptr)`.
4. Access the underlying ctypes function via `_ggml` (imported from `._ffi`), never `import ggml` directly.

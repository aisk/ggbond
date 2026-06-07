# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GGBond is an object-oriented Python wrapper over GGML (Georgi Gerganov's Machine Learning library). The OO layer (`ggbond.ggml`) is implemented in **pure Python on top of [`ggml-python`](https://github.com/abetlen/ggml-python)** (a ctypes binding to GGML) — there is no C/C++ source, no compiled extension, and no GGML submodule in this repo. The actual GGML shared library ships inside the `ggml-python` dependency.

- **Language**: Pure Python 3.10+
- **Build backend**: `hatchling` (pure-Python wheel, no compilation step)
- **Dependencies**: `ggml-python>=0.0.38` (provides the `ggml` ctypes module and `ggml.utils`) and `numpy` (the data-transfer API is numpy-based).

> ⚠️ **Stale tree — read this first.** A past refactor (`♻️ Reimplement OO ggml wrapper on top of ggml-python`) replaced the previous pybind11/CMake design but left its files in place. The following are **stale and broken**:
> - **Top-level modules** `ggbond/session.py`, `tensor.py`, `backend.py`, `context.py`, `graph.py`, `gguf.py` are leftovers from the old `Session`/`Tensor` design. They reference a removed C-mapping API and **fail to import** (verify with `python -c "import ggbond.session"`). There is no `Session` class anymore. The *live* code is the same-named files under `ggbond/ggml/`.
> - **`README.md`** still describes the old "pybind11 / CMake / three-API" design, `CMAKE_ARGS="-DGGML_CUDA=ON"` build flags, and a `ggbond.Session` API — all **out of date**.
> - **Examples** `gpt2.py`, `magika.py`, `clip.py`, `tensor_demo.py` use the old `Session` API and **do not run**. Only `examples/simple.py` uses the current API (it is the de-facto smoke test).
>
> When asked to "fix" or "continue" work here, the likely intent is to delete the stale modules/examples or port them onto the `ggbond.ggml` OO API. Confirm before assuming.

## Development Commands

```bash
# Editable install (no compile step — pure Python). uv is the project's tool of choice (uv.lock present).
uv pip install -e .          # or: pip install -e .

# Smoke test the current API
.venv/bin/python examples/simple.py            # or: uv run examples/simple.py
.venv/bin/python examples/simple.py -b cuda    # select backend (cpu|cuda|metal|hip)
```

There is no test suite, linter config, or CI. `examples/simple.py` is the only runnable example and the de-facto smoke test.

### Backend availability
Backend init dispatches in `ggbond/ggml/backend.py` (`_init_backend`). Whether `cuda`/`metal`/`hip` actually work depends on **how the installed `ggml-python` wheel was built** — this repo no longer builds GGML, so the old `CMAKE_ARGS` instructions in the README no longer apply. `cpu` is always available. HIP is guarded with a `hasattr` check and raises a clear error if absent.

## Architecture

Everything live lives under **`ggbond/ggml/`** — a thin OO wrapper whose classes map to GGML concepts and whose methods map ~1:1 to `ggml_*` / `gguf_*` C functions. Computation is **explicit and two-phase; nothing is lazy**.

| Module | Class | Wraps / Notes |
|--------|-------|---------------|
| `_ffi.py` | — | **The only module that imports the top-level `ggml` (ggml-python) package**, re-exported as `_ggml` / `_utils`. Also defines `GGMLError`, `check_status`, `nonnull`. All other modules import the bindings *from here*, never `import ggml` directly — this avoids the name clash between the site-packages `ggml` and the `ggbond.ggml` subpackage. |
| `types.py` | `DType` | Reuses `ggml.utils.GGML_TYPE` directly (values never drift). Aliases `F32`, `F16`, `I32`, … and `dtype_from_numpy` / `dtype_to_numpy`. |
| `context.py` | `Context` | `ggml_context`. Constructor takes `mem_size` (bytes, positional) + `no_alloc=True` / `mem_buffer`. Size it for tensor metadata + graphs (and tensor data when `no_alloc=False`); `examples/simple.py` uses `1 << 20`. Factories: `new_tensor_1d/2d/3d/4d(dtype, *ne, name=…)`, `new_graph()`. `from_ptr()` wraps an externally created context (e.g. the one `gguf_init_from_file` fills in). |
| `tensor.py` | `Tensor` | A `(Context, ggml_tensor_p)` pair. Op methods return new `Tensor`s in the same context. Simple ops are table-generated from `_UNARY_OPS` / `_BINARY_OPS`; scalar/shape ops (`scale`, `norm`, `rms_norm`, `reshape_*`, `permute`, `view_*`) are explicit methods. **No numpy methods** — data transfer is done via the `Backend` (see below). |
| `graph.py` | `Graph` | `ggml_cgraph`. Holds a strong ref to its backing `Context`; no independent `close()` (freed with the context). `build_forward_expand()`, `compute_with_ctx()` (CPU-only convenience for `no_alloc=False`), node iteration via struct fields. |
| `gallocr.py` | `GAllocr` | `ggml_gallocr`. `from_backend(be)`, `reserve()`, `alloc_graph()`. |
| `backend.py` | `Backend` | CPU/CUDA/Metal/HIP handle. Tracks buffers it allocates and frees them on `close()`. `alloc_ctx_tensors()`, `tensor_set/get()`, `compute(graph)`, `synchronize()`. **This is where numpy lives**: `tensor_set` accepts any array-like (incl. plain Python lists — it runs `np.ascontiguousarray`); `tensor_get(tensor, out)` writes into a **pre-allocated numpy array** via `out.ctypes`, so reading results back requires numpy. |
| `scheduler.py` | `Scheduler` | `ggml_backend_sched`. Splits a graph across one or more backends and computes it — the multi-backend analog of `GAllocr` + `Backend.compute`. Ctor takes a sequence of `Backend`s (`from_backends(*backends)` factory), optional `bufts` (NULL ⇒ each backend's default buffer type), `graph_size`, `parallel`. `reserve()`, `alloc_graph()`, `graph_compute()` / `graph_compute_async()`, `synchronize()`, `reset()`, `get_n_splits/get_n_copies()`, `set/get_tensor_backend()`. Holds strong refs to its `Backend`s so they outlive it; `close()` frees only the `ggml_backend_sched` (the backends are owned by their own wrappers). |
| `gguf.py` | `GGUF` | `gguf_context` (parsed GGUF header/metadata). `from_file(path)` parses via `gguf_init_from_file` and, by default, also wraps the metadata `ggml_context` it fills as `.context`. **Ownership is split**: `GGUF.close()` frees only the `gguf_context`; `.context` is an independently owned `Context` you must close yourself (and that must outlive any graph referencing its tensors). `GGUF` does *not* read tensor data — `data_offset` / `get_tensor_offset` / `get_tensor_name` give the caller what they need to seek into the file and upload weights. |

`ggbond/__init__.py` re-exports the subpackage as `ggbond.ggml` plus `Context, Tensor, Backend, Graph, GAllocr, DType` (note: `GGUF` is exported from `ggbond.ggml`, not the top-level `ggbond`).

### Canonical computation flow

GGML is two-phase. The pattern (see `examples/simple.py` and the `ggbond/ggml/__init__.py` docstring):

```python
import numpy as np
from ggbond.ggml import Backend, Context, GAllocr, F32

a = np.random.rand(4, 3).astype(np.float32)
b = np.random.rand(2, 3).astype(np.float32)

with Backend("cpu", n_threads=4) as be, \
        Context(mem_size=1 << 20, no_alloc=True) as ctx:
    ta = ctx.new_tensor_2d(F32, *reversed(a.shape), name="a")  # ggml order = reversed numpy shape
    tb = ctx.new_tensor_2d(F32, *reversed(b.shape), name="b")
    tc = ta.mul_mat(tb)                                        # build metadata graph

    be.alloc_ctx_tensors(ctx)                                  # allocate backend memory for ctx tensors
    be.tensor_set(ta, a)                                       # upload (lists also accepted)
    be.tensor_set(tb, b)

    graph = ctx.new_graph().build_forward_expand(tc)
    with GAllocr.from_backend(be) as alloc:
        alloc.alloc_graph(graph)                               # allocate working memory
        be.compute(graph); be.synchronize()

    out = np.empty(tuple(reversed(tc.ne)), dtype=np.float32)   # tensor_get needs a numpy buffer
    be.tensor_get(tc, out)
```

### Shape convention (frequent source of bugs)
- `Tensor.ne` is **GGML order** `(ne0, ne1, …)` — column-major-ish, the reverse of numpy.
- `new_tensor_Nd` / `reshape_Nd` take dims in **GGML order**, so construct from a numpy array with `*reversed(arr.shape)`.
- To read a result back, allocate the output array with `np.empty(tuple(reversed(tensor.ne)), …)`.
- `a.mul_mat(b)` requires `a.ne[0] == b.ne[0]` (shared inner dim) and yields `ne = (a.ne[1], b.ne[1])`. This is GGML's `ggml_mul_mat`, **not** a plain numpy `@`; to compute numpy `a @ b` arrange it as `b.mul_mat(a)`.

### Lifecycle rules
- A `Context` must outlive every `Tensor`/`Graph` it produced — `ggml_free` releases all their metadata at once. `Tensor` and `Graph` hold strong refs back to their `Context` to enforce this; don't break those refs.
- For `GGUF`, remember the split ownership above: closing the `GGUF` does **not** close its data `.context`.
- All wrappers are context managers with idempotent `close()` + `__del__`. Prefer `with` blocks.

## Adding / changing ops

1. Find the `ggml_*` function name in the `ggml-python` bindings (it must already be exposed by ggml-python — this repo cannot add new C bindings).
2. For a simple element-wise unary or binary op, add an entry to `_UNARY_OPS` / `_BINARY_OPS` in `ggbond/ggml/tensor.py` (`name -> "ggml_funcname"`); it is generated onto the class automatically.
3. For ops with scalar params or non-trivial signatures, add an explicit method that calls `_ggml.ggml_xxx(self._c, self.ptr, …)` and returns `self._wrap(ptr)`.
4. Access the underlying ctypes functions via `_ggml` (imported from `._ffi`), never `import ggml` directly.

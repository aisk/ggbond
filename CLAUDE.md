# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GGBond is an object-oriented Python wrapper over GGML (Georgi Gerganov's Machine Learning library). The OO layer (`ggbond.ggml`) is implemented in **pure Python on top of [`ggml-python`](https://github.com/abetlen/ggml-python)** (a ctypes binding to GGML) — there is no C/C++ source, no compiled extension, and no GGML submodule in this repo. The actual GGML shared library ships inside the `ggml-python` dependency.

- **Language**: Pure Python 3.10+
- **Build backend**: `hatchling` (pure-Python wheel, no compilation step)
- **Dependencies**: `ggml-python>=0.0.45` (provides the `ggml` ctypes module and `ggml.utils`) and `numpy` (the data-transfer API is numpy-based).

All library code lives under **`ggbond/ggml/`**. Every checked-in example uses the current API: `simple.py`, `gpt2.py`, `magika.py`, and `nam.py`. The model-backed examples require external model files.

## Development Commands

```bash
# Editable install (no compile step — pure Python). uv is the project's tool of choice (uv.lock present).
uv pip install -e .          # or: pip install -e .

# Tests and smoke test
.venv/bin/python -m unittest discover -v       # or: uv run python -m unittest discover -v
.venv/bin/python examples/simple.py            # or: uv run examples/simple.py
```

The standard-library `unittest` suite lives under `tests/`. `examples/simple.py` is the numerical smoke test.

### Backend availability
Backend factories live in `ggbond/ggml/backend.py`; the GPT-2 and NAM examples provide their own `_init_backend` CLI dispatch. Whether `cuda`/`metal`/`hip` actually work depends on **how the installed `ggml-python` package was built** (this repo does not build GGML). `cpu` is always available. Unavailable optional backends raise `GGMLError`.

## Architecture

Everything live lives under **`ggbond/ggml/`** — a thin OO wrapper whose classes map to GGML concepts and whose methods map ~1:1 to `ggml_*` / `gguf_*` C functions. Computation is **explicit and two-phase; nothing is lazy**.

| Module | Class | Wraps / Notes |
|--------|-------|---------------|
| `errors.py` | — | Defines `GGMLError`, `check_status`, `nonnull`. Each module imports the bindings directly as `import ggml as _ggml` (and `from ggml import utils as _utils` in `types.py`) — Python 3 absolute imports resolve `ggml` to the site-packages package, never the `ggbond.ggml` subpackage; the `_ggml` alias keeps the external package visually distinct. |
| `types.py` | `DType` | Reuses `ggml.utils.GGML_TYPE` directly (values never drift). Aliases `F32`, `F16`, `I32`, … and `dtype_from_numpy` / `dtype_to_numpy`. |
| `context.py` | `Context` | `ggml_context`. Constructor takes `mem_size` (bytes, positional) + `no_alloc=True` / `mem_buffer`. Size it for tensor metadata + graphs (and tensor data when `no_alloc=False`); `examples/simple.py` uses `1 << 20`. Factories: `new_tensor_1d/2d/3d/4d(dtype, *ne, name=…)`, `new_graph()`. `from_ptr()` wraps an externally created context (e.g. the one `gguf_init_from_file` fills in). `ref(tensor)` rebinds a tensor created in another context to this one, which a graph must do before reading model weights or a KV cache so its result nodes land here. |
| `tensor.py` | `Tensor` | A `(Context, ggml_tensor_p)` pair. Op methods return new `Tensor`s in the same context. **All** ops are table-generated from `_UNARY_OPS` / `_BINARY_OPS` / `_PARAM_OPS` (see "Adding / changing ops"); the tables cover elementwise math, reductions, normalization, attention (`rope`, `rope_ext`, `soft_max_ext`, `flash_attn_ext`, `diag_mask_inf`), shape (`reshape_*`, `cont_*`, `view_1d`…`view_4d`, `permute`, `concat`, `pad`, `repeat*`), and convolution/pooling. `ne`/`nb` follow `ggml_n_dims` and trim trailing 1-sized dims; `ne4`/`nb4` give the untrimmed truth and `shape` is `ne` reversed (numpy order). **This is where numpy lives**: `set(x)` uploads any array-like (incl. plain Python lists — runs `np.ascontiguousarray`) and `get(out)` downloads into a **pre-allocated numpy array** via `out.ctypes`. These wrap ggml's sync transfer `ggml_backend_tensor_set/get`, which are keyed on the tensor's own buffer and take **no backend handle** — hence they live on `Tensor`, not `Backend`. The tensor must already be allocated. |
| `graph.py` | `Graph` | `ggml_cgraph`. Holds a strong ref to its backing `Context`; no independent `close()` (freed with the context). `build_forward_expand()`, `compute_with_ctx()` (CPU-only convenience for `no_alloc=False`), node iteration via struct fields. |
| `gallocr.py` | `GAllocr` | `ggml_gallocr`. `from_backend(be)`, `reserve()`, `alloc_graph()`. |
| `backend.py` | `Backend` | CPU/CUDA/Metal/HIP handle. Use `Backend.cpu_init()` / `cuda_init()` / `metal_init()` / `hip_init()`, or the device registry: `Backend.load_all()` then `Backend.init_best()` / `Backend.init_by_type(DEVICE_CPU\|DEVICE_GPU\|DEVICE_IGPU\|DEVICE_ACCEL)`. The constructor itself adopts a raw backend pointer. Tracks buffers it allocates and frees them on `close()`. Tensor data transfer is **not** here (it lives on `Tensor`, see above). |
| `buffer.py` | `Buffer`, `BufferType` | `ggml_backend_buffer_t` and `ggml_backend_buffer_type_t`. `Backend.alloc_ctx_tensors()` returns a `Buffer` (`size`, `name`, `is_host`, `clear()`); it is backend-owned by default, and `Buffer.release()` transfers ownership so the buffer's own `close()` frees it (still before the backend closes). `Backend.buffer_type()` returns a `BufferType`, which `GAllocr` and `Scheduler` also accept as a raw pointer. |
| `scheduler.py` | `Scheduler` | `ggml_backend_sched`. Splits a graph across one or more backends and computes it — the multi-backend analog of `GAllocr` + `Backend.compute`. `from_backends(*backends)` factory. Holds strong refs to its `Backend`s so they outlive it; `close()` frees only the `ggml_backend_sched` (the backends are owned by their own wrappers). |
| `gguf.py` | `GGUF` | `gguf_context` (parsed GGUF header/metadata). `from_file(path)` parses via `gguf_init_from_file` and, by default, also wraps the metadata `ggml_context` it fills as `.context`. **Ownership is split**: `GGUF.close()` frees only the `gguf_context`; `.context` is an independently owned `Context` you must close yourself (and that must outlive any graph referencing its tensors). `GGUF` does *not* read tensor data — `data_offset` / `get_tensor_offset` / `get_tensor_name` give the caller what they need to seek into the file and upload weights. Typed `get_val_*` / `set_val_*` accessors are generated from `_KV_TYPES`; `value(key)`, `keys()` and `items()` dispatch on the stored type for you. Writing is supported too: `init_empty()`, the setters, `set_arr_data` / `set_arr_str`, `add_tensor`, `set_tensor_data` / `set_tensor_type`, `write_to_file`. |

`ggbond/__init__.py` exports only `__version__`. Import all wrapper classes and constants explicitly from `ggbond.ggml`.

### Canonical computation flow

GGML is two-phase. The pattern (see `examples/simple.py` and the `ggbond/ggml/__init__.py` docstring):

```python
import numpy as np
from ggbond.ggml import Backend, Context, GAllocr, F32

a = np.random.rand(4, 3).astype(np.float32)
b = np.random.rand(2, 3).astype(np.float32)

with Backend.cpu_init() as be, \
        Context(mem_size=1 << 20, no_alloc=True) as ctx:
    be.set_n_threads(4)
    ta = ctx.new_tensor_2d(F32, *reversed(a.shape), name="a").set_input()
    tb = ctx.new_tensor_2d(F32, *reversed(b.shape), name="b").set_input()
    tc = ta.mul_mat(tb).set_output()                            # builds b @ a.T

    be.alloc_ctx_tensors(ctx)                                  # allocate backend memory for ctx tensors
    ta.set(a)                                                  # upload via the tensor (lists also accepted)
    tb.set(b)

    graph = ctx.new_graph().build_forward_expand(tc)
    with GAllocr.from_backend(be) as alloc:
        alloc.alloc_graph(graph)                               # allocate working memory
        be.compute(graph); be.synchronize()

    out = tc.get(np.empty(tuple(reversed(tc.ne)), dtype=np.float32))  # download into a numpy buffer
```

### Shape convention (frequent source of bugs)
- `Tensor.ne` is **GGML order** `(ne0, ne1, …)` — column-major-ish, the reverse of numpy.
- `new_tensor_Nd` / `reshape_Nd` take dims in **GGML order**, so construct from a numpy array with `*reversed(arr.shape)`.
- To read a result back, allocate the output array with `np.empty(tuple(reversed(tensor.ne)), …)`.
- `a.mul_mat(b)` requires `a.ne[0] == b.ne[0]` (shared inner dim) and yields `ne = (a.ne[1], b.ne[1])`. For NumPy arrays with matching column counts, `ta.mul_mat(tb)` computes `b @ a.T`. To compute a conventional `left @ right`, store `right.T` as the first GGML operand and `left` as the second.

### Lifecycle rules
- A `Context` must outlive every `Tensor`/`Graph` it produced — `ggml_free` releases all their metadata at once. `Tensor` and `Graph` keep the context alive unless it is explicitly closed. A `Context` also retains an external `mem_buffer` object for its native lifetime.
- For `GGUF`, remember the split ownership above: closing the `GGUF` does **not** close its data `.context`.
- A `GAllocr` owns the memory of every graph node it allocated. Download results **before** the allocator closes; reading afterwards silently returns garbage. (In the canonical flow above `tc` is created before `alloc_ctx_tensors`, so it lives in the backend buffer instead and can be read later.)
- All wrappers are context managers with idempotent `close()` + `__del__`. Prefer `with` blocks.

## Adding / changing ops

Every tensor op is generated from a table in `ggbond/ggml/tensor.py`; none are hand-written.

1. Find the `ggml_*` function name in the `ggml-python` bindings (it must already be exposed by ggml-python — this repo cannot add new C bindings).
2. For an op taking only the receiver, add `name -> "ggml_funcname"` to `_UNARY_OPS`. For an op taking the receiver plus one more tensor, add it to `_BINARY_OPS`.
3. For anything with extra arguments, add an entry to `_PARAM_OPS`: `name -> (ggml_funcname, params, doc)`, where each param is `(arg_name, kind)` or `(arg_name, kind, default)` and `kind` is `_TENSOR`, `_TENSOR_OPT` (None becomes NULL), `_INT`, `_FLOAT`, `_BOOL`, or `_DTYPE`. `self` is always passed as ggml's `a` argument.
4. `_make_op` generates the method source and `exec`s it, so each op gets a real signature (keyword args, defaults, `help()`) with no per-call binding overhead. Only add a hand-written method if a signature genuinely cannot be expressed by the table (nothing currently needs one).
5. Access the underlying ctypes functions via `_ggml` (each module does `import ggml as _ggml`); use the `_ggml` alias rather than a bare `import ggml` so the external package stays visually distinct from the `ggbond.ggml` subpackage.

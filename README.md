# GGBond

GGBond is an object-oriented Python wrapper over [GGML](https://github.com/ggml-org/ggml). It is implemented in pure Python on top of [`ggml-python`](https://github.com/abetlen/ggml-python), which provides the ctypes bindings and bundled GGML shared libraries.

It comes in two layers:

- **`ggbond`** — a small high-level API: a `Session` picks the devices and owns every resource, variables hold weights and caches, and a `with sess.graph()` block is one computation.
- **`ggbond.ggml`** — a 1:1 mapping of GGML's own concepts (`Context`, `Backend`, `GAllocr`, `Scheduler`, `GGUF`, …) for when you want exactly what the C API does.

Both use GGML's op names, argument order and shape order. Graphs are static and explicit: operations build a graph, then it is allocated and computed. Nothing is lazy.

## Requirements and installation

- Python 3.10+
- `ggml-python>=0.0.45`
- NumPy

```bash
uv sync
# or
pip install -e .
```

This repository has no C/C++ build step. CPU is always available; CUDA, Metal,
and HIP availability depends on how the installed `ggml-python` package was
built. CMake options passed while installing GGBond do not enable those
backends.

## Quick start

```python
import numpy as np

from ggbond import Session

a = np.array([[2, 8], [5, 1], [4, 2], [8, 6]], dtype=np.float32)
b = np.array([[10, 5], [9, 9], [5, 4]], dtype=np.float32)

# The session picks the best available device; the graph is one computation.
with Session() as sess, sess.graph() as g:
    ta = g.input(a, name="a")
    tb = g.input(b, name="b")
    result = g.run(ta.mul_mat(tb))      # allocate, upload, compute, download

np.testing.assert_allclose(result, b @ a.T)
```

## The high-level API

A model keeps its weights in variables and builds a graph per evaluation:

```python
from ggbond import F32, I32, Session

with Session(device="cpu", graph_size=4096) as sess:
    weights, gguf = sess.load_gguf("model.gguf")          # {name: Variable}, data uploaded
    cache = sess.variable(F32, 768, 1024, name="cache")   # zero-initialised, persistent

    for step in range(n_steps):
        with sess.graph() as g:
            tokens = g.input(I32, n_tokens, name="tokens")        # placeholder
            hidden = weights["wte"].get_rows(tokens)
            g.update(hidden.cpy(cache.view_1d(hidden.nelements, step * hidden.nbytes)))
            logits = weights["head"].mul_mat(hidden)
            out = g.run(logits, feed={tokens: ids})
```

- `Session(*backends, device=None, n_threads=None, graph_size=None)` opens the best device plus a CPU fallback, a named device (`Session.devices()` lists them) or a device type (`"cpu"`, `"gpu"`, `"accel"`), or uses the `ggbond.ggml.Backend`s you pass. `n_threads` defaults to the number of physical cores. Closing the session frees everything it created.
- Variables persist for the session: `sess.variables({name: (dtype, ne0, …)})`, `sess.variable(dtype, *ne)`, `sess.load_arrays({name: array})` and `sess.load_gguf(path)`. Read and write them at any time with `write()` / `to_numpy()`.
- `with sess.graph() as g:` is one computation, and one is open at a time. Tensors created inside it are gone when the block ends, so download what you need with `g.run()` first.
- `g.input(array)` is an input that already has its data; `g.input(dtype, *ne)` is a placeholder fed with `g.run(..., feed={inp: array})`. NumPy's default float64 / int64 data is stored as F32 / I32 unless you pass `dtype=`.
- Fetch contiguous tensors: for a transposed or permuted result, fetch `t.cont()`.
- `g.update(node)` adds a node that is computed for its side effect, such as a copy into a cache variable. Updates run in the order they were added.
- `g.run(*fetches)` returns NumPy arrays; calling it again recomputes, and may fetch other tensors. `g.reserve(*fetches)` pre-sizes the compute buffers for a worst-case graph.
- Ops are the GGML ops: `a.mul_mat(b)`, `g.mul_mat(a, b)`, the same operators as `Tensor`. `g.ctx`, `g.cgraph` and `sess.sched` give you the underlying `ggbond.ggml` objects.

## The 1:1 API

```python
import numpy as np

from ggbond.ggml import Backend, Context, F32, GAllocr

a = np.array([[2, 8], [5, 1], [4, 2], [8, 6]], dtype=np.float32)
b = np.array([[10, 5], [9, 9], [5, 4]], dtype=np.float32)

with Backend.cpu_init() as backend, Context(1 << 20, no_alloc=True) as ctx:
    ta = ctx.new_tensor_2d(F32, *reversed(a.shape), name="a").set_input()
    tb = ctx.new_tensor_2d(F32, *reversed(b.shape), name="b").set_input()
    out = ta.mul_mat(tb).set_output()

    graph = ctx.new_graph().build_forward_expand(out)
    with GAllocr.from_backend(backend) as allocator:
        allocator.alloc_graph(graph)
        ta.write(a)
        tb.write(b)
        backend.compute(graph)

        result = out.to_numpy()

np.testing.assert_allclose(result, b @ a.T)
```

Import these classes from `ggbond.ggml`. The rules and classes below describe this layer; the shape rules apply to both.

## Shape and lifetime rules

- Tensor dimensions are in GGML order `(ne0, ne1, ...)`, the reverse of a
  conventional NumPy shape. Use `*reversed(array.shape)` when creating tensors.
  `Tensor.ne` is the four dimensions GGML stores; `Tensor.shape` is the NumPy
  view of them.
- Every op exists twice: `ctx.mul_mat(a, b)` is `ggml_mul_mat(ctx, a, b)` with
  the target context explicit, and `a.mul_mat(b)` is sugar for
  `a.ctx.mul_mat(a, b)`. Use the context form when the first operand lives in
  another context, such as model weights.
- For two NumPy matrices whose rows share the same inner dimension,
  `ta.mul_mat(tb)` computes `b @ a.T`.
- Mark graph inputs and outputs with `set_input()` and `set_output()` before
  graph allocation.
- A `Context` owns tensor and graph metadata and must outlive everything created
  from it. Prefer context managers.
- `GGUF.close()` only closes the GGUF metadata handle. Its `.context` is owned
  separately and must also be closed.
- `Tensor.write()`, `read()` and `to_numpy()` require allocated tensor
  storage. `write()` checks the array against `Tensor.shape`; use `write_raw()`
  for quantized or otherwise non-NumPy weight data.
- Every wrapper that owns a native handle raises `GGMLError` once closed.
  Closing a `Backend` first closes the buffers, allocators and schedulers built
  on it.
- A `GAllocr` or `Scheduler` owns the memory of the graph tensors it allocated.
  Read results before closing it; afterwards those tensors are unallocated and
  reading them raises `GGMLError`.
- A `GAllocr` or `Scheduler` holds one graph at a time. Allocating a different graph unallocates the previous one's tensors, so download results before allocating the next graph and upload shared inputs again.
- `Context.reset()` empties a context for reuse, for example to build a new
  graph per generated token. Tensors and graphs created before the reset can no
  longer be used.

## Main classes

| Class | Purpose |
| --- | --- |
| `Context` | Owns GGML tensor and graph metadata; every op as `ctx.op(a, ...)` |
| `Tensor` | Wraps a tensor pointer and exposes GGML operations and transfers |
| `Graph` | Builds and inspects a computation graph |
| `Backend` | A CPU/CUDA/Metal/HIP backend handle |
| `Buffer` | Backend memory holding a context's tensor data |
| `Device` | An entry of the backend device registry |
| `GAllocr` | Allocates a graph on one backend |
| `Scheduler` | Allocates and computes a graph across multiple backends |
| `GGUF` | Reads and writes GGUF metadata and tensor entries |

## Examples

Each example exists on both layers:

| | High-level (`ggbond`) | 1:1 (`ggbond.ggml`) |
| --- | --- | --- |
| Matrix multiply | [`examples/simple.py`](examples/simple.py) | [`examples/ggml/simple.py`](examples/ggml/simple.py) |
| Legacy GGML GPT-2 inference | [`examples/gpt2.py`](examples/gpt2.py) | [`examples/ggml/gpt2.py`](examples/ggml/gpt2.py) |
| Magika GGUF file classification | [`examples/magika.py`](examples/magika.py) | [`examples/ggml/magika.py`](examples/ggml/magika.py) |
| Offline NAM A1 WaveNet inference | [`examples/nam.py`](examples/nam.py) | [`examples/ggml/nam.py`](examples/ggml/nam.py) |

In the high-level GPT-2 and NAM examples `--backend` / `-b` is a device name or type as accepted by `Session(device=…)` and defaults to the best available device; `-t` defaults to the number of physical cores. The 1:1 versions take `cpu`, `cuda`, `metal` or `hip`. Model files are not included.

## Development

```bash
uv run python -m unittest discover -v
uv run examples/simple.py
uv run examples/ggml/simple.py
uv run scripts/gen_stubs.py   # after changing an op table
uv build
```

## License

MIT

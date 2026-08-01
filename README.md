# GGBond

GGBond is a thin, object-oriented Python wrapper over
[GGML](https://github.com/ggml-org/ggml). It is implemented in pure Python on
top of [`ggml-python`](https://github.com/abetlen/ggml-python), which provides
the ctypes bindings and bundled GGML shared libraries.

The API is explicit and two-phase: tensor operations build a graph, then a
backend allocator allocates and computes it. Nothing is lazy.

## Requirements and installation

- Python 3.10+
- `ggml-python>=0.0.38`
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
        ta.set(a)
        tb.set(b)
        backend.compute(graph)
        backend.synchronize()

        result = out.get(
            np.empty(tuple(reversed(out.ne)), dtype=np.float32)
        )

np.testing.assert_allclose(result, b @ a.T)
```

Import wrapper classes from `ggbond.ggml`; the top-level `ggbond` package only
exposes `__version__`.

## Shape and lifetime rules

- Tensor dimensions are in GGML order `(ne0, ne1, ...)`, the reverse of a
  conventional NumPy shape. Use `*reversed(array.shape)` when creating tensors.
- For two NumPy matrices whose rows share the same inner dimension,
  `ta.mul_mat(tb)` computes `b @ a.T`.
- Mark graph inputs and outputs with `set_input()` and `set_output()` before
  graph allocation.
- A `Context` owns tensor and graph metadata and must outlive everything created
  from it. Prefer context managers.
- `GGUF.close()` only closes the GGUF metadata handle. Its `.context` is owned
  separately and must also be closed.
- `Tensor.set()` and `Tensor.get()` require allocated tensor storage. Use
  `set_raw()` for quantized or otherwise non-NumPy weight data.

## Main classes

| Class | Purpose |
| --- | --- |
| `Context` | Owns GGML tensor and graph metadata |
| `Tensor` | Wraps a tensor pointer and exposes GGML operations and transfers |
| `Graph` | Builds and inspects a computation graph |
| `Backend` | Owns a CPU/CUDA/Metal/HIP backend and context buffers |
| `GAllocr` | Allocates a graph on one backend |
| `Scheduler` | Allocates and computes a graph across multiple backends |
| `GGUF` | Reads GGUF metadata and tensor offsets |

## Examples

- [`examples/simple.py`](examples/simple.py) — scheduler-based matrix multiply
- [`examples/gpt2.py`](examples/gpt2.py) — legacy GGML GPT-2 inference
- [`examples/magika.py`](examples/magika.py) — Magika GGUF file classification
- [`examples/nam.py`](examples/nam.py) — offline NAM A1 WaveNet inference

GPT-2 and NAM accept `--backend` / `-b` (`cpu`, `cuda`, `metal`, or `hip`).
Magika currently uses the CPU backend. Model files are not included.

## Development

```bash
uv run python -m unittest discover -v
uv run examples/simple.py
uv build
```

## License

MIT

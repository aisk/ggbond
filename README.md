# GGBond

![logo](https://www.totallicensing.com/wp-content/uploads/2023/07/Winsing-to-use-1280x640.jpg)

*GGBond* stands for **GG**ML **Bo**u**nd**ing: an object-oriented Python wrapper over [GGML](https://github.com/ggml-org/ggml), written in pure Python on top of [`ggml-python`](https://github.com/abetlen/ggml-python).

It has two layers. `ggbond` is a small high-level API where a `Session` owns the devices and every resource. `ggbond.ggml` maps GGML's own concepts (`Context`, `Backend`, `GAllocr`, `Scheduler`, `GGUF`, …) 1:1. Both keep GGML's op names, argument order and shape order.

## Installation

```bash
pip install -e .   # or: uv sync
```

Requires Python 3.10+. There is no build step; CPU always works, and GPU backends depend on how your `ggml-python` was built.

## High-level API

```python
import numpy as np
from ggbond import Session

a = np.array([[2, 8], [5, 1], [4, 2], [8, 6]], dtype=np.float32)
b = np.array([[10, 5], [9, 9], [5, 4]], dtype=np.float32)

with Session() as sess, sess.graph() as g:   # best available device
    ta = g.input(a, name="a")
    tb = g.input(b, name="b")
    result = g.run(ta.mul_mat(tb))           # allocate, upload, compute, download

np.testing.assert_allclose(result, b @ a.T)
```

## 1:1 API

The same computation with every GGML step spelled out:

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

Tensor dimensions are in GGML order, the reverse of a NumPy shape, which is why `ta.mul_mat(tb)` is `b @ a.T`.

## More examples

[`examples/`](examples) has matrix multiplication, GPT-2, Magika and NAM inference on the high-level API, and [`examples/ggml/`](examples/ggml) has the same programs on the 1:1 layer. Model files are not included.

## License

MIT

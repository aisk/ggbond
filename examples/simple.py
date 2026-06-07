"""Minimal matrix-multiply demo on the ``ggbond.ggml`` OO wrapper.

Reproduces the original ``simple.py`` (a single ``mul_mat``) using the current
explicit, two-phase API: build a metadata graph, allocate backend memory,
upload inputs, compute, then download the result.
"""

import getopt
import sys

import numpy as np

from ggbond.ggml import Backend, Context, GAllocr, F32


def main():
    # Parse command line arguments
    try:
        opts, _ = getopt.getopt(sys.argv[1:], "b:", ["backend="])
    except getopt.GetoptError as err:
        print(f"Error: {err}")
        print("Usage: python simple.py [-b|--backend cpu|cuda|metal|hip]")
        sys.exit(1)

    backend_type = "cpu"
    for opt, arg in opts:
        if opt in ("-b", "--backend"):
            backend_type = arg.lower()

    a = np.array([
        [2, 8],
        [5, 1],
        [4, 2],
        [8, 6],
    ], dtype=np.float32)
    b = np.array([
        [10, 5],
        [9, 9],
        [5, 4],
    ], dtype=np.float32)

    with Backend(backend_type, n_threads=4) as be, \
            Context(mem_size=1 << 20, no_alloc=True) as ctx:
        # ggml shape is the reverse of numpy's; a.ne[0] == b.ne[0] (== 2) is the
        # shared inner dim mul_mat contracts over.
        ta = ctx.new_tensor_2d(F32, *reversed(a.shape), name="a")
        tb = ctx.new_tensor_2d(F32, *reversed(b.shape), name="b")
        tc = ta.mul_mat(tb)  # ggml_mul_mat(a, b): ne -> (a.ne1, b.ne1)

        be.alloc_ctx_tensors(ctx)
        be.tensor_set(ta, a)
        be.tensor_set(tb, b)

        graph = ctx.new_graph().build_forward_expand(tc)
        with GAllocr.from_backend(be) as alloc:
            alloc.alloc_graph(graph)
            be.compute(graph)
            be.synchronize()

        result = np.empty(tuple(reversed(tc.ne)), dtype=np.float32)
        be.tensor_get(tc, result)

    rows, cols = result.shape
    print(f"mul mat ({cols} x {rows}):")
    print(result)


if __name__ == "__main__":
    main()

"""Minimal matrix-multiply demo on the ``ggbond.ggml`` OO wrapper.

Mirrors ggml's ``simple-backend`` example: a single ``mul_mat`` computed through
a multi-backend :class:`Scheduler` built from the best available device plus a
CPU backend. The scheduler allocates the graph's tensors itself
(``reset`` -> ``alloc_graph``), so inputs are uploaded *after* ``alloc_graph``.
"""

import numpy as np

from ggbond.ggml import Backend, Context, Scheduler, DEVICE_CPU, F32


def main():
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

    # Like ggml's init_model: register all backends, then schedule over the best
    # device plus an explicit CPU backend.
    Backend.load_all()
    cpu = Backend.init_by_type(DEVICE_CPU)
    cpu.set_n_threads(4)
    backends = [Backend.init_best(), cpu]

    try:
        with Context(mem_size=1 << 20, no_alloc=True) as ctx, \
                Scheduler.from_backends(backends) as sched:
            # ggml shape is the reverse of numpy's; a.ne[0] == b.ne[0] (== 2) is
            # the shared inner dim mul_mat contracts over.
            ta = ctx.new_tensor_2d(F32, *reversed(a.shape), name="a").set_input()
            tb = ctx.new_tensor_2d(F32, *reversed(b.shape), name="b").set_input()
            tc = ta.mul_mat(tb)  # ggml_mul_mat(a, b): ne -> (a.ne1, b.ne1)
            tc.set_output()

            graph = ctx.new_graph().build_forward_expand(tc)

            # The scheduler allocates the graph's tensors; upload inputs after.
            sched.reset()
            sched.alloc_graph(graph)
            ta.set(a)
            tb.set(b)

            sched.graph_compute(graph)
            sched.synchronize()

            result = tc.to_numpy()
    finally:
        for backend in backends:
            backend.close()

    np.testing.assert_allclose(result, b @ a.T)

    rows, cols = result.shape
    print(f"mul mat ({cols} x {rows}):")
    print(result)


if __name__ == "__main__":
    main()

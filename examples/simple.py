"""Minimal matrix-multiply demo on the high-level ``ggbond`` API.

Mirrors ggml's ``simple-backend`` example: a single ``mul_mat``. The session
picks the device (the best one available, with the CPU behind it); the graph
block holds one computation. ``examples/ggml/simple.py`` is the same program on
the 1:1 ggml classes.
"""

import numpy as np

from ggbond import Session


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

    # The session lives as long as the devices; the graph is one computation.
    # A program that computes more than once keeps the session and opens a
    # graph per evaluation.
    with Session() as sess, sess.graph() as g:
        # Inputs take their type and shape from the arrays. ggml's shape is
        # the reverse of numpy's; a.ne[0] == b.ne[0] (== 2) is the shared
        # inner dim mul_mat contracts over.
        ta = g.input(a, name="a")
        tb = g.input(b, name="b")
        tc = ta.mul_mat(tb)  # ggml_mul_mat(a, b): ne -> (a.ne1, b.ne1)

        result = g.run(tc)  # allocate, upload a and b, compute, download

    np.testing.assert_allclose(result, b @ a.T)

    rows, cols = result.shape
    print(f"mul mat ({cols} x {rows}):")
    print(result)


if __name__ == "__main__":
    main()

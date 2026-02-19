"""Demonstrate the high-level Tensor API."""

import numpy as np

from ggbond import Tensor, Session


def main():
    matrix_a = np.array([
        [2, 8],
        [5, 1],
        [4, 2],
        [8, 6],
    ], dtype=np.float32)

    matrix_b = np.array([
        [10, 5],
        [9, 9],
        [5, 4],
    ], dtype=np.float32)

    s = Session("metal")

    # --- basic matrix multiplication ---
    a = s.tensor(data=matrix_a)
    b = s.tensor(data=matrix_b)
    result = (a @ b).numpy()

    print("Tensor API — a @ b:")
    print(result)

    # Compare with numpy (ggml mul_mat computes a @ b^T equivalent,
    # but in GGML column-major so result matches numpy a @ b.T)
    expected = matrix_a @ matrix_b.T
    print("\nnumpy a @ b.T:")
    print(expected)
    assert np.allclose(result, expected), "mismatch!"
    print("✓ results match\n")

    # --- chained operations ---
    a = s.tensor(data=matrix_a)
    b = s.tensor(data=matrix_b)
    result = (a @ b).relu().numpy()
    print("(a @ b).relu():")
    print(result)
    assert np.all(result >= 0), "relu failed!"
    print("✓ relu correct\n")

    # --- scalar operations ---
    a = s.tensor(data=matrix_a)
    scaled = (a * 2.0).numpy()
    print("a * 2.0:")
    print(scaled)
    assert np.allclose(scaled, matrix_a * 2.0), "scale failed!"
    print("✓ scalar multiply correct\n")

    # --- arithmetic chain ---
    a = s.tensor(data=np.array([[1, 2], [3, 4]], dtype=np.float32))
    b = s.tensor(data=np.array([[5, 6], [7, 8]], dtype=np.float32))
    result = (a + b).numpy()
    print("a + b:")
    print(result)
    expected = np.array([[6, 8], [10, 12]], dtype=np.float32)
    assert np.allclose(result, expected), "add failed!"
    print("✓ element-wise add correct\n")

    print("All tests passed!")
    s.close()


if __name__ == "__main__":
    main()

import getopt
import sys

import numpy as np

from ggbond import Tensor, ggml


def main():
    # Parse command line arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:], "b:", ["backend="])
    except getopt.GetoptError as err:
        print(f"Error: {err}")
        print("Usage: python simple.py [-b|--backend cpu|metal]")
        sys.exit(1)

    backend_type = "cpu"
    for opt, arg in opts:
        if opt in ("-b", "--backend"):
            backend_type = arg.lower()

    ggml.time_init()
    ggml.log_set_default()

    matrix_a = np.array([
        [2, 8],
        [5, 1],
        [4, 2],
        [8, 6]
    ], dtype=np.float32)

    matrix_b = np.array([
        [10, 5],
        [9, 9],
        [5, 4]
    ], dtype=np.float32)

    a = Tensor(data=matrix_a)
    b = Tensor(data=matrix_b)
    result = (a @ b).numpy(backend=backend_type)

    rows, cols = result.shape
    print(f"mul mat ({cols} x {rows}):")
    print(result)


if __name__ == "__main__":
    main()

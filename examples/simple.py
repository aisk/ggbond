import getopt
import sys

import numpy as np

import ggbond
from ggbond import ggml


def main():
    # Parse command line arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:], "b:", ["backend="])
    except getopt.GetoptError as err:
        print(f"Error: {err}")
        print("Usage: python ggbond_simple_backend.py [-b|--backend cpu|metal]")
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

    rows_a, cols_a = matrix_a.shape
    rows_b, cols_b = matrix_b.shape

    with ggbond.Session(backend_type) as s:
        ctx_model = s.context(n_tensors=2)
        tensor_a = ctx_model.new_tensor(ggml.Type.F32, cols_a, rows_a)
        tensor_b = ctx_model.new_tensor(ggml.Type.F32, cols_b, rows_b)
        s.alloc(ctx_model)

        s.set(tensor_a, matrix_a)
        s.set(tensor_b, matrix_b)

        ctx_graph = s.graph_context()
        result = ggml.mul_mat(ctx_graph.raw, tensor_a, tensor_b)
        graph = ctx_graph.new_graph()
        ggml.build_forward_expand(graph, result)

        mem_size = s.reserve(graph)
        print(f"compute buffer size: {mem_size / 1024.0:.4f} KB")

        s.run(graph)

        result_tensor = ggml.graph_node(graph, -1)
        result_ne0 = ggml.tensor_ne(result_tensor, 0)
        result_ne1 = ggml.tensor_ne(result_tensor, 1)
        out = s.get(graph, result_tensor)

        print(f"mul mat ({result_ne0} x {result_ne1}) (transposed result):")
        print(out.reshape(result_ne1, result_ne0))


if __name__ == "__main__":
    main()

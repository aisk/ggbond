import getopt
import sys

import numpy as np

from ggbond import ggml
from ggbond.backend import Backend
from ggbond.context import Context
from ggbond.graph import GraphRunner


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

    with Backend(backend_type) as backend:
        with Context(n_tensors=2) as ctx_model:
            tensor_a = ctx_model.new_tensor(ggml.Type.F32, cols_a, rows_a)
            tensor_b = ctx_model.new_tensor(ggml.Type.F32, cols_b, rows_b)
            buffer = backend.alloc_ctx(ctx_model)

            backend.tensor_set(tensor_a, matrix_a)
            backend.tensor_set(tensor_b, matrix_b)

            with Context.for_graph() as ctx_graph:
                result = ggml.mul_mat(ctx_graph.raw, tensor_a, tensor_b)
                graph = ctx_graph.new_graph()
                ggml.build_forward_expand(graph, result)

                with GraphRunner(backend) as runner:
                    mem_size = runner.reserve(graph)
                    print(f"compute buffer size: {mem_size / 1024.0:.4f} KB")

                    runner.compute(graph)

                    result_tensor = ggml.graph_node(graph, -1)
                    result_ne0 = ggml.tensor_ne(result_tensor, 0)
                    result_ne1 = ggml.tensor_ne(result_tensor, 1)
                    out = runner.get(graph, result_tensor)

                print(f"mul mat ({result_ne0} x {result_ne1}) (transposed result):")
                print(out.reshape(result_ne1, result_ne0))

            ggml.backend_buffer_free(buffer)


if __name__ == "__main__":
    main()

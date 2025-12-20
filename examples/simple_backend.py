from ggbond import ggml
import numpy as np
import ctypes

def main():
    rows_a, cols_a = 4, 2
    a = np.array([
        2, 8,
        5, 1,
        4, 2,
        8, 6
    ], dtype=np.float32)

    rows_b, cols_b = 3, 2
    b = np.array([
        10, 5,
        9, 9,
        5, 4
    ], dtype=np.float32)

    print("Matrix A ({}x{}):".format(rows_a, cols_a))
    print(a.reshape(rows_a, cols_a))
    print("Matrix B ({}x{}):".format(rows_b, cols_b))
    print(b.reshape(rows_b, cols_b))

    expected = np.dot(a.reshape(rows_a, cols_a), b.reshape(rows_b, cols_b).T)
    print("Expected result A * B^T ({}x{}):".format(rows_a, rows_b))
    print(expected)

    num_tensors = 2
    model_ctx = ggml.context_init(mem_size=ggml.tensor_overhead() * num_tensors + 512, no_alloc=True)

    backend = ggml.backend_cpu_init()

    tensor_a = ggml.new_tensor_2d(model_ctx, ggml.Type.F32, cols_a, rows_a)
    tensor_b = ggml.new_tensor_2d(model_ctx, ggml.Type.F32, cols_b, rows_b)

    ggml.backend_alloc_ctx_tensors(model_ctx, backend)

    ggml.backend_tensor_set(tensor_a, a.astype(np.float32), 0, 0)
    ggml.backend_tensor_set(tensor_b, b.astype(np.float32), 0, 0)

    print("Successfully set tensor data using backend_tensor_set")

    graph_ctx_size = ggml.tensor_overhead() * 100 + ggml.graph_overhead()
    compute_ctx = ggml.context_init(mem_size=graph_ctx_size + 512, no_alloc=True)
    gf = ggml.new_graph(compute_ctx)

    result = ggml.mul_mat(compute_ctx, tensor_a, tensor_b)

    ggml.build_forward_expand(gf, result)

    ggml.backend_alloc_ctx_tensors(compute_ctx, backend)

    n_threads = 1
    status = ggml.graph_compute_with_ctx(compute_ctx, gf, n_threads)

    if status == ggml.Status.SUCCESS:
        ptr_result = ggml.get_data_f32(result)
        ptr_result_ctypes = ctypes.cast(ptr_result, ctypes.POINTER(ctypes.c_float))

        result_values = []
        for i in range(12):
            result_values.append(ptr_result_ctypes[i])

        result_matrix = np.zeros((4, 3), dtype=np.float32)
        result_matrix[:, 0] = result_values[0:4]
        result_matrix[:, 1] = result_values[4:8]
        result_matrix[:, 2] = result_values[8:12]

        print("Result matrix:")
        print(result_matrix)

        if np.allclose(result_matrix, expected):
            print("✅ Results match expected!")
        else:
            print("❌ Results differ from expected")

    else:
        print(f"❌ Computation failed with status: {status}")

    ggml.context_free(model_ctx)
    ggml.context_free(compute_ctx)


if __name__ == "__main__":
    main()
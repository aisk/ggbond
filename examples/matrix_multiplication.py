from ggbond import ggml
import numpy as np
import ctypes

def main():
    rows_A, cols_A = 4, 2
    matrix_A = np.array([
        2, 8,
        5, 1,
        4, 2,
        8, 6
    ], dtype=np.float32)

    rows_B, cols_B = 3, 2
    matrix_B = np.array([
        10, 5,
        9, 9,
        5, 4
    ], dtype=np.float32)

    print("Matrix A ({}x{}):".format(rows_A, cols_A))
    print(matrix_A.reshape(rows_A, cols_A))
    print("Matrix B ({}x{}):".format(rows_B, cols_B))
    print(matrix_B.reshape(rows_B, cols_B))

    # Calculate expected result for reference
    expected = np.dot(matrix_A.reshape(rows_A, cols_A), matrix_B.reshape(rows_B, cols_B).T)
    print("Expected result A * B^T ({}x{}):".format(rows_A, rows_B))
    print(expected)

    # Create GGML context with automatic memory allocation
    ctx_size = 100 * 1024  # 100KB should be sufficient
    params = ggml.InitParams(mem_size=ctx_size, no_alloc=False)
    ctx = ggml.init(params)

    # Create tensors
    # Note: new_tensor_2d takes (columns, rows)
    tensor_a = ggml.new_tensor_2d(ctx, ggml.Type.F32, cols_A, rows_A)
    tensor_b = ggml.new_tensor_2d(ctx, ggml.Type.F32, cols_B, rows_B)

    # Set tensor data
    ptr_a = ggml.get_data_f32(tensor_a)
    ptr_b = ggml.get_data_f32(tensor_b)
    ptr_a_ctypes = ctypes.cast(ptr_a, ctypes.POINTER(ctypes.c_float))
    ptr_b_ctypes = ctypes.cast(ptr_b, ctypes.POINTER(ctypes.c_float))

    # Copy data to tensors (equivalent to memcpy in C)
    for i in range(matrix_A.size):
        ptr_a_ctypes[i] = matrix_A[i]

    for i in range(matrix_B.size):
        ptr_b_ctypes[i] = matrix_B[i]

    # Create computation graph
    graph = ggml.new_graph(ctx)

    # Create matrix multiplication: result = a * b^T
    result = ggml.mul_mat(ctx, tensor_a, tensor_b)

    # Build forward computation graph
    ggml.build_forward_expand(graph, result)

    # Execute computation
    n_threads = 1
    status = ggml.graph_compute_with_ctx(ctx, graph, n_threads)

    if status == ggml.Status.SUCCESS:
        # Read result data
        ptr_result = ggml.get_data_f32(result)
        ptr_result_ctypes = ctypes.cast(ptr_result, ctypes.POINTER(ctypes.c_float))

        # Read the result values
        result_values = []
        for i in range(12):  # Expect 4x3 = 12 elements
            result_values.append(ptr_result_ctypes[i])

        # GGML stores results in column-major order
        # Reconstruct the 4x3 matrix
        result_matrix = np.zeros((4, 3), dtype=np.float32)
        result_matrix[:, 0] = result_values[0:4]   # First column
        result_matrix[:, 1] = result_values[4:8]   # Second column
        result_matrix[:, 2] = result_values[8:12]  # Third column

        print("Result matrix:")
        print(result_matrix)

        # Verify the result
        if np.allclose(result_matrix, expected):
            print("Results match expected!")
        else:
            print("️Results differ from expected")

    else:
        print(f"Computation failed with status: {status}")

    # Free memory
    ggml.free(ctx)


if __name__ == "__main__":
    main()
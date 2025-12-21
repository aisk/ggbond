from ggbond import ggml
import numpy as np


def main():
    ggml.time_init()

    # Initialize data for matrices to perform matrix multiplication
    # Matrix A: 4x2
    rows_a, cols_a = 4, 2
    a = np.array([
        [2, 8],
        [5, 1],
        [4, 2],
        [8, 6]
    ], dtype=np.float32)

    # Matrix B: 3x2 (will be transposed to 2x3 during computation)
    rows_b, cols_b = 3, 2
    b = np.array([
        [10, 5],
        [9, 9],
        [5, 4]
    ], dtype=np.float32)

    print("Matrix A ({}x{}):".format(rows_a, cols_a))
    print(a)
    print("Matrix B ({}x{}):".format(rows_b, cols_b))
    print(b)

    # Expected result: A * B^T
    expected = a @ b.T
    print("Expected result A * B^T ({}x{}):".format(rows_a, rows_b))
    print(expected)

    # Initialize backend
    backend = ggml.backend_cpu_init()
    if backend is None:
        print("Failed to initialize CPU backend")
        return

    print(f"Backend initialized: {backend is not None}")
    print(f"Is CPU backend: {ggml.backend_is_cpu(backend)}")

    # Set number of threads for CPU backend
    n_threads = 4  # Can be adjusted based on your CPU cores
    ggml.backend_cpu_set_n_threads(backend, n_threads)
    print(f"Set CPU backend threads to: {n_threads}")

    # Number of tensors in our model
    num_tensors = 2

    # Create context with no_alloc=true for manual memory management
    ctx = ggml.context_init(ggml.tensor_overhead() * num_tensors, no_alloc=True)

    # Create tensors
    tensor_a = ggml.new_tensor_2d(ctx, ggml.Type.F32, cols_a, rows_a)
    tensor_b = ggml.new_tensor_2d(ctx, ggml.Type.F32, cols_b, rows_b)

    # Allocate tensors to backend buffer
    buffer = ggml.backend_alloc_ctx_tensors(ctx, backend)

    # Set tensor data
    ggml.backend_tensor_set(tensor_a, a.flatten(), 0, ggml.nbytes(tensor_a))
    ggml.backend_tensor_set(tensor_b, b.flatten(), 0, ggml.nbytes(tensor_b))

    print(f"Tensor A dimensions: [{ggml.tensor_ne(tensor_a, 0)}, {ggml.tensor_ne(tensor_a, 1)}]")
    print(f"Tensor B dimensions: [{ggml.tensor_ne(tensor_b, 0)}, {ggml.tensor_ne(tensor_b, 1)}]")

    # Calculate temporary memory required for computation
    buf_size = ggml.tensor_overhead() * ggml.DEFAULT_GRAPH_SIZE() + ggml.graph_overhead()

    # Create a temporary context to build the graph
    ctx_graph = ggml.context_init(buf_size, no_alloc=True)
    graph = ggml.new_graph(ctx_graph)

    # result = a * b^T (matrix multiplication)
    result = ggml.mul_mat(ctx_graph, tensor_a, tensor_b)

    # Build operation nodes
    ggml.build_forward_expand(graph, result)

    # Create graph allocator
    buffer_type = ggml.backend_get_default_buffer_type(backend)
    allocr = ggml.gallocr_new(buffer_type)

    # Reserve memory for graph computation
    ggml.gallocr_reserve(allocr, graph)
    mem_size = ggml.gallocr_get_buffer_size(allocr, 0)
    print(f"Compute buffer size: {mem_size/1024.0:.4f} KB")

    # Allocate tensors for computation
    ggml.gallocr_alloc_graph(allocr, graph)

    # Compute using backend
    ggml.backend_graph_compute(backend, graph)

    # Get result tensor (last node in the graph)
    result_tensor = ggml.graph_node(graph, -1)
    print(f"Result tensor dimensions: [{ggml.tensor_ne(result_tensor, 0)}, {ggml.tensor_ne(result_tensor, 1)}]")

    # Create buffer to store result
    out_data = np.empty(ggml.nelements(result_tensor), dtype=np.float32)

    # Get data from backend memory
    ggml.backend_tensor_get(result_tensor, out_data, 0, ggml.nbytes(result_tensor))

    # Reshape result for display (GGML uses column-major order)
    # Result dimensions are [4, 3], data is stored column-major
    result_matrix = out_data.reshape(ggml.tensor_ne(result_tensor, 1), ggml.tensor_ne(result_tensor, 0)).T

    print("Result matrix:")
    print(result_matrix)

    if np.allclose(result_matrix, expected):
        print("✅ Results match expected!")
    else:
        print("❌ Results differ from expected")
        print(f"Max difference: {np.max(np.abs(result_matrix - expected))}")

    # Clean up
    ggml.gallocr_free(allocr)
    ggml.context_free(ctx)
    ggml.backend_buffer_free(buffer)
    ggml.backend_free(backend)

    print("Memory cleanup completed.")


if __name__ == "__main__":
    main()
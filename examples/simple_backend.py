import numpy as np

from ggbond import ggml


def main():
    ggml.time_init()
    ggml.log_set_default()

    # =========================================================================
    # Phase 1: Preparation - Define metadata and build graph
    # =========================================================================

    # 1.1 Initialize context for tensor metadata (no actual memory allocation)
    num_tensors = 2
    ctx_model = ggml.context_init(ggml.tensor_overhead() * num_tensors, no_alloc=True)

    # 1.2 Initialize context for graph building
    BUF_SIZE = ggml.tensor_overhead() * ggml.DEFAULT_GRAPH_SIZE() + ggml.graph_overhead()
    ctx_graph = ggml.context_init(BUF_SIZE, no_alloc=True)

    # 1.3 Define tensor metadata in ctx (shape, type, etc.)
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

    tensor_a = ggml.new_tensor_2d(ctx_model, ggml.Type.F32, cols_a, rows_a)
    tensor_b = ggml.new_tensor_2d(ctx_model, ggml.Type.F32, cols_b, rows_b)

    # 1.4 Build computation graph
    graph = ggml.new_graph(ctx_graph)
    result = ggml.mul_mat(ctx_graph, tensor_a, tensor_b)
    ggml.build_forward_expand(graph, result)

    # =========================================================================
    # Phase 2: Memory Allocation - Allocate actual memory on backend
    # =========================================================================

    # 2.1 Initialize backend (CPU in this case)
    backend = ggml.backend_cpu_init()

    # 2.2 Allocate actual memory for tensors in ctx_model on the backend
    buffer = ggml.backend_alloc_ctx_tensors(ctx_model, backend)

    # 2.3 Create allocator for computation graph and reserve memory
    buffer_type = ggml.backend_get_default_buffer_type(backend)
    allocr = ggml.gallocr_new(buffer_type)
    ggml.gallocr_reserve(allocr, graph)

    mem_size = ggml.gallocr_get_buffer_size(allocr, 0)
    print(f"compute buffer size: {mem_size/1024.0:.4f} KB")

    # =========================================================================
    # Phase 3: Data Transfer and Execution
    # =========================================================================

    # 3.1 Transfer data from CPU memory to backend buffer
    ggml.backend_tensor_set(tensor_a, matrix_a.flatten(), 0, ggml.nbytes(tensor_a))
    ggml.backend_tensor_set(tensor_b, matrix_b.flatten(), 0, ggml.nbytes(tensor_b))

    # 3.2 Allocate graph computation memory
    ggml.gallocr_alloc_graph(allocr, graph)

    # 3.3 Configure and execute computation
    ggml.backend_cpu_set_n_threads(backend, 4)

    ggml.backend_graph_compute(backend, graph)

    # 3.4 Get result from backend memory
    result_tensor = ggml.graph_node(graph, -1)
    out_data = np.empty(ggml.nelements(result_tensor), dtype=np.float32)
    ggml.backend_tensor_get(result_tensor, out_data, 0, ggml.nbytes(result_tensor))

    # =========================================================================
    # Display Result
    # =========================================================================

    result_ne0 = ggml.tensor_ne(result_tensor, 0)
    result_ne1 = ggml.tensor_ne(result_tensor, 1)
    result_matrix = out_data.reshape(result_ne1, result_ne0)

    print(f"mul mat ({result_ne0} x {result_ne1}) (transposed result):")
    print(result_matrix)

    # =========================================================================
    # Cleanup - Release all resources
    # =========================================================================

    ggml.gallocr_free(allocr)
    ggml.context_free(ctx_model)
    ggml.backend_buffer_free(buffer)
    ggml.backend_free(backend)
    ggml.context_free(ctx_graph)


if __name__ == "__main__":
    main()

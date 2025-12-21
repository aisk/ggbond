from ggbond import ggml
import numpy as np

# create a static buffer for graph building (similar to C++ version)
BUF_SIZE = ggml.tensor_overhead() * ggml.DEFAULT_GRAPH_SIZE() + ggml.graph_overhead()
CTX_GRAPH_BUFFER = ggml.context_init(BUF_SIZE, no_alloc=True)


def load_model(a, b, rows_A, cols_A, rows_B, cols_B):
    """
    initialize the tensors of the model in this case two matrices 2x2
    Returns: (backend, buffer, ctx, tensor_a, tensor_b)
    """
    # initialize the backend
    # if there aren't GPU Backends fallback to CPU backend
    backend = ggml.backend_cpu_init()
    if not backend:
        raise RuntimeError("Failed to initialize CPU backend")

    num_tensors = 2

    # create context
    ctx = ggml.context_init(ggml.tensor_overhead() * num_tensors, no_alloc=True)

    # create tensors
    tensor_a = ggml.new_tensor_2d(ctx, ggml.Type.F32, cols_A, rows_A)
    tensor_b = ggml.new_tensor_2d(ctx, ggml.Type.F32, cols_B, rows_B)

    # create a backend buffer (backend memory) and alloc the tensors from the context
    buffer = ggml.backend_alloc_ctx_tensors(ctx, backend)

    # load data from cpu memory to backend buffer
    ggml.backend_tensor_set(tensor_a, a.flatten(), 0, ggml.nbytes(tensor_a))
    ggml.backend_tensor_set(tensor_b, b.flatten(), 0, ggml.nbytes(tensor_b))

    return backend, buffer, ctx, tensor_a, tensor_b


def build_graph(tensor_a, tensor_b):
    """
    build the compute graph to perform a matrix multiplication
    Returns: graph
    """
    # use the static context buffer (similar to C++ version)
    graph = ggml.new_graph(CTX_GRAPH_BUFFER)

    # result = a*b^T
    result = ggml.mul_mat(CTX_GRAPH_BUFFER, tensor_a, tensor_b)

    # build operations nodes
    ggml.build_forward_expand(graph, result)

    return graph


def compute(backend, allocr, tensor_a, tensor_b):
    """
    compute with backend
    Returns: result tensor
    """
    # reset the allocator to free all the memory allocated during the previous inference
    graph = build_graph(tensor_a, tensor_b)

    # allocate tensors
    ggml.gallocr_alloc_graph(allocr, graph)

    n_threads = 1  # number of threads to perform some operations with multi-threading

    if ggml.backend_is_cpu(backend):
        ggml.backend_cpu_set_n_threads(backend, n_threads)

    ggml.backend_graph_compute(backend, graph)

    # in this case, the output tensor is the last one in the graph
    result = ggml.graph_node(graph, -1)
    return result


def main():
    ggml.time_init()

    # initialize data of matrices to perform matrix multiplication
    rows_A, cols_A = 4, 2
    matrix_A = np.array([
        [2, 8],
        [5, 1],
        [4, 2],
        [8, 6]
    ], dtype=np.float32)

    rows_B, cols_B = 3, 2
    # Transpose([
    #    10, 9, 5,
    #    5, 9, 4
    # ]) 2 rows, 3 cols
    matrix_B = np.array([
        [10, 5],
        [9, 9],
        [5, 4]
    ], dtype=np.float32)

    backend, buffer, ctx, tensor_a, tensor_b = load_model(
        matrix_A, matrix_B, rows_A, cols_A, rows_B, cols_B
    )

    # calculate the temporaly memory required to compute
    buffer_type = ggml.backend_get_default_buffer_type(backend)
    allocr = ggml.gallocr_new(buffer_type)

    # create the worst case graph for memory usage estimation
    graph = build_graph(tensor_a, tensor_b)
    ggml.gallocr_reserve(allocr, graph)
    mem_size = ggml.gallocr_get_buffer_size(allocr, 0)

    print(f"{__name__}: compute buffer size: {mem_size/1024.0:.4f} KB")

    # perform computation
    result = compute(backend, allocr, tensor_a, tensor_b)

    # create a array to print result
    out_data = np.empty(ggml.nelements(result), dtype=np.float32)

    # bring the data from the backend memory
    ggml.backend_tensor_get(result, out_data, 0, ggml.nbytes(result))

    # expected result:
    # [ 60.00 55.00 50.00 110.00
    #  90.00 54.00 54.00 126.00
    #  42.00 29.00 28.00 64.00 ]

    result_ne0 = ggml.tensor_ne(result, 0)
    result_ne1 = ggml.tensor_ne(result, 1)

    print(f"mul mat ({result_ne0} x {result_ne1}) (transposed result):")
    print("[", end="")
    for j in range(result_ne1):  # rows
        if j > 0:
            print("\n", end="")

        for i in range(result_ne0):  # cols
            print(f" {out_data[j * result_ne0 + i]:.2f}", end="")
    print(" ]")

    # release backend memory used for computation
    ggml.gallocr_free(allocr)

    # free memory
    ggml.context_free(ctx)

    # release backend memory and free backend
    ggml.backend_buffer_free(buffer)
    ggml.backend_free(backend)

    # free the static graph buffer
    ggml.context_free(CTX_GRAPH_BUFFER)


if __name__ == "__main__":
    main()
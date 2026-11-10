import gc
import unittest

import numpy as np

from ggbond.ggml import Backend, Context, F32, GAllocr, GGMLError, GGUF, Scheduler


class ClosedContextTests(unittest.TestCase):
    def test_tensor_raises_after_its_context_closes(self):
        ctx = Context(1 << 20, no_alloc=True)
        tensor = ctx.new_tensor_1d(F32, 3)
        ctx.close()

        with self.assertRaisesRegex(GGMLError, "context was closed"):
            tensor.ne
        with self.assertRaisesRegex(GGMLError, "closed"):
            tensor.relu()
        self.assertEqual(repr(tensor), "<ggml.Tensor closed>")

    def test_op_raises_when_target_context_is_closed(self):
        with Context(1 << 20, no_alloc=True) as weights:
            weight = weights.new_tensor_1d(F32, 3)
            graph_ctx = Context(1 << 20, no_alloc=True)
            graph_ctx.close()

            with self.assertRaisesRegex(GGMLError, "target context is closed"):
                graph_ctx.relu(weight)

    def test_op_raises_when_an_operand_context_is_closed(self):
        weights = Context(1 << 20, no_alloc=True)
        weight = weights.new_tensor_1d(F32, 3)
        with Context(1 << 20, no_alloc=True) as graph_ctx:
            weights.close()

            with self.assertRaisesRegex(GGMLError, "context was closed"):
                graph_ctx.relu(weight)

    def test_graph_raises_after_its_context_closes(self):
        ctx = Context(1 << 20, no_alloc=True)
        graph = ctx.new_graph()
        ctx.close()

        with self.assertRaisesRegex(GGMLError, "context was closed"):
            len(graph)
        self.assertEqual(repr(graph), "<ggml.Graph closed>")

    def test_context_raises_after_close(self):
        ctx = Context(1 << 20, no_alloc=True)
        ctx.close()
        ctx.close()

        self.assertTrue(ctx.closed)
        with self.assertRaisesRegex(GGMLError, "Context is closed"):
            ctx.new_tensor_1d(F32, 3)
        with self.assertRaisesRegex(GGMLError, "Context is closed"):
            ctx.used_mem

    def test_a_context_keeps_the_contexts_it_reads_from_alive(self):
        def build():
            weights = Context(1 << 20, no_alloc=True)
            graph_ctx = Context(1 << 20, no_alloc=True)
            return graph_ctx, graph_ctx.relu(weights.new_tensor_1d(F32, 3, name="w"))

        graph_ctx, result = build()
        gc.collect()

        self.assertEqual(result.src[0].name, "w")
        graph_ctx.close()


class ClosedHandleTests(unittest.TestCase):
    """Every owned handle raises once closed instead of passing NULL to ggml."""

    def test_backend(self):
        with Context(1 << 20, no_alloc=True) as ctx:
            graph = ctx.new_graph().build_forward_expand(ctx.new_tensor_1d(F32, 3).relu())
            backend = Backend.cpu_init()
            backend.close()

            with self.assertRaisesRegex(GGMLError, "Backend is closed"):
                backend.compute(graph)
            with self.assertRaisesRegex(GGMLError, "Backend is closed"):
                backend.alloc_ctx_tensors(ctx)
            with self.assertRaisesRegex(GGMLError, "Backend is closed"):
                Scheduler(backend)
            self.assertEqual(repr(backend), "<ggml.Backend closed>")

    def test_gallocr(self):
        with Backend.cpu_init() as backend, Context(1 << 20, no_alloc=True) as ctx:
            graph = ctx.new_graph().build_forward_expand(ctx.new_tensor_1d(F32, 3).relu())
            allocator = GAllocr.from_backend(backend)
            allocator.close()

            with self.assertRaisesRegex(GGMLError, "GAllocr is closed"):
                allocator.alloc_graph(graph)

    def test_scheduler(self):
        with Backend.cpu_init() as backend, Context(1 << 20, no_alloc=True) as ctx:
            graph = ctx.new_graph().build_forward_expand(ctx.new_tensor_1d(F32, 3).relu())
            scheduler = Scheduler(backend)
            scheduler.close()

            with self.assertRaisesRegex(GGMLError, "Scheduler is closed"):
                scheduler.alloc_graph(graph)

    def test_gguf(self):
        gguf = GGUF.init_empty()
        gguf.close()

        with self.assertRaisesRegex(GGMLError, "GGUF is closed"):
            gguf.n_kv
        self.assertEqual(repr(gguf), "<ggml.GGUF closed>")

    def test_closing_a_backend_closes_what_was_built_on_it(self):
        backend = Backend.cpu_init()
        allocator = GAllocr.from_backend(backend)
        scheduler = Scheduler(backend)

        backend.close()

        self.assertTrue(allocator.closed)
        self.assertTrue(scheduler.closed)


class UnallocatedTensorTests(unittest.TestCase):
    def test_transfers_raise_before_allocation(self):
        with Context(1 << 20, no_alloc=True) as ctx:
            tensor = ctx.new_tensor_1d(F32, 3, name="x")

            with self.assertRaisesRegex(GGMLError, "'x' has no backend buffer"):
                tensor.write([1, 2, 3])
            with self.assertRaisesRegex(GGMLError, "no backend buffer"):
                tensor.write_raw(b"\0" * 12)
            with self.assertRaisesRegex(GGMLError, "no backend buffer"):
                tensor.to_numpy()


class BufferRetentionTests(unittest.TestCase):
    def test_context_keeps_its_backend_buffer_alive(self):
        n = 1 << 16

        def load():
            backend = Backend.cpu_init()
            ctx = Context(1 << 20, no_alloc=True)
            tensor = ctx.new_tensor_1d(F32, n)
            backend.alloc_ctx_tensors(ctx)
            tensor.write(np.arange(n, dtype=np.float32))
            return ctx, tensor

        ctx, tensor = load()
        gc.collect()
        # Churn the allocator so freed memory would be reused.
        junk = [np.full(n, 7.0, dtype=np.float32) for _ in range(8)]

        np.testing.assert_array_equal(tensor.to_numpy(), np.arange(n, dtype=np.float32))
        del junk
        ctx.close()


if __name__ == "__main__":
    unittest.main()

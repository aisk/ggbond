import gc
import unittest

import numpy as np

from ggbond.ggml import Backend, Context, F32, GGMLError


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
            bound = graph_ctx.ref(weight)
            graph_ctx.close()

            with self.assertRaisesRegex(GGMLError, "target context is closed"):
                bound.relu()

    def test_ref_tracks_the_owning_context(self):
        weights = Context(1 << 20, no_alloc=True)
        weight = weights.new_tensor_1d(F32, 3)
        with Context(1 << 20, no_alloc=True) as graph_ctx:
            bound = graph_ctx.ref(weight)
            weights.close()

            with self.assertRaisesRegex(GGMLError, "context was closed"):
                bound.relu()

    def test_graph_raises_after_its_context_closes(self):
        ctx = Context(1 << 20, no_alloc=True)
        graph = ctx.new_graph()
        ctx.close()

        with self.assertRaisesRegex(GGMLError, "context was closed"):
            len(graph)


class UnallocatedTensorTests(unittest.TestCase):
    def test_transfers_raise_before_allocation(self):
        with Context(1 << 20, no_alloc=True) as ctx:
            tensor = ctx.new_tensor_1d(F32, 3, name="x")

            with self.assertRaisesRegex(GGMLError, "'x' has no backend buffer"):
                tensor.set([1, 2, 3])
            with self.assertRaisesRegex(GGMLError, "no backend buffer"):
                tensor.set_raw(b"\0" * 12)
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
            tensor.set(np.arange(n, dtype=np.float32))
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

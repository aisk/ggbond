import gc
import unittest

import numpy as np

from ggbond.ggml import Backend, Context, DeviceType, F32, GAllocr, GGMLError, Scheduler


def relu_graph(ctx, n=4, name="x"):
    x = ctx.new_tensor_1d(F32, n, name=name).set_input()
    out = x.relu().set_output()
    return ctx.new_graph().build_forward_expand(out), x, out


class ContextResetTests(unittest.TestCase):
    def test_reset_empties_the_arena_and_kills_old_tensors_and_graphs(self):
        with Context(1 << 20, no_alloc=True) as ctx:
            graph, x, out = relu_graph(ctx)
            used = ctx.used_mem

            ctx.reset()

            self.assertEqual(ctx.used_mem, 0)
            with self.assertRaisesRegex(GGMLError, "tensor used after its context was reset"):
                x.ne
            with self.assertRaisesRegex(GGMLError, "context was reset"):
                out.relu()
            with self.assertRaisesRegex(GGMLError, "graph used after its context was reset"):
                len(graph)
            self.assertEqual(repr(x), "<ggml.Tensor closed>")

            # The context is fully usable again.
            graph, x, out = relu_graph(ctx)
            self.assertEqual(ctx.used_mem, used)
            self.assertEqual(len(graph), 1)

    def test_a_reset_context_can_be_recomputed_every_step(self):
        with Backend.cpu_init() as backend, Context(1 << 20, no_alloc=True) as ctx, \
                GAllocr.from_backend(backend) as allocator:
            for step in range(5):
                ctx.reset()
                graph, x, out = relu_graph(ctx)
                allocator.alloc_graph(graph)
                x.write([step, -step, 2 * step, -1])
                backend.compute(graph)
                np.testing.assert_array_equal(out.to_numpy(), [step, 0, 2 * step, 0])
            # Graphs of earlier steps are forgotten, not accumulated.
            self.assertLessEqual(len(allocator._graphs), 5)

    def test_reset_releases_the_buffers_of_its_tensors(self):
        with Backend.cpu_init() as backend, Context(1 << 20, no_alloc=True) as ctx:
            ctx.new_tensor_1d(F32, 64)
            backend.alloc_ctx_tensors(ctx)
            self.assertEqual(len(backend._dependents), 1)

            ctx.reset()
            gc.collect()
            self.assertEqual(len(backend._dependents), 0)

    def test_a_tensor_of_a_reset_operand_context_is_rejected(self):
        with Context(1 << 20, no_alloc=True) as weights, \
                Context(1 << 20, no_alloc=True) as ctx:
            weight = weights.new_tensor_1d(F32, 4)
            weights.reset()

            with self.assertRaisesRegex(GGMLError, "context was reset"):
                ctx.relu(weight)


class AllocatorInvalidationTests(unittest.TestCase):
    def test_reading_after_the_allocator_closes_raises(self):
        with Backend.cpu_init() as backend, Context(1 << 20, no_alloc=True) as ctx:
            graph, x, out = relu_graph(ctx)
            with GAllocr.from_backend(backend) as allocator:
                allocator.alloc_graph(graph)
                x.write([1, -2, 3, -4])
                backend.compute(graph)
                np.testing.assert_array_equal(out.to_numpy(), [1, 0, 3, 0])

            for tensor in (x, out):
                with self.assertRaisesRegex(GGMLError, "has no backend buffer"):
                    tensor.to_numpy()
            with self.assertRaisesRegex(GGMLError, "has no backend buffer"):
                x.write([1, 2, 3, 4])

    def test_the_graph_can_be_allocated_again_by_another_allocator(self):
        with Backend.cpu_init() as backend, Context(1 << 20, no_alloc=True) as ctx:
            graph, x, out = relu_graph(ctx)
            with GAllocr.from_backend(backend) as allocator:
                allocator.alloc_graph(graph)
            with GAllocr.from_backend(backend) as allocator:
                allocator.alloc_graph(graph)
                x.write([-1, 2, -3, 4])
                backend.compute(graph)
                np.testing.assert_array_equal(out.to_numpy(), [0, 2, 0, 4])

    def test_tensors_in_a_context_buffer_survive_the_allocator(self):
        with Backend.cpu_init() as backend, Context(1 << 20, no_alloc=True) as ctx:
            graph, x, out = relu_graph(ctx)
            backend.alloc_ctx_tensors(ctx)
            with GAllocr.from_backend(backend) as allocator:
                allocator.alloc_graph(graph)
                x.write([1, -2, 3, -4])
                backend.compute(graph)

            np.testing.assert_array_equal(out.to_numpy(), [1, 0, 3, 0])

    def test_growing_the_buffer_invalidates_earlier_graphs(self):
        with Backend.cpu_init() as backend, Context(1 << 20, no_alloc=True) as ctx, \
                GAllocr.from_backend(backend) as allocator:
            small, x_small, out_small = relu_graph(ctx, 4)
            allocator.alloc_graph(small)
            x_small.write([1, 2, 3, 4])

            # Same size again: the buffer is reused and the tensor stays valid.
            same, x_same, _ = relu_graph(ctx, 4)
            allocator.alloc_graph(same)
            x_small.to_numpy()

            large, x_large, out_large = relu_graph(ctx, 1 << 16)
            allocator.alloc_graph(large)

            with self.assertRaisesRegex(GGMLError, "has no backend buffer"):
                x_small.to_numpy()
            with self.assertRaisesRegex(GGMLError, "has no backend buffer"):
                x_same.to_numpy()
            x_large.write(np.ones(1 << 16, dtype=np.float32))
            backend.compute(large)
            self.assertEqual(out_large.to_numpy().sum(), 1 << 16)

    def test_a_larger_reserve_invalidates_earlier_graphs(self):
        with Backend.cpu_init() as backend, Context(1 << 20, no_alloc=True) as ctx, \
                GAllocr.from_backend(backend) as allocator:
            small, x_small, _ = relu_graph(ctx, 4)
            allocator.alloc_graph(small)
            large, _, _ = relu_graph(ctx, 1 << 16)

            allocator.reserve(large)

            with self.assertRaisesRegex(GGMLError, "has no backend buffer"):
                x_small.to_numpy()

    def test_two_allocators_only_release_their_own_tensors(self):
        with Backend.cpu_init() as backend, Context(1 << 20, no_alloc=True) as ctx:
            first = GAllocr.from_backend(backend)
            second = GAllocr.from_backend(backend)
            graph_a, x_a, out_a = relu_graph(ctx, 4, "a")
            first.alloc_graph(graph_a)
            x_a.write([1, -1, 2, -2])
            backend.compute(graph_a)

            # The second graph reads a tensor the first allocator owns.
            doubled = out_a.scale(2.0).set_output()
            graph_b = ctx.new_graph().build_forward_expand(doubled)
            second.alloc_graph(graph_b)
            backend.compute(graph_b)

            second.close()
            np.testing.assert_array_equal(out_a.to_numpy(), [1, 0, 2, 0])
            with self.assertRaisesRegex(GGMLError, "has no backend buffer"):
                doubled.to_numpy()

            first.close()
            with self.assertRaisesRegex(GGMLError, "has no backend buffer"):
                out_a.to_numpy()

    def test_closing_the_scheduler_invalidates_what_it_allocated(self):
        Backend.load_all()
        with Backend.init_by_type(DeviceType.CPU) as backend, \
                Context(1 << 20, no_alloc=True) as ctx:
            graph, x, out = relu_graph(ctx)
            with Scheduler(backend) as scheduler:
                scheduler.alloc_graph(graph)
                x.write([1, -2, 3, -4])
                scheduler.compute(graph)
                np.testing.assert_array_equal(out.to_numpy(), [1, 0, 3, 0])

            with self.assertRaisesRegex(GGMLError, "has no backend buffer"):
                out.to_numpy()

    def test_closing_the_backend_invalidates_through_its_allocator(self):
        with Context(1 << 20, no_alloc=True) as ctx:
            backend = Backend.cpu_init()
            graph, x, out = relu_graph(ctx)
            allocator = GAllocr.from_backend(backend)
            allocator.alloc_graph(graph)

            backend.close()

            with self.assertRaisesRegex(GGMLError, "has no backend buffer"):
                out.to_numpy()

    def test_closing_an_allocator_after_its_graph_context_is_harmless(self):
        with Backend.cpu_init() as backend:
            allocator = GAllocr.from_backend(backend)
            with Context(1 << 20, no_alloc=True) as ctx:
                graph, _, _ = relu_graph(ctx)
                allocator.alloc_graph(graph)
            allocator.close()


class BufferInvalidationTests(unittest.TestCase):
    def test_closing_a_buffer_unallocates_its_tensors(self):
        with Backend.cpu_init() as backend, Context(1 << 20, no_alloc=True) as ctx:
            tensor = ctx.new_tensor_1d(F32, 4, name="w")
            buffer = backend.alloc_ctx_tensors(ctx)
            tensor.write([1, 2, 3, 4])

            buffer.close()

            with self.assertRaisesRegex(GGMLError, "'w' has no backend buffer"):
                tensor.to_numpy()

            # ggml sees it as unallocated too, so it can be allocated afresh.
            backend.alloc_ctx_tensors(ctx)
            tensor.write([5, 6, 7, 8])
            np.testing.assert_array_equal(tensor.to_numpy(), [5, 6, 7, 8])

    def test_closing_the_backend_unallocates_context_tensors(self):
        with Context(1 << 20, no_alloc=True) as ctx:
            backend = Backend.cpu_init()
            tensor = ctx.new_tensor_1d(F32, 4)
            backend.alloc_ctx_tensors(ctx)

            backend.close()

            with self.assertRaisesRegex(GGMLError, "has no backend buffer"):
                tensor.to_numpy()

    def test_a_view_in_another_context_notices_the_freed_buffer(self):
        with Backend.cpu_init() as backend, \
                Context(1 << 20, no_alloc=True) as weights, \
                Context(1 << 20, no_alloc=True) as ctx:
            weight = weights.new_tensor_1d(F32, 4, name="w")
            buffer = backend.alloc_ctx_tensors(weights)
            weight.write([1, 2, 3, 4])

            view = ctx.view_1d(weight, 2, 0).set_output()
            graph = ctx.new_graph().build_forward_expand(view)
            with GAllocr.from_backend(backend) as allocator:
                allocator.alloc_graph(graph)
                np.testing.assert_array_equal(view.to_numpy(), [1, 2])

                buffer.close()
                with self.assertRaisesRegex(GGMLError, "was freed"):
                    view.to_numpy()


if __name__ == "__main__":
    unittest.main()

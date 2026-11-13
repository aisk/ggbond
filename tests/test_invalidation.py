import functools
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
            # Only the current graph is remembered, not every earlier step.
            self.assertEqual(len(allocator._held), 1)

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

            # Same size again: the buffer is reused for the new graph.
            same, x_same, _ = relu_graph(ctx, 4)
            allocator.alloc_graph(same)
            with self.assertRaisesRegex(GGMLError, "has no backend buffer"):
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


class Runner:
    """One graph allocator under test, driven the way its users drive it."""

    def __init__(self, kind):
        self.kind = kind
        if kind == "scheduler":
            Backend.load_all()
            self.backend = Backend.init_by_type(DeviceType.CPU)
            self.allocator = Scheduler(self.backend)
        else:
            self.backend = Backend.cpu_init()
            self.allocator = GAllocr.from_backend(self.backend)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.backend.close()

    def release(self):
        """What a user does between two evaluations."""
        if self.kind == "scheduler":
            self.allocator.reset()

    def alloc(self, graph):
        self.release()
        self.allocator.alloc_graph(graph)

    def compute(self, graph):
        if self.kind == "scheduler":
            self.allocator.compute(graph)
        else:
            self.backend.compute(graph)


def each_allocator(test):
    """Run ``test(self, runner, ctx)`` with a ``GAllocr`` and with a ``Scheduler``."""

    @functools.wraps(test)
    def run(self):
        for kind in ("gallocr", "scheduler"):
            with self.subTest(kind), Runner(kind) as runner, \
                    Context(1 << 20, no_alloc=True) as ctx:
                test(self, runner, ctx)

    return run


class SuccessiveGraphTests(unittest.TestCase):
    """An allocator holds one graph at a time: the next one takes its memory."""

    def assert_unallocated(self, *tensors):
        for tensor in tensors:
            with self.assertRaisesRegex(GGMLError, "has no backend buffer"):
                tensor.to_numpy()

    @each_allocator
    def test_the_next_graph_unallocates_the_previous_one(self, runner, ctx):
        # Same buffer size: the old output used to read back the new data.
        first, x1, out1 = relu_graph(ctx)
        runner.alloc(first)
        x1.write([1, -2, 3, -4])
        runner.compute(first)
        np.testing.assert_array_equal(out1.to_numpy(), [1, 0, 3, 0])

        # Releasing the graph alone keeps its results readable.
        runner.release()
        np.testing.assert_array_equal(out1.to_numpy(), [1, 0, 3, 0])

        second, x2, out2 = relu_graph(ctx)
        runner.alloc(second)
        self.assert_unallocated(x1, out1)
        x2.write([-5, 6, -7, 8])
        runner.compute(second)
        np.testing.assert_array_equal(out2.to_numpy(), [0, 6, 0, 8])
        self.assert_unallocated(x1, out1)

    def test_computing_without_alloc_graph_unallocates_the_previous_one(self):
        Backend.load_all()
        with Backend.init_by_type(DeviceType.CPU) as backend, Scheduler(backend) as scheduler, \
                Context(1 << 20, no_alloc=True) as ctx:
            first, x1, out1 = relu_graph(ctx)
            scheduler.alloc_graph(first)
            x1.write([1, -2, 3, -4])
            scheduler.compute(first)
            scheduler.reset()

            x2 = ctx.arange(0.0, 4.0, 1.0)
            out2 = x2.neg().relu().set_output()
            second = ctx.new_graph().build_forward_expand(out2)
            scheduler.compute(second)

            self.assert_unallocated(x1, out1)
            np.testing.assert_array_equal(out2.to_numpy(), [0, 0, 0, 0])

    @each_allocator
    def test_a_tensor_shared_with_a_larger_graph_is_allocated_afresh(self, runner, ctx):
        # Used to segfault: the shared input stayed in the buffer that the
        # larger graph made the allocator free.
        x = ctx.new_tensor_1d(F32, 4, name="x").set_input()
        out1 = x.scale(2.0).set_output()
        first = ctx.new_graph().build_forward_expand(out1)
        runner.alloc(first)
        x.write([1, 1, 1, 1])
        runner.compute(first)
        np.testing.assert_array_equal(out1.to_numpy(), [2, 2, 2, 2])

        p = ctx.new_tensor_1d(F32, 4, name="p").set_input()
        big = ctx.new_tensor_1d(F32, 1 << 16, name="big").set_input()
        out2 = p.add(x).sqr().set_output()
        total = big.sqr().sum().set_output()
        second = ctx.new_graph().build_forward_expand(out2).build_forward_expand(total)
        runner.alloc(second)

        self.assert_unallocated(out1)
        x.write([1, 2, 3, 4])
        p.write([3, 3, 3, 3])
        big.write(np.ones(1 << 16, dtype=np.float32))
        runner.compute(second)
        np.testing.assert_array_equal(out2.to_numpy(), [16, 25, 36, 49])
        self.assertEqual(total.to_numpy().item(), 1 << 16)

    @each_allocator
    def test_a_tensor_shared_with_a_same_sized_graph_is_allocated_afresh(self, runner, ctx):
        x = ctx.new_tensor_1d(F32, 4, name="x").set_input()
        out1 = x.scale(2.0).set_output()
        first = ctx.new_graph().build_forward_expand(out1)
        runner.alloc(first)
        x.write([1, 1, 1, 1])
        runner.compute(first)

        out2 = x.scale(3.0).set_output()
        second = ctx.new_graph().build_forward_expand(out2)
        runner.alloc(second)
        x.write([1, 2, 3, 4])
        runner.compute(second)
        np.testing.assert_array_equal(out2.to_numpy(), [3, 6, 9, 12])
        self.assert_unallocated(out1)

    @each_allocator
    def test_reserving_for_a_graph_that_shares_a_tensor(self, runner, ctx):
        x = ctx.new_tensor_1d(F32, 4, name="x").set_input()
        out1 = x.scale(2.0).set_output()
        first = ctx.new_graph().build_forward_expand(out1)
        runner.alloc(first)
        x.write([1, 1, 1, 1])
        runner.compute(first)

        big = ctx.new_tensor_1d(F32, 1 << 16, name="big").set_input()
        out2 = x.add(big.sqr().sum()).set_output()
        second = ctx.new_graph().build_forward_expand(out2)
        runner.allocator.reserve(second)
        self.assert_unallocated(x, out1)

        runner.alloc(second)
        x.write([1, 2, 3, 4])
        big.write(np.ones(1 << 16, dtype=np.float32))
        runner.compute(second)
        np.testing.assert_array_equal(out2.to_numpy(), np.array([1, 2, 3, 4]) + (1 << 16))

    @each_allocator
    def test_a_kept_output_of_a_dropped_graph_is_unallocated(self, runner, ctx):
        # Used to segfault: the tensors of a graph whose wrapper was collected
        # were never unallocated, so the output kept pointing at freed memory.
        def evaluate(runner, ctx):
            graph, x, out = relu_graph(ctx)
            runner.alloc(graph)
            x.write([1, -2, 3, -4])
            runner.compute(graph)
            return out

        out = evaluate(runner, ctx)
        gc.collect()
        np.testing.assert_array_equal(out.to_numpy(), [1, 0, 3, 0])

        large, x_large, out_large = relu_graph(ctx, 1 << 16)
        runner.alloc(large)
        self.assert_unallocated(out)
        x_large.write(np.ones(1 << 16, dtype=np.float32))
        runner.compute(large)
        self.assertEqual(out_large.to_numpy().sum(), 1 << 16)
        self.assert_unallocated(out)

    @each_allocator
    def test_a_kept_output_of_a_dropped_graph_is_unallocated_on_close(self, runner, ctx):
        out = relu_graph(ctx)[2]
        runner.alloc(ctx.new_graph().build_forward_expand(out))
        gc.collect()
        runner.allocator.close()
        self.assert_unallocated(out)

    @each_allocator
    def test_allocating_the_same_graph_again_keeps_its_tensors(self, runner, ctx):
        x = ctx.new_tensor_1d(F32, 4, name="x").set_input()
        bias = ctx.new_tensor_1d(F32, 4, name="bias").set_input()
        out = x.add(bias).set_output()
        graph = ctx.new_graph().build_forward_expand(out)
        runner.alloc(graph)
        bias.write([10, 20, 30, 40])
        address = bias.ptr.contents.data

        for step in range(3):
            runner.alloc(graph)
            self.assertEqual(bias.ptr.contents.data, address)
            x.write([step] * 4)
            runner.compute(graph)
            np.testing.assert_array_equal(out.to_numpy(), np.array([10, 20, 30, 40]) + step)
        np.testing.assert_array_equal(bias.to_numpy(), [10, 20, 30, 40])

    @each_allocator
    def test_a_graph_that_grew_is_allocated_afresh(self, runner, ctx):
        x = ctx.new_tensor_1d(F32, 4, name="x").set_input()
        out1 = x.scale(2.0).set_output()
        graph = ctx.new_graph().build_forward_expand(out1)
        runner.alloc(graph)
        x.write([1, 1, 1, 1])
        runner.compute(graph)

        big = ctx.new_tensor_1d(F32, 1 << 16, name="big").set_input()
        total = big.sqr().sum().set_output()
        graph.build_forward_expand(total)
        runner.alloc(graph)
        x.write([1, 2, 3, 4])
        big.write(np.ones(1 << 16, dtype=np.float32))
        runner.compute(graph)
        np.testing.assert_array_equal(out1.to_numpy(), [2, 4, 6, 8])
        self.assertEqual(total.to_numpy().item(), 1 << 16)

    @each_allocator
    def test_tensors_in_other_buffers_are_never_unallocated(self, runner, ctx):
        # Weights and a cache written by a cpy node, as in the GPT-2 example.
        with Context(1 << 16, no_alloc=True) as model:
            weight = model.new_tensor_1d(F32, 4, name="w")
            cache = model.new_tensor_1d(F32, 8, name="cache")
            runner.backend.alloc_ctx_tensors(model)
            weight.write([1, 2, 3, 4])
            cache.write(np.zeros(8, dtype=np.float32))

            for step in range(2):
                x = ctx.new_tensor_1d(F32, 4, name="x").set_input()
                out = x.add(weight)
                stored = out.cpy(ctx.view_1d(cache, 4, 4 * 4 * step)).set_output()
                graph = ctx.new_graph().build_forward_expand(stored)
                runner.alloc(graph)
                x.write([10 * (step + 1)] * 4)
                runner.compute(graph)

            np.testing.assert_array_equal(weight.to_numpy(), [1, 2, 3, 4])
            np.testing.assert_array_equal(
                cache.to_numpy(), [11, 12, 13, 14, 21, 22, 23, 24])
            runner.allocator.close()
            np.testing.assert_array_equal(weight.to_numpy(), [1, 2, 3, 4])
            np.testing.assert_array_equal(
                cache.to_numpy(), [11, 12, 13, 14, 21, 22, 23, 24])

    @each_allocator
    def test_a_tensor_of_another_context_outlives_the_reset_graph_context(self, runner, ctx):
        # The graph context is reset each step, as a session does, while the
        # input lives in a context of its own and is shared by every step.
        with Context(1 << 16, no_alloc=True) as inputs:
            x = inputs.new_tensor_1d(F32, 4, name="x").set_input()
            for step, n in enumerate((4, 1 << 16, 4)):
                runner.release()
                ctx.reset()
                big = ctx.new_tensor_1d(F32, n, name="big").set_input()
                out = ctx.add(ctx.scale(x, 2.0), ctx.sum(ctx.sqr(big))).set_output()
                graph = ctx.new_graph().build_forward_expand(out)
                runner.alloc(graph)
                x.write([step] * 4)
                big.write(np.ones(n, dtype=np.float32))
                runner.compute(graph)
                np.testing.assert_array_equal(out.to_numpy(), [2 * step + n] * 4)

            # Without the graph context nothing says where x was used, yet
            # the next graph still takes its memory.
            runner.release()
            ctx.reset()
            x.to_numpy()
            runner.alloc(relu_graph(ctx)[0])
            self.assert_unallocated(x)

    def test_two_allocators_sharing_a_graph_keep_their_own_tensors(self):
        with Backend.cpu_init() as backend, Context(1 << 20, no_alloc=True) as ctx:
            first = GAllocr.from_backend(backend)
            second = GAllocr.from_backend(backend)
            graph_a, x_a, out_a = relu_graph(ctx, 4, "a")
            first.alloc_graph(graph_a)
            x_a.write([1, -1, 2, -2])
            backend.compute(graph_a)

            for factor in (2.0, 3.0):
                scaled = out_a.scale(factor).set_output()
                graph_b = ctx.new_graph().build_forward_expand(scaled)
                second.alloc_graph(graph_b)
                backend.compute(graph_b)
                np.testing.assert_array_equal(scaled.to_numpy(), np.array([1, 0, 2, 0]) * factor)
                np.testing.assert_array_equal(out_a.to_numpy(), [1, 0, 2, 0])


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


class OutputFlagTests(unittest.TestCase):
    """Marking another tensor as output asks for a new layout of the same graph."""

    def test_a_tensor_marked_output_later_is_not_computed_over(self):
        with Backend.cpu_init() as backend, \
                Context(1 << 20, no_alloc=True) as ctx, \
                GAllocr.from_backend(backend) as allocator:
            x = ctx.new_tensor_1d(F32, 4).set_input()
            hidden = x.scale(2.0)
            out = hidden.scale_bias(1.0, 1.0).set_output()
            graph = ctx.new_graph().build_forward_expand(out)

            allocator.alloc_graph(graph)
            x.write(np.ones(4, np.float32))
            backend.compute(graph)
            np.testing.assert_allclose(out.to_numpy(), 3.0)

            # ggml keeps a tensor that still has data where it is, and there
            # ``out`` was computed in place over ``hidden``.
            hidden.set_output()
            allocator.reserve(graph)
            allocator.alloc_graph(graph)
            x.write(np.ones(4, np.float32))
            backend.compute(graph)
            np.testing.assert_allclose(hidden.to_numpy(), 2.0)
            np.testing.assert_allclose(out.to_numpy(), 3.0)

    def _rerun_with_another_output(self, allocator, compute, reset):
        with Context(1 << 20, no_alloc=True) as ctx:
            x = ctx.new_tensor_1d(F32, 4).set_input()
            hidden = x.scale(2.0)
            out = hidden.scale_bias(1.0, 1.0).set_output()
            graph = ctx.new_graph().build_forward_expand(out)
            for mark in (False, True):
                if mark:
                    hidden.set_output()
                reset()
                allocator.alloc_graph(graph)  # no reserve: the wrapper plans afresh
                x.write(np.ones(4, np.float32))
                compute(graph)
                np.testing.assert_allclose(out.to_numpy(), 3.0)
            np.testing.assert_allclose(hidden.to_numpy(), 2.0)

    def test_alloc_graph_alone_lays_the_changed_request_out_afresh(self):
        with Backend.cpu_init() as backend:
            with GAllocr.from_backend(backend) as allocator:
                self._rerun_with_another_output(allocator, backend.compute, lambda: None)
            with Scheduler(backend) as sched:
                self._rerun_with_another_output(sched, sched.compute, sched.reset)

    def test_a_same_shaped_graph_gets_its_own_layout(self):
        # ggml reuses its plan for any graph with the same node count and
        # sizes, even when other tensors are the outputs.
        with Backend.cpu_init() as backend, GAllocr.from_backend(backend) as allocator:
            for keep_hidden in (False, True):
                with Context(1 << 20, no_alloc=True) as ctx:
                    x = ctx.new_tensor_1d(F32, 4).set_input()
                    hidden = x.scale(2.0)
                    if keep_hidden:
                        hidden.set_output()
                    out = hidden.scale_bias(1.0, 1.0).set_output()
                    graph = ctx.new_graph().build_forward_expand(out)
                    allocator.alloc_graph(graph)
                    x.write(np.ones(4, np.float32))
                    backend.compute(graph)
                    np.testing.assert_allclose(out.to_numpy(), 3.0)
                    if keep_hidden:
                        np.testing.assert_allclose(hidden.to_numpy(), 2.0)


class UnallocatedGraphTests(unittest.TestCase):
    """Backend.compute on a graph without memory raises; ggml would crash."""

    def _graph(self, ctx, factor):
        x = ctx.new_tensor_1d(F32, 4).set_input()
        out = x.scale(factor).set_output()
        return ctx.new_graph().build_forward_expand(out), x, out

    def test_computing_a_graph_that_was_never_allocated_raises(self):
        with Backend.cpu_init() as backend, Context(1 << 20, no_alloc=True) as ctx:
            graph, _, _ = self._graph(ctx, 2.0)
            with self.assertRaisesRegex(GGMLError, "is not allocated"):
                backend.compute(graph)
            with self.assertRaisesRegex(GGMLError, "is not allocated"):
                backend.compute_async(graph)

    def test_computing_a_graph_its_allocator_replaced_raises(self):
        with Backend.cpu_init() as backend, \
                Context(1 << 20, no_alloc=True) as ctx_a, \
                Context(1 << 20, no_alloc=True) as ctx_b, \
                GAllocr.from_backend(backend) as allocator:
            first, x1, out1 = self._graph(ctx_a, 2.0)
            second, x2, out2 = self._graph(ctx_b, 3.0)
            allocator.alloc_graph(first)
            x1.write(np.ones(4, np.float32))
            backend.compute(first)
            np.testing.assert_allclose(out1.to_numpy(), 2.0)

            allocator.alloc_graph(second)
            x2.write(np.ones(4, np.float32))
            backend.compute(second)
            np.testing.assert_allclose(out2.to_numpy(), 3.0)

            with self.assertRaisesRegex(GGMLError, "is not allocated"):
                backend.compute(first)
            # Allocating it again makes it computable again.
            allocator.alloc_graph(first)
            x1.write(np.ones(4, np.float32))
            backend.compute(first)
            np.testing.assert_allclose(out1.to_numpy(), 2.0)

    def test_computing_after_the_allocator_closed_raises(self):
        with Backend.cpu_init() as backend, Context(1 << 20, no_alloc=True) as ctx:
            graph, x, _ = self._graph(ctx, 2.0)
            with GAllocr.from_backend(backend) as allocator:
                allocator.alloc_graph(graph)
                x.write(np.ones(4, np.float32))
                backend.compute(graph)
            with self.assertRaisesRegex(GGMLError, "is not allocated"):
                backend.compute(graph)


if __name__ == "__main__":
    unittest.main()

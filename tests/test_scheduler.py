import unittest

import numpy as np

from ggbond.ggml import Backend, Context, Device, DeviceType, F32, GGMLError, Scheduler


def _add_graph(ctx, value, n=4):
    """A graph in ``ctx`` adding ``value`` to an input; returns (graph, input, output)."""
    x = ctx.new_tensor_1d(F32, n).set_input()
    out = x.scale_bias(1.0, float(value)).set_output()
    return ctx.new_graph().build_forward_expand(out), x, out


class SchedulerValidationTests(unittest.TestCase):
    def test_constructs_with_latest_scheduler_signature(self):
        with Backend.cpu_init() as backend:
            with Scheduler(backend, op_offload=False) as scheduler:
                self.assertIsNotNone(scheduler.ptr)
                self.assertEqual(scheduler.backends, (backend,))

    def test_rejects_wrong_number_of_buffer_types(self):
        with Backend.cpu_init() as backend:
            with self.assertRaisesRegex(ValueError, "one buffer type per backend"):
                Scheduler(backend, bufts=[])

    def test_rejects_null_buffer_type(self):
        with Backend.cpu_init() as backend:
            with self.assertRaisesRegex(ValueError, "buffer types must be non-null"):
                Scheduler(backend, bufts=[None])

    def test_rejects_closed_backend(self):
        backend = Backend.cpu_init()
        backend.close()

        with self.assertRaisesRegex(GGMLError, "Backend is closed"):
            Scheduler(backend)

    def test_rejects_non_positive_graph_size(self):
        with Backend.cpu_init() as backend:
            for graph_size in (0, -1):
                with self.subTest(graph_size=graph_size):
                    with self.assertRaisesRegex(ValueError, "graph_size must be positive"):
                        Scheduler(backend, graph_size=graph_size)


class SchedulerBackendOrderTests(unittest.TestCase):
    def test_rejects_a_last_backend_that_is_not_cpu(self):
        Backend.load_all()
        device = next((d for d in Device.all() if d.type != DeviceType.CPU), None)
        if device is None:
            self.skipTest("no non-CPU device registered")
        with device.init_backend() as other, Backend.cpu_init() as cpu:
            with self.assertRaisesRegex(ValueError, "last backend .* must be a CPU"):
                Scheduler(cpu, other)
            with Scheduler(other, cpu) as scheduler:
                self.assertEqual(scheduler.backends, (other, cpu))


class SchedulerStateTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend.cpu_init()
        self.addCleanup(self.backend.close)
        self.ctx = Context(1 << 20, no_alloc=True)
        self.addCleanup(self.ctx.close)

    def test_a_second_alloc_graph_needs_a_reset(self):
        with Scheduler(self.backend) as scheduler:
            first, _, _ = _add_graph(self.ctx, 1)
            second, x, out = _add_graph(self.ctx, 2)
            scheduler.alloc_graph(first)
            with self.assertRaisesRegex(GGMLError, "already holds an allocated graph"):
                scheduler.alloc_graph(second)
            with self.assertRaisesRegex(GGMLError, "already holds an allocated graph"):
                scheduler.alloc_graph(first)

            scheduler.reset()
            scheduler.alloc_graph(second)
            x.write(np.zeros(4, np.float32))
            scheduler.compute(second)
            np.testing.assert_allclose(out.to_numpy(), 2.0)

    def test_computing_another_graph_than_the_allocated_one_raises(self):
        with Scheduler(self.backend) as scheduler:
            first, x1, out1 = _add_graph(self.ctx, 1)
            second, _, _ = _add_graph(self.ctx, 2)
            scheduler.alloc_graph(first)
            x1.write(np.zeros(4, np.float32))
            with self.assertRaisesRegex(GGMLError, "another allocated graph"):
                scheduler.compute(second)
            with self.assertRaisesRegex(GGMLError, "another allocated graph"):
                scheduler.compute_async(second)
            # The allocated graph still computes, repeatedly. (ggml may run
            # the op in place over its input, so upload it each time.)
            for _ in range(2):
                x1.write(np.zeros(4, np.float32))
                scheduler.compute(first)
                np.testing.assert_allclose(out1.to_numpy(), 1.0)

    def test_compute_allocates_and_then_holds_the_graph(self):
        with Scheduler(self.backend) as scheduler:
            first, _, _ = _add_graph(self.ctx, 1)
            second, _, _ = _add_graph(self.ctx, 2)
            scheduler.compute(first)
            with self.assertRaisesRegex(GGMLError, "another allocated graph"):
                scheduler.compute(second)
            scheduler.reset()
            scheduler.compute(second)

    def test_a_context_reset_without_a_scheduler_reset_raises(self):
        # ggml would recompute the allocated graph, whose tensors are gone.
        with Scheduler(self.backend) as scheduler:
            first, _, _ = _add_graph(self.ctx, 1)
            scheduler.compute(first)
            self.ctx.reset()
            second, x, out = _add_graph(self.ctx, 2)
            with self.assertRaisesRegex(GGMLError, "another allocated graph"):
                scheduler.compute(second)
            scheduler.reset()
            scheduler.alloc_graph(second)
            x.write(np.zeros(4, np.float32))
            scheduler.compute(second)
            np.testing.assert_allclose(out.to_numpy(), 2.0)

    def test_reserve_leaves_nothing_allocated(self):
        with Scheduler(self.backend) as scheduler:
            big, _, _ = _add_graph(self.ctx, 1, n=1024)
            small, x, out = _add_graph(self.ctx, 2)
            scheduler.reserve(big)
            size = scheduler.buffer_size(self.backend)
            self.assertGreater(size, 0)
            scheduler.alloc_graph(small)
            x.write(np.zeros(4, np.float32))
            scheduler.compute(small)
            np.testing.assert_allclose(out.to_numpy(), 2.0)
            self.assertEqual(scheduler.buffer_size(self.backend), size)

    def test_a_graph_larger_than_the_scheduler_raises(self):
        with Scheduler(self.backend, graph_size=4) as scheduler:
            x = self.ctx.new_tensor_1d(F32, 4).set_input()
            out = x
            for _ in range(8):
                out = out.scale(2.0)
            graph = self.ctx.new_graph().build_forward_expand(out.set_output())
            for call in (scheduler.reserve, scheduler.alloc_graph, scheduler.compute):
                with self.subTest(call=call.__name__):
                    with self.assertRaisesRegex(GGMLError, "larger graph_size"):
                        call(graph)

    def test_buffer_size_rejects_a_foreign_backend(self):
        with Scheduler(self.backend) as scheduler, Backend.cpu_init() as other:
            self.assertEqual(scheduler.buffer_size(self.backend), 0)
            with self.assertRaisesRegex(ValueError, "does not belong"):
                scheduler.buffer_size(other)


if __name__ == "__main__":
    unittest.main()

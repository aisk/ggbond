import unittest

from ggbond.ggml import Backend, Scheduler


class SchedulerValidationTests(unittest.TestCase):
    def test_constructs_with_latest_scheduler_signature(self):
        with Backend.cpu_init() as backend:
            with Scheduler.from_backends(backend, op_offload=False) as scheduler:
                self.assertIsNotNone(scheduler.ptr)

    def test_rejects_wrong_number_of_buffer_types(self):
        with Backend.cpu_init() as backend:
            with self.assertRaisesRegex(ValueError, "one buffer type per backend"):
                Scheduler.from_backends(backend, bufts=[])

    def test_rejects_null_buffer_type(self):
        with Backend.cpu_init() as backend:
            with self.assertRaisesRegex(ValueError, "buffer types must be non-null"):
                Scheduler.from_backends(backend, bufts=[None])

    def test_rejects_closed_backend(self):
        backend = Backend.cpu_init()
        backend.close()

        with self.assertRaisesRegex(ValueError, "backends must be open"):
            Scheduler.from_backends(backend)

    def test_rejects_non_positive_graph_size(self):
        with Backend.cpu_init() as backend:
            for graph_size in (0, -1):
                with self.subTest(graph_size=graph_size):
                    with self.assertRaisesRegex(ValueError, "graph_size must be positive"):
                        Scheduler.from_backends(backend, graph_size=graph_size)


if __name__ == "__main__":
    unittest.main()

import gc
import unittest

from ggbond.ggml import Backend, Context, F32, GGMLError


class BufferTests(unittest.TestCase):
    def test_alloc_ctx_tensors_returns_a_sized_buffer(self):
        with Backend.cpu_init() as backend, Context(1 << 20, no_alloc=True) as ctx:
            ctx.new_tensor_1d(F32, 64)
            buffer = backend.alloc_ctx_tensors(ctx)

            self.assertGreaterEqual(buffer.size, 64 * 4)
            self.assertTrue(buffer.is_host)
            self.assertIsInstance(buffer.name, str)
            self.assertIsInstance(buffer.buffer_type.name, str)

    def test_backend_close_closes_its_open_buffers_first(self):
        backend = Backend.cpu_init()
        with Context(1 << 20, no_alloc=True) as ctx:
            ctx.new_tensor_1d(F32, 64)
            buffer = backend.alloc_ctx_tensors(ctx)

            backend.close()
            self.assertTrue(buffer.closed)
            with self.assertRaisesRegex(GGMLError, "Buffer is closed"):
                buffer.size

    def test_buffer_close_frees_it_and_is_idempotent(self):
        with Backend.cpu_init() as backend, Context(1 << 20, no_alloc=True) as ctx:
            ctx.new_tensor_1d(F32, 64)
            with backend.alloc_ctx_tensors(ctx) as buffer:
                self.assertFalse(buffer.closed)
            self.assertTrue(buffer.closed)
            buffer.close()
            self.assertEqual(repr(buffer), "<ggml.Buffer closed>")
            # The backend no longer tracks it, so closing the backend is clean.
            gc.collect()

    def test_buffer_is_freed_once_context_and_handle_are_gone(self):
        with Backend.cpu_init() as backend:
            ctx = Context(1 << 20, no_alloc=True)
            ctx.new_tensor_1d(F32, 64)
            backend.alloc_ctx_tensors(ctx)
            self.assertEqual(len(backend._dependents), 1)

            ctx.close()
            gc.collect()
            self.assertEqual(len(backend._dependents), 0)

    def test_buffer_type_has_a_name(self):
        with Backend.cpu_init() as backend:
            self.assertIsInstance(backend.buffer_type.name, str)
            self.assertIsInstance(backend.name, str)
            self.assertEqual(repr(backend), f"<ggml.Backend name={backend.name!r}>")


if __name__ == "__main__":
    unittest.main()

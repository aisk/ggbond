import unittest

from ggbond.ggml import Backend, Context, F32


class BufferTests(unittest.TestCase):
    def test_alloc_ctx_tensors_returns_a_sized_buffer(self):
        with Backend.cpu_init() as backend, Context(1 << 20, no_alloc=True) as ctx:
            ctx.new_tensor_1d(F32, 64)
            buffer = backend.alloc_ctx_tensors(ctx)

            self.assertGreaterEqual(buffer.size, 64 * 4)
            self.assertTrue(buffer.is_host)
            self.assertIsInstance(buffer.name, str)

    def test_backend_close_frees_the_buffers_it_owns(self):
        backend = Backend.cpu_init()
        with Context(1 << 20, no_alloc=True) as ctx:
            ctx.new_tensor_1d(F32, 64)
            buffer = backend.alloc_ctx_tensors(ctx)

        backend.close()
        self.assertIsNone(buffer.ptr)

    def test_released_buffer_is_freed_by_its_own_close(self):
        with Backend.cpu_init() as backend, Context(1 << 20, no_alloc=True) as ctx:
            ctx.new_tensor_1d(F32, 64)
            buffer = backend.alloc_ctx_tensors(ctx).release()

            self.assertEqual(backend._buffers, [])
            buffer.close()
            self.assertIsNone(buffer.ptr)

    def test_buffer_type_has_a_name(self):
        with Backend.cpu_init() as backend:
            self.assertIsInstance(backend.buffer_type().name, str)
            self.assertIsInstance(backend.name, str)


if __name__ == "__main__":
    unittest.main()

import ctypes
import gc
import unittest
import weakref

from ggbond.ggml import Context


class ContextBufferLifetimeTests(unittest.TestCase):
    def test_context_retains_external_buffer(self):
        size = 1 << 20
        raw_buffer = ctypes.create_string_buffer(size)
        raw_buffer_ref = weakref.ref(raw_buffer)
        buffer_pointer = ctypes.cast(raw_buffer, ctypes.c_void_p)

        ctx = Context(size, no_alloc=True, mem_buffer=buffer_pointer)
        del raw_buffer, buffer_pointer
        gc.collect()

        self.assertIsNotNone(raw_buffer_ref())

        ctx.close()
        gc.collect()
        self.assertIsNone(raw_buffer_ref())


if __name__ == "__main__":
    unittest.main()

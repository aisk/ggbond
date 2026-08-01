import unittest

from ggbond.ggml import Backend, Context, F32


class TensorRawTransferTests(unittest.TestCase):
    def test_set_raw_rejects_negative_offset(self):
        with Backend.cpu_init() as backend, Context(1 << 20, no_alloc=True) as ctx:
            tensor = ctx.new_tensor_1d(F32, 4)
            backend.alloc_ctx_tensors(ctx)

            with self.assertRaisesRegex(ValueError, "offset must be non-negative"):
                tensor.set_raw(b"1234", offset=-1)


if __name__ == "__main__":
    unittest.main()

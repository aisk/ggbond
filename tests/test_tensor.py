import unittest

import numpy as np

from ggbond.ggml import Backend, Context, F32, I32, TensorFlag


class TransferTestCase(unittest.TestCase):
    def setUp(self):
        self.backend = Backend.cpu_init()
        self.ctx = Context(1 << 20, no_alloc=True)
        self.addCleanup(self.backend.close)
        self.addCleanup(self.ctx.close)

    def allocated(self, *ne, dtype=F32):
        tensor = self.ctx.new_tensor(dtype, *ne)
        self.backend.alloc_ctx_tensors(self.ctx)
        return tensor


class WriteTests(TransferTestCase):
    def test_write_accepts_the_tensor_shape_or_a_flat_array(self):
        tensor = self.allocated(4, 3)
        value = np.arange(12, dtype=np.float32).reshape(3, 4)

        tensor.write(value)
        np.testing.assert_array_equal(tensor.to_numpy(), value)
        tensor.write(value.reshape(-1) + 1)
        np.testing.assert_array_equal(tensor.to_numpy(), value + 1)
        tensor.write((value + 2).tolist())
        np.testing.assert_array_equal(tensor.to_numpy(), value + 2)

    def test_write_rejects_a_ggml_order_array(self):
        tensor = self.allocated(4, 3)

        with self.assertRaisesRegex(ValueError, r"shape \(4, 3\) does not match tensor shape \(3, 4\)"):
            tensor.write(np.zeros((4, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "does not match"):
            tensor.write(np.zeros(11, dtype=np.float32))

    def test_write_ignores_one_sized_axes(self):
        tensor = self.allocated(5, 1)

        tensor.write(np.arange(5, dtype=np.float32).reshape(5, 1))
        tensor.write(np.arange(5, dtype=np.float32).reshape(1, 5))
        self.assertEqual(tensor.to_numpy().shape, (5,))

    def test_write_raw_accepts_bytes_bytearray_and_arrays(self):
        tensor = self.allocated(4, dtype=I32)
        value = np.array([1, 2, 3, 4], dtype=np.int32)

        for data in (value.tobytes(), bytearray(value.tobytes()), value, memoryview(value.tobytes())):
            tensor.write(np.zeros(4, dtype=np.int32))
            tensor.write_raw(data)
            np.testing.assert_array_equal(tensor.to_numpy(), value)

        tensor.write_raw(np.array([9], dtype=np.int32), offset=8)
        np.testing.assert_array_equal(tensor.to_numpy(), [1, 2, 9, 4])

    def test_write_raw_checks_bounds(self):
        tensor = self.allocated(4)

        with self.assertRaisesRegex(ValueError, "offset must be non-negative"):
            tensor.write_raw(b"1234", offset=-1)
        with self.assertRaisesRegex(ValueError, "exceed tensor size"):
            tensor.write_raw(b"\0" * 8, offset=12)

    def test_copy_to_copies_between_allocated_tensors(self):
        source = self.ctx.new_tensor_1d(F32, 4)
        target = self.ctx.new_tensor_1d(F32, 4)
        other = self.ctx.new_tensor_1d(F32, 5)
        self.backend.alloc_ctx_tensors(self.ctx)
        source.write([1, 2, 3, 4])

        source.copy_to(target)

        np.testing.assert_array_equal(target.to_numpy(), [1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, "layouts differ"):
            source.copy_to(other)


class FlagTests(unittest.TestCase):
    def test_flags_report_what_was_set(self):
        with Context(1 << 20, no_alloc=True) as ctx:
            tensor = ctx.new_tensor_1d(F32, 4)
            self.assertEqual(tensor.flags, TensorFlag(0))

            tensor.set_input().set_output()
            self.assertEqual(tensor.flags, TensorFlag.INPUT | TensorFlag.OUTPUT)

            self.assertIn(TensorFlag.PARAM, ctx.new_tensor_1d(F32, 4).set_param().flags)
            with self.assertRaisesRegex(ValueError, "leaf"):
                tensor.relu().set_param()
            with self.assertRaisesRegex(ValueError, "scalar"):
                tensor.set_loss()

    def test_backend_reports_op_support(self):
        with Backend.cpu_init() as backend, Context(1 << 20, no_alloc=True) as ctx:
            self.assertTrue(backend.supports_op(ctx.new_tensor_1d(F32, 4).relu()))


if __name__ == "__main__":
    unittest.main()

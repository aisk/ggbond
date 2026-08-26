import unittest

from ggbond.ggml import Context, F32


class TensorShapeTests(unittest.TestCase):
    def test_ne_trims_trailing_ones_but_ne4_does_not(self):
        with Context(1 << 20, no_alloc=True) as ctx:
            tensor = ctx.new_tensor_2d(F32, 6, 1)

            self.assertEqual(tensor.ne, (6,))
            self.assertEqual(tensor.ne4, (6, 1, 1, 1))
            self.assertEqual(tensor.n_dims, 1)

    def test_nb4_keeps_the_row_stride_of_a_single_row_tensor(self):
        with Context(1 << 20, no_alloc=True) as ctx:
            tensor = ctx.new_tensor_2d(F32, 6, 1)

            # .nb trims along with .ne and loses the row stride; .nb4 keeps it.
            self.assertEqual(len(tensor.nb), 1)
            self.assertEqual(tensor.nb4[1], 6 * tensor.element_size)

    def test_shape_is_ne_reversed(self):
        with Context(1 << 20, no_alloc=True) as ctx:
            tensor = ctx.new_tensor_3d(F32, 2, 3, 4)

            self.assertEqual(tensor.ne, (2, 3, 4))
            self.assertEqual(tensor.shape, (4, 3, 2))


if __name__ == "__main__":
    unittest.main()

import unittest

from ggbond.ggml import Context, F32


class TensorShapeTests(unittest.TestCase):
    def test_ne_is_the_four_dims_ggml_stores(self):
        with Context(1 << 20, no_alloc=True) as ctx:
            tensor = ctx.new_tensor_2d(F32, 6, 1)

            self.assertEqual(tensor.ne, (6, 1, 1, 1))
            self.assertEqual(tensor.ne[1], 1)
            self.assertEqual(tensor.n_dims, 1)

    def test_nb_keeps_the_row_stride_of_a_single_row_tensor(self):
        with Context(1 << 20, no_alloc=True) as ctx:
            tensor = ctx.new_tensor_2d(F32, 6, 1)

            self.assertEqual(len(tensor.nb), 4)
            self.assertEqual(tensor.nb[1], 6 * tensor.element_size)

    def test_shape_is_the_numpy_order_view(self):
        with Context(1 << 20, no_alloc=True) as ctx:
            tensor = ctx.new_tensor_3d(F32, 2, 3, 4)

            self.assertEqual(tensor.ne, (2, 3, 4, 1))
            self.assertEqual(tensor.shape, (4, 3, 2))
            # Trailing 1-sized ggml dims do not count, following ggml_n_dims.
            self.assertEqual(ctx.new_tensor_2d(F32, 6, 1).shape, (6,))
            self.assertEqual(ctx.new_tensor_2d(F32, 1, 6).shape, (6, 1))

    def test_predicates(self):
        with Context(1 << 20, no_alloc=True) as ctx:
            matrix = ctx.new_tensor_2d(F32, 3, 4)

            self.assertTrue(matrix.is_matrix)
            self.assertFalse(matrix.is_vector)
            self.assertTrue(matrix.is_contiguous)
            self.assertFalse(matrix.is_view)
            self.assertFalse(matrix.is_quantized)

            view = matrix.T
            self.assertTrue(view.is_view)
            self.assertTrue(view.is_transposed)
            self.assertFalse(view.is_contiguous)
            self.assertEqual(view.view_src, matrix)
            self.assertIsNone(matrix.view_src)
            self.assertTrue(ctx.new_tensor_1d(F32, 1).is_scalar)


if __name__ == "__main__":
    unittest.main()

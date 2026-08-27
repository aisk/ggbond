import unittest

import numpy as np

from ggbond.ggml import Backend, Context, F32, GAllocr, graph_overhead_custom, tensor_overhead


def evaluate(build):
    """Run ``build(ctx)`` and download its single output."""
    with Backend.cpu_init() as backend, Context(1 << 22, no_alloc=True) as ctx:
        uploads, output = build(ctx)
        output.set_output()
        backend.alloc_ctx_tensors(ctx)
        for tensor, value in uploads.items():
            tensor.set(value)
        graph = ctx.new_graph().build_forward_expand(output)
        with GAllocr.from_backend(backend) as allocator:
            allocator.alloc_graph(graph)
            backend.compute(graph)
            backend.synchronize()
            return output.to_numpy()


class ToNumpyTests(unittest.TestCase):
    def test_to_numpy_allocates_the_right_shape_and_dtype(self):
        source = np.arange(6, dtype=np.float32).reshape(2, 3)

        def build(ctx):
            a = ctx.new_tensor_2d(F32, 3, 2).set_input()
            return {a: source}, a.scale(2.0)

        result = evaluate(build)
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, source * 2)


class OperatorTests(unittest.TestCase):
    def setUp(self):
        self.left = np.arange(6, dtype=np.float32).reshape(2, 3)
        self.right = np.ones((2, 3), dtype=np.float32) * 3

    def run_operator(self, apply):
        def build(ctx):
            a = ctx.new_tensor_2d(F32, 3, 2).set_input()
            b = ctx.new_tensor_2d(F32, 3, 2).set_input()
            return {a: self.left, b: self.right}, apply(a, b)

        return evaluate(build)

    def test_elementwise_operators(self):
        np.testing.assert_allclose(
            self.run_operator(lambda a, b: a + b), self.left + self.right
        )
        np.testing.assert_allclose(
            self.run_operator(lambda a, b: a - b), self.left - self.right
        )
        np.testing.assert_allclose(
            self.run_operator(lambda a, b: a * b), self.left * self.right
        )
        np.testing.assert_allclose(
            self.run_operator(lambda a, b: a / b), self.left / self.right
        )
        np.testing.assert_allclose(
            self.run_operator(lambda a, b: -a), -self.left
        )

    def test_scalar_multiplication_uses_scale(self):
        np.testing.assert_allclose(
            self.run_operator(lambda a, b: a * 2.0), self.left * 2
        )
        np.testing.assert_allclose(
            self.run_operator(lambda a, b: 2.0 * a), self.left * 2
        )
        np.testing.assert_allclose(
            self.run_operator(lambda a, b: a / 4.0), self.left / 4
        )

    def test_mul_mat_has_no_operator(self):
        # ggml's mul_mat takes its operands transposed relative to numpy, so it
        # deliberately stays a named method.
        with Context(1 << 20, no_alloc=True) as ctx:
            a = ctx.new_tensor_2d(F32, 3, 2)
            with self.assertRaises(TypeError):
                a @ a


class ContextSizingTests(unittest.TestCase):
    def test_for_tensors_matches_the_hand_rolled_arithmetic(self):
        with Context.for_tensors(8, graph_size=64) as ctx:
            for i in range(8):
                ctx.new_tensor_1d(F32, 4, name=f"t{i}")
            graph = ctx.new_graph(size=64)

            self.assertEqual(len(list(ctx.tensors())), 8)
            self.assertEqual(len(graph), 0)

    def test_for_tensors_rejects_negative_counts(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            Context.for_tensors(-1)

    def test_overhead_helpers_are_still_exported(self):
        self.assertGreater(tensor_overhead(), 0)
        self.assertGreater(graph_overhead_custom(16, False), 0)


class GenericTensorFactoryTests(unittest.TestCase):
    def test_new_tensor_takes_dims_in_ggml_order(self):
        source = np.zeros((2, 3, 4), dtype=np.float32)

        with Context(1 << 20, no_alloc=True) as ctx:
            tensor = ctx.new_tensor(F32, *reversed(source.shape), name="t")

            self.assertEqual(tensor.shape, source.shape)
            self.assertEqual(tensor.name, "t")

    def test_new_tensor_rejects_bad_dimension_counts(self):
        with Context(1 << 20, no_alloc=True) as ctx:
            with self.assertRaisesRegex(ValueError, "between 1 and 4"):
                ctx.new_tensor(F32)
            with self.assertRaisesRegex(ValueError, "between 1 and 4"):
                ctx.new_tensor(F32, 1, 2, 3, 4, 5)


if __name__ == "__main__":
    unittest.main()

import inspect
import unittest

import numpy as np

from ggbond.ggml import Backend, Context, F32, GAllocr, I32, ROPE_TYPE_NEOX, Tensor


def compute(build):
    """Build a graph with ``build(ctx)`` and return the outputs as numpy arrays.

    ``build`` returns ``(uploads, outputs)`` where ``uploads`` maps a tensor to
    the array to upload into it.
    """
    with Backend.cpu_init() as backend, Context(1 << 22, no_alloc=True) as ctx:
        uploads, outputs = build(ctx)
        for tensor in outputs:
            tensor.set_output()
        backend.alloc_ctx_tensors(ctx)
        for tensor, value in uploads.items():
            tensor.set(value)

        graph = ctx.new_graph()
        for tensor in outputs:
            graph.build_forward_expand(tensor)
        with GAllocr.from_backend(backend) as allocator:
            allocator.alloc_graph(graph)
            backend.compute(graph)
            backend.synchronize()

        return [
            tensor.get(np.empty(tensor.shape, dtype=np.float32))
            for tensor in outputs
        ]


class GeneratedOpTests(unittest.TestCase):
    def test_unary_math_ops(self):
        source = np.arange(1, 13, dtype=np.float32).reshape(3, 4)

        def build(ctx):
            a = ctx.new_tensor_2d(F32, 4, 3).set_input()
            return {a: source}, [a.sqr(), a.sqrt(), a.log(), a.neg(), a.sum()]

        sqr, sqrt, log, neg, total = compute(build)
        np.testing.assert_allclose(sqr, source**2)
        np.testing.assert_allclose(sqrt, np.sqrt(source), rtol=1e-6)
        np.testing.assert_allclose(log, np.log(source), rtol=1e-6)
        np.testing.assert_allclose(neg, -source)
        np.testing.assert_allclose(total, [source.sum()], rtol=1e-6)

    def test_concat_and_pad_shapes(self):
        source = np.arange(12, dtype=np.float32).reshape(3, 4)

        def build(ctx):
            a = ctx.new_tensor_2d(F32, 4, 3).set_input()
            # dim 1 is ggml's second dimension, i.e. numpy's rows.
            return {a: source}, [a.concat(a, 1), a.pad(2, 0, 0, 0)]

        joined, padded = compute(build)
        np.testing.assert_allclose(joined, np.concatenate([source, source], axis=0))
        self.assertEqual(padded.shape, (3, 6))
        np.testing.assert_allclose(padded[:, :4], source)
        np.testing.assert_allclose(padded[:, 4:], 0)

    def test_soft_max_ext_applies_its_scale(self):
        source = np.arange(12, dtype=np.float32).reshape(3, 4)

        def build(ctx):
            a = ctx.new_tensor_2d(F32, 4, 3).set_input()
            return {a: source}, [a.soft_max_ext(None, scale=0.5)]

        (result,) = compute(build)
        scaled = source * 0.5
        expected = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_rope_leaves_position_zero_untouched(self):
        source = np.arange(8, dtype=np.float32).reshape(1, 8)

        def build(ctx):
            a = ctx.new_tensor_3d(F32, 8, 1, 1).set_input()
            positions = ctx.new_tensor_1d(I32, 1).set_input()
            rotated = a.rope(positions, 8, ROPE_TYPE_NEOX)
            return {a: source, positions: np.zeros(1, dtype=np.int32)}, [rotated]

        (result,) = compute(build)
        np.testing.assert_allclose(result.reshape(1, 8), source, atol=1e-6)

    def test_view_4d_reads_a_strided_block(self):
        source = np.arange(24, dtype=np.float32)

        def build(ctx):
            a = ctx.new_tensor_1d(F32, 24).set_input()
            item = a.element_size
            block = a.view_4d(2, 2, 2, 1, 2 * item, 4 * item, 8 * item, 8 * item)
            return {a: source}, [block.cont()]

        (result,) = compute(build)
        np.testing.assert_allclose(result.reshape(-1), source[8:16])


class OpGenerationTests(unittest.TestCase):
    def test_generated_methods_keep_a_real_signature(self):
        parameters = inspect.signature(Tensor.view_2d).parameters

        self.assertEqual(list(parameters), ["self", "ne0", "ne1", "nb1", "offset"])
        self.assertEqual(parameters["offset"].default, 0)

    def test_generated_methods_carry_the_wrapped_function_name(self):
        self.assertIn("ggml_soft_max_ext", Tensor.soft_max_ext.__doc__)


if __name__ == "__main__":
    unittest.main()

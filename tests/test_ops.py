import inspect
import unittest

import numpy as np

import ggml as _ggml

from ggbond.ggml import (
    Backend, Context, F32, GAllocr, I32, Op, RopeType, ScaleMode, SortOrder, Tensor,
)


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
            tensor.write(value)

        graph = ctx.new_graph()
        for tensor in outputs:
            graph.build_forward_expand(tensor)
        with GAllocr.from_backend(backend) as allocator:
            allocator.alloc_graph(graph)
            backend.compute(graph)

        return [tensor.to_numpy() for tensor in outputs]


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
            rotated = a.rope(positions, 8, RopeType.NEOX)
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


    def test_rounding_and_cumulative_ops(self):
        source = np.array([[-1.5, 0.25, 2.75], [3.5, -0.5, 1.0]], dtype=np.float32)

        def build(ctx):
            a = ctx.new_tensor_2d(F32, 3, 2).set_input()
            return {a: source}, [a.floor(), a.ceil(), a.sgn(), a.cumsum(), a.cont_1d(6)]

        floor, ceil, sgn, cumsum, flat = compute(build)
        np.testing.assert_array_equal(floor, np.floor(source))
        np.testing.assert_array_equal(ceil, np.ceil(source))
        np.testing.assert_array_equal(sgn, np.sign(source))
        np.testing.assert_allclose(cumsum, np.cumsum(source, axis=1))
        np.testing.assert_array_equal(flat, source.reshape(-1))

    def test_argsort_arange_and_upscale(self):
        source = np.array([[3.0, 1.0, 2.0]], dtype=np.float32)

        def build(ctx):
            a = ctx.new_tensor_2d(F32, 3, 1).set_input()
            image = ctx.new_tensor_4d(F32, 3, 1, 1, 1).set_input()
            return {a: source, image: source}, [
                ctx.arange(1, 7, 2),
                image.upscale(2, ScaleMode.NEAREST),
                a.argsort(SortOrder.DESC),
            ]

        with Backend.cpu_init() as backend, Context(1 << 22, no_alloc=True) as ctx:
            uploads, (steps, upscaled, order) = build(ctx)
            graph = ctx.new_graph()
            for tensor in (steps, upscaled, order):
                graph.build_forward_expand(tensor.set_output())
            self.assertIs(order.dtype, I32)
            with GAllocr.from_backend(backend) as allocator:
                allocator.alloc_graph(graph)
                for tensor, value in uploads.items():
                    tensor.write(value)
                backend.compute(graph)

                np.testing.assert_array_equal(steps.to_numpy(), [1, 3, 5])
                np.testing.assert_array_equal(
                    upscaled.to_numpy().reshape(2, 6), [[3, 3, 1, 1, 2, 2]] * 2
                )
                np.testing.assert_array_equal(order.to_numpy(), [0, 2, 1])

    def test_inplace_ops_write_into_their_operand(self):
        source = np.array([1.0, -2.0, 3.0], dtype=np.float32)

        def build(ctx):
            a = ctx.new_tensor_1d(F32, 3).set_input()
            doubled = a.scale(2.0)
            negated = doubled.neg_inplace()
            self.assertEqual(negated.view_src, doubled)
            return {a: source}, [negated, doubled]

        negated, doubled = compute(build)
        np.testing.assert_array_equal(negated, -2 * source)
        np.testing.assert_array_equal(doubled, -2 * source)

    def test_set_1d_writes_into_a_view(self):
        def build(ctx):
            a = ctx.new_tensor_1d(F32, 4).set_input()
            b = ctx.new_tensor_1d(F32, 2).set_input()
            updated = a.dup().set_1d(b, 1 * a.element_size)
            return {a: np.zeros(4, np.float32), b: np.array([5, 6], np.float32)}, [updated]

        (updated,) = compute(build)
        np.testing.assert_array_equal(updated, [0, 5, 6, 0])

    def test_reshape_like_another_tensor(self):
        with Context(1 << 20, no_alloc=True) as ctx:
            a = ctx.new_tensor_2d(F32, 4, 3)

            self.assertEqual(a.reshape(ctx.new_tensor_2d(F32, 6, 2)).ne, (6, 2, 1, 1))
            with self.assertRaisesRegex(ValueError, "reshape: element counts differ"):
                a.reshape(ctx.new_tensor_1d(F32, 5))
            with self.assertRaisesRegex(ValueError, "empty range"):
                ctx.arange(3, 1)


class OpGenerationTests(unittest.TestCase):
    def test_every_op_exists_on_both_tensor_and_context(self):
        from ggbond.ggml.tensor import _op_specs

        names = [name for name, *_ in _op_specs()]
        self.assertEqual(len(names), len(set(names)))
        for name, fn_name, params, *_ in _op_specs():
            with self.subTest(op=name):
                on_tensor = list(inspect.signature(getattr(Tensor, name)).parameters)
                on_context = list(inspect.signature(getattr(Context, name)).parameters)
                self.assertEqual(on_context, ["self", "a"] + on_tensor[1:])
                # The tables must describe every argument after ggml's ctx and a.
                binding = inspect.signature(getattr(_ggml, fn_name)).parameters
                self.assertEqual(len(binding), 2 + len(params))

    def test_the_stub_file_matches_the_op_tables(self):
        from pathlib import Path

        import ggbond.ggml.ops
        from ggbond.ggml.tensor import render_ops_stub

        stub = Path(ggbond.ggml.ops.__file__).with_suffix(".pyi")
        self.assertEqual(
            stub.read_text(), render_ops_stub(),
            "ops.pyi is out of date; run: python scripts/gen_stubs.py",
        )
        compile(stub.read_text(), str(stub), "exec")

    def test_no_class_attribute_shadows_an_op(self):
        from ggbond.ggml.ops import ContextOps, TensorOps
        from ggbond.ggml.tensor import _op_specs

        for name, *_ in _op_specs():
            self.assertIn(name, vars(TensorOps))
            self.assertIn(name, vars(ContextOps))
            self.assertNotIn(name, vars(Tensor))
            self.assertNotIn(name, vars(Context))

    def test_inplace_variants_are_generated_from_the_bindings(self):
        self.assertTrue(hasattr(Tensor, "add_inplace"))
        self.assertTrue(hasattr(Context, "rope_ext_inplace"))
        # ggml has no ggml_mul_mat_inplace, so neither do we.
        self.assertFalse(hasattr(Tensor, "mul_mat_inplace"))
        self.assertIn("returns a view", Tensor.add_inplace.__doc__)

    def test_context_form_documents_ggml_argument_order(self):
        parameters = inspect.signature(Context.conv_1d).parameters

        self.assertEqual(list(parameters), ["self", "a", "data", "s0", "p0", "d0"])
        self.assertIn("``a`` is the kernel", Context.conv_1d.__doc__)
        with Context(1 << 20, no_alloc=True) as ctx:
            self.assertIs(ctx.sqr(ctx.new_tensor_1d(F32, 2)).op, Op.SQR)

    def test_generated_methods_keep_a_real_signature(self):
        parameters = inspect.signature(Tensor.view_2d).parameters

        self.assertEqual(list(parameters), ["self", "ne0", "ne1", "nb1", "offset"])
        self.assertEqual(parameters["offset"].default, 0)

    def test_generated_methods_carry_the_wrapped_function_name(self):
        self.assertIn("ggml_soft_max_ext", Tensor.soft_max_ext.__doc__)


if __name__ == "__main__":
    unittest.main()

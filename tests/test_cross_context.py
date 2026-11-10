import unittest

import numpy as np

from ggbond.ggml import Backend, Context, F32, GAllocr, GGMLError, Op


class ContextOpTests(unittest.TestCase):
    def test_context_op_lands_in_that_context(self):
        with Context(1 << 20, no_alloc=True) as weights, \
                Context(1 << 20, no_alloc=True) as graph_ctx:
            a = weights.new_tensor_2d(F32, 3, 4, name="a")
            b = weights.new_tensor_2d(F32, 3, 2, name="b")
            used = weights.used_mem

            product = graph_ctx.mul_mat(a, b)

            self.assertIs(product.ctx, graph_ctx)
            self.assertIs(product.op, Op.MUL_MAT)
            self.assertEqual(product.ne, (4, 2, 1, 1))
            self.assertEqual(weights.used_mem, used)
            self.assertEqual(product.src, (a, b))
            self.assertIs(product.src[0].ctx, weights)

    def test_tensor_method_is_sugar_for_its_own_context(self):
        with Context(1 << 20, no_alloc=True) as ctx:
            a = ctx.new_tensor_1d(F32, 4)
            used = ctx.used_mem

            self.assertIs(a.relu().ctx, ctx)
            self.assertGreater(ctx.used_mem, used)

    def test_context_ops_take_keywords_defaults_and_checks(self):
        with Context(1 << 20, no_alloc=True) as weights, \
                Context(1 << 20, no_alloc=True) as ctx:
            a = weights.new_tensor_2d(F32, 3, 4)
            b = weights.new_tensor_2d(F32, 5, 2)

            self.assertIs(ctx.soft_max_ext(a, mask=None, scale=2.0).op, Op.SOFT_MAX)
            with self.assertRaisesRegex(ValueError, "mul_mat"):
                ctx.mul_mat(a, b)

    def test_a_graph_can_span_two_contexts(self):
        weight = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)

        with Backend.cpu_init() as backend, \
                Context(1 << 20, no_alloc=True) as weights:
            stored = weights.new_tensor_1d(F32, 4, name="w").set_input()
            backend.alloc_ctx_tensors(weights)
            stored.write(weight)

            # Result nodes land in the graph context, not the weight one.
            with Context(1 << 20, no_alloc=True) as graph_ctx:
                doubled = graph_ctx.scale(stored, 2.0).set_output()
                graph = graph_ctx.new_graph().build_forward_expand(doubled)
                with GAllocr.from_backend(backend) as allocator:
                    allocator.alloc_graph(graph)
                    backend.compute(graph)
                    # doubled lives in the allocator's buffer, so read it here.
                    result = doubled.to_numpy()

        np.testing.assert_allclose(result, weight * 2)


class GraphLookupTests(unittest.TestCase):
    def test_graph_lookups_resolve_the_owning_context(self):
        weights = Context(1 << 20, no_alloc=True)
        with Context(1 << 20, no_alloc=True) as graph_ctx:
            w = weights.new_tensor_1d(F32, 4, name="w")
            x = graph_ctx.new_tensor_1d(F32, 4, name="x")
            total = graph_ctx.add(w, x)
            total.name = "total"
            graph = graph_ctx.new_graph().build_forward_expand(total)

            found = graph.get_tensor("w")
            self.assertEqual(found, w)
            self.assertIs(found.ctx, weights)
            self.assertIs(graph.get_tensor("total").ctx, graph_ctx)
            self.assertEqual({leaf.name for leaf in graph.leafs()}, {"w", "x"})
            self.assertEqual([node.name for node in graph], ["total"])
            self.assertEqual(graph.node(-1), total)

            # Closing the weight context invalidates the weight, not a crash.
            weights.close()
            with self.assertRaisesRegex(GGMLError, "context was closed"):
                found.name
            with self.assertRaisesRegex(GGMLError, "closed or unknown"):
                graph.get_tensor("w")

    def test_wrappers_of_one_tensor_are_equal_and_hash_alike(self):
        with Context(1 << 20, no_alloc=True) as ctx:
            a = ctx.new_tensor_1d(F32, 4, name="a")
            b = ctx.new_tensor_1d(F32, 4, name="b")

            again = ctx.get_tensor("a")
            self.assertIsNot(again, a)
            self.assertEqual(again, a)
            self.assertNotEqual(a, b)
            self.assertEqual(len({a, again, b}), 2)


if __name__ == "__main__":
    unittest.main()

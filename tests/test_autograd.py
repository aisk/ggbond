import unittest

import numpy as np

from ggbond.ggml import Backend, Context, F32, GAllocr, GGMLError


class BackwardTests(unittest.TestCase):
    def test_gradient_of_a_sum_of_squares(self):
        value = np.array([1.0, -2.0, 3.0], dtype=np.float32)

        with Backend.cpu_init() as backend, Context(1 << 22, no_alloc=True) as ctx:
            x = ctx.new_tensor_1d(F32, 3, name="x").set_param()
            loss = x.sqr().sum().set_loss()

            graph = ctx.new_graph(grads=True).build_forward_expand(loss)
            graph.build_backward_expand()
            with GAllocr.from_backend(backend) as allocator:
                allocator.alloc_graph(graph)
                x.write(value)
                graph.reset()
                backend.compute(graph)

                np.testing.assert_allclose(loss.to_numpy(), [np.sum(value**2)])
                np.testing.assert_allclose(graph.get_grad(x).to_numpy(), 2 * value)

    def test_backward_needs_a_grads_graph(self):
        with Context(1 << 22, no_alloc=True) as ctx:
            x = ctx.new_tensor_1d(F32, 3).set_param()
            graph = ctx.new_graph().build_forward_expand(x.sqr().sum().set_loss())

            with self.assertRaisesRegex(GGMLError, "grads=True"):
                graph.build_backward_expand()
            self.assertIsNone(graph.get_grad(x))

    def test_clear_empties_the_graph(self):
        with Context(1 << 22, no_alloc=True) as ctx:
            x = ctx.new_tensor_1d(F32, 3)
            graph = ctx.new_graph().build_forward_expand(x.relu())
            self.assertEqual((graph.n_nodes, graph.n_leafs), (1, 1))

            graph.clear()
            self.assertEqual((graph.n_nodes, graph.n_leafs), (0, 0))
            graph.build_forward_expand(x.relu().neg())
            self.assertEqual(len(graph), 2)


if __name__ == "__main__":
    unittest.main()

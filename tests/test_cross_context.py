import unittest

import numpy as np

from ggbond.ggml import Backend, Context, F32, GAllocr


class ContextRefTests(unittest.TestCase):
    def test_ref_shares_the_tensor_but_rebinds_the_context(self):
        with Context(1 << 20, no_alloc=True) as weights, \
                Context(1 << 20, no_alloc=True) as graph_ctx:
            original = weights.new_tensor_1d(F32, 4, name="w")
            rebound = graph_ctx.ref(original)

            self.assertIs(rebound.ptr, original.ptr)
            self.assertIs(rebound.ctx, graph_ctx)
            self.assertEqual(rebound.name, "w")

    def test_a_graph_can_span_two_contexts(self):
        weight = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)

        with Backend.cpu_init() as backend, \
                Context(1 << 20, no_alloc=True) as weights:
            stored = weights.new_tensor_1d(F32, 4, name="w").set_input()
            backend.alloc_ctx_tensors(weights)
            stored.set(weight)

            # Result nodes must land in the graph context, not the weight one.
            with Context(1 << 20, no_alloc=True) as graph_ctx:
                doubled = graph_ctx.ref(stored).scale(2.0).set_output()
                graph = graph_ctx.new_graph().build_forward_expand(doubled)
                with GAllocr.from_backend(backend) as allocator:
                    allocator.alloc_graph(graph)
                    backend.compute(graph)
                    backend.synchronize()
                    # doubled lives in the allocator's buffer, so read it here.
                    result = doubled.get(np.empty(doubled.shape, dtype=np.float32))

        np.testing.assert_allclose(result, weight * 2)


if __name__ == "__main__":
    unittest.main()

import unittest

from ggbond.ggml import Context, F32, LogLevel, log_set


class LogTests(unittest.TestCase):
    def test_log_set_routes_ggml_output_to_python(self):
        seen = []
        log_set(lambda level, text: seen.append((level, text)))
        self.addCleanup(log_set, None)

        with Context(1 << 20, no_alloc=True) as ctx:
            ctx.new_graph().build_forward_expand(ctx.new_tensor_1d(F32, 3).relu()).print()

        self.assertTrue(seen)
        self.assertTrue(all(isinstance(level, LogLevel) for level, _ in seen))
        self.assertIn("n_nodes", "".join(text for _, text in seen))

    def test_a_raising_callback_does_not_propagate(self):
        def boom(level, text):
            raise RuntimeError("no")

        log_set(boom)
        self.addCleanup(log_set, None)
        with Context(1 << 20, no_alloc=True) as ctx:
            ctx.new_graph().print()


if __name__ == "__main__":
    unittest.main()

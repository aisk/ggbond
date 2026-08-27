import os
import tempfile
import unittest

from ggbond.ggml import Context, F32, GGMLError
from ggbond.ggml.errors import check_status, status_name


class StatusReportingTests(unittest.TestCase):
    def test_status_name_uses_ggmls_own_wording(self):
        self.assertIn("success", status_name(0))
        self.assertIn("failed", status_name(-1))

    def test_check_status_reports_the_name_alongside_the_code(self):
        with self.assertRaisesRegex(GGMLError, r"demo failed with status -1 \(.*failed.*\)"):
            check_status(-1, "demo")


class GraphDumpTests(unittest.TestCase):
    def test_dump_dot_writes_the_receiver(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = os.path.join(directory.name, "graph.dot")

        with Context(1 << 20, no_alloc=True) as ctx:
            a = ctx.new_tensor_1d(F32, 4, name="a")
            graph = ctx.new_graph().build_forward_expand(a.scale(2.0))
            graph.dump_dot(path)

        with open(path) as handle:
            contents = handle.read()
        self.assertIn("digraph G", contents)
        self.assertIn("node_0", contents)


if __name__ == "__main__":
    unittest.main()

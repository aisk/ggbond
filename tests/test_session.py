import os
import tempfile
import unittest

import numpy as np

import ggbond
from ggbond import F16, F32, I32, GGMLError, GraphScope, Session, Tensor, Variable
from ggbond.ggml import GGUF, Backend, Context, Device, DeviceType


class SessionDeviceTests(unittest.TestCase):
    def test_default_session_ends_with_one_cpu_backend(self):
        with Session() as sess:
            backends = sess.backends
            self.assertEqual(backends[-1].device.type, DeviceType.CPU)
            self.assertEqual(sum(b.device.type == DeviceType.CPU for b in backends), 1)
            self.assertEqual(sess.sched.backends, backends)
            self.assertEqual(sess.n_threads, ggbond.default_n_threads())
            self.assertGreaterEqual(sess.n_threads, 1)
            self.assertLessEqual(sess.n_threads, os.cpu_count())
        self.assertTrue(sess.closed)
        self.assertTrue(all(backend.closed for backend in backends))
        sess.close()
        with self.assertRaisesRegex(GGMLError, "Session is closed"):
            sess.graph()
        with self.assertRaisesRegex(GGMLError, "Session is closed"):
            sess.backends

    def test_device_by_name_or_type(self):
        self.assertIn("CPU", Session.devices())
        for device in ("cpu", "CPU"):
            with Session(device=device) as sess:
                self.assertEqual(len(sess.backends), 1)
        with self.assertRaisesRegex(ValueError, "no device 'nosuch'; available devices"):
            Session(device="nosuch")

    def test_a_cpu_backend_follows_a_device_that_is_not_the_cpu(self):
        other = next((d for d in Device.all() if d.type != DeviceType.CPU), None)
        if other is None:
            self.skipTest("no non-CPU device registered")
        with Session(device=other.name) as sess:
            self.assertEqual(len(sess.backends), 2)
            self.assertEqual(sess.backends[0].device.name, other.name)
            self.assertEqual(sess.backends[1].device.type, DeviceType.CPU)
            with sess.graph() as g:
                np.testing.assert_allclose(g.run(g.input(np.ones(4, np.float32)) * 3.0), 3.0)
            # Variables can be placed on any of the session's devices.
            on_cpu = sess.variables({"v": (F32, 4)}, device="cpu")
            self.assertTrue(on_cpu["v"].ptr.contents.buffer)
            with self.assertRaisesRegex(ValueError, "does not run on a device 'nosuch'"):
                sess.variables({"v": (F32, 4)}, device="nosuch")

    def test_passed_backends_stay_open(self):
        with Backend.cpu_init() as backend:
            with Session(backend) as sess:
                self.assertEqual(sess.backends, (backend,))
                with sess.graph() as g:
                    np.testing.assert_allclose(g.run(g.input(np.ones(4, np.float32)) * 3.0), 3.0)
            self.assertFalse(backend.closed)

    def test_rejects_bad_arguments(self):
        with self.assertRaisesRegex(ValueError, "graph_size must be positive"):
            Session(graph_size=0)
        with Backend.cpu_init() as backend:
            with self.assertRaisesRegex(ValueError, "either backends or device"):
                Session(backend, device="cpu")
        backend = Backend.cpu_init()
        backend.close()
        with self.assertRaisesRegex(GGMLError, "Backend is closed"):
            Session(backend)


class SessionTestCase(unittest.TestCase):
    def setUp(self):
        self.sess = Session(device="cpu")
        self.addCleanup(self.sess.close)


class VariableTests(SessionTestCase):
    def test_variables_are_zeroed_named_and_writable(self):
        v = self.sess.variables({"a": (F32, 3, 2), "b": (I32, 4)})
        self.assertEqual(set(v), {"a", "b"})
        self.assertIsInstance(v["a"], Variable)
        self.assertIsInstance(v["a"], Tensor)
        self.assertEqual(v["a"].name, "a")
        self.assertEqual(v["a"].shape, (2, 3))
        np.testing.assert_array_equal(v["a"].to_numpy(), np.zeros((2, 3), np.float32))
        np.testing.assert_array_equal(v["b"].to_numpy(), np.zeros(4, np.int32))
        v["b"].write([1, 2, 3, 4])
        np.testing.assert_array_equal(v["b"].to_numpy(), [1, 2, 3, 4])
        self.assertEqual(self.sess.variables({}), {})
        single = self.sess.variable(F32, 5, name="s")
        self.assertEqual((single.name, single.shape), ("s", (5,)))

    def test_a_bad_spec_raises(self):
        with self.assertRaisesRegex(ValueError, "expected \\(dtype, ne0"):
            self.sess.variables({"a": F32})

    def test_load_arrays_takes_type_and_shape_from_the_arrays(self):
        a = np.arange(6, dtype=np.float32).reshape(2, 3)
        v = self.sess.load_arrays({"a": a, "ids": np.array([3, 1], np.int32), "s": np.float32(2)})
        self.assertEqual(v["a"].ne[:2], (3, 2))
        np.testing.assert_array_equal(v["a"].to_numpy(), a)
        self.assertEqual(v["ids"].dtype, I32)
        np.testing.assert_array_equal(v["s"].to_numpy(), [2.0])

    def test_variables_die_with_the_session(self):
        v = self.sess.load_arrays({"a": np.ones(3, np.float32)})["a"]
        self.sess.close()
        with self.assertRaises(GGMLError):
            v.to_numpy()

    def test_expressions_on_a_variable_need_an_open_graph(self):
        v = self.sess.load_arrays({"a": np.ones(3, np.float32)})["a"]
        with self.assertRaisesRegex(GGMLError, "scale: no graph is open"):
            v.scale(2.0)
        with self.assertRaisesRegex(GGMLError, "no graph is open"):
            v + v

    def test_load_gguf_reads_data_and_metadata(self):
        w = np.arange(12, dtype=np.float32).reshape(3, 4)
        h = np.arange(4, dtype=np.float16)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.gguf")
            with GGUF.init_empty() as out, Context(1 << 20, no_alloc=False) as ctx:
                out.set_val_str("general.architecture", "tiny")
                out.set_val_u32("tiny.width", 4)
                for name, array in (("w", w), ("h", h)):
                    dtype = F32 if array.dtype == np.float32 else F16
                    tensor = ctx.new_tensor(dtype, *reversed(array.shape), name=name)
                    out.add_tensor(tensor)
                    out.set_tensor_data(name, array.tobytes())
                out.write_to_file(path)

            variables, gguf = self.sess.load_gguf(path)
            self.assertEqual(set(variables), {"w", "h"})
            np.testing.assert_array_equal(variables["w"].to_numpy(), w)
            np.testing.assert_array_equal(variables["h"].to_numpy(), h)
            self.assertEqual(gguf["general.architecture"], "tiny")
            self.assertEqual(gguf["tiny.width"], 4)
            with self.sess.graph() as g:
                x = g.input(np.ones((2, 4), np.float32))
                np.testing.assert_allclose(g.run(variables["w"].mul_mat(x)), np.ones((2, 4)) @ w.T)

            with open(path, "r+b") as f:
                f.truncate(os.path.getsize(path) - 40)
            with self.assertRaisesRegex(ValueError, "truncated"):
                self.sess.load_gguf(path)


class GraphTests(SessionTestCase):
    def test_inputs_from_arrays_and_results_as_numpy(self):
        a = np.arange(8, dtype=np.float32).reshape(4, 2)
        b = np.arange(6, dtype=np.float32).reshape(3, 2)
        with self.sess.graph() as g:
            self.assertIsInstance(g, GraphScope)
            ta, tb = g.input(a, name="a"), g.input(b)
            self.assertEqual((ta.name, ta.ne[:2]), ("a", (2, 4)))
            out = g.run(ta.mul_mat(tb))
        np.testing.assert_allclose(out, b @ a.T)

    def test_variables_work_as_either_operand(self):
        w = self.sess.load_arrays({"w": np.arange(8, dtype=np.float32).reshape(4, 2),
                                   "b": np.ones(4, np.float32)})
        x = np.arange(6, dtype=np.float32).reshape(3, 2)
        expected = x @ w["w"].to_numpy().T + 1
        with self.sess.graph() as g:
            tx = g.input(x)
            used = w["w"].ctx.used_mem
            method = w["w"].mul_mat(tx) + w["b"]
            through_graph = g.add(g.mul_mat(w["w"], tx), w["b"])
            reversed_operands = w["w"].mul_mat(tx).add(w["b"])
            scalar = (2.0 * w["b"]).add(-w["b"])
            self.assertIs(method.ctx, g.ctx)
            self.assertEqual(w["w"].ctx.used_mem, used)  # nothing landed in the variables
            r1, r2, r3, r4 = g.run(method, through_graph, reversed_operands, scalar)
        for result in (r1, r2, r3):
            np.testing.assert_allclose(result, expected)
        np.testing.assert_allclose(r4, 1.0)

    def test_placeholders_are_fed(self):
        with self.sess.graph() as g:
            x = g.input(F32, 4, name="x")
            y = x * 2.0
            with self.assertRaisesRegex(ValueError, "input 'x' has no data"):
                g.run(y)
            np.testing.assert_allclose(g.run(y, feed={x: np.ones(4, np.float32)}), 2.0)
            # The data stays for later runs until it is fed again.
            np.testing.assert_allclose(g.run(y), 2.0)
            np.testing.assert_allclose(g.run(y, feed={x: [3, 3, 3, 3]}), 6.0)
            with self.assertRaises(ValueError):
                g.run(y, feed={x: np.ones(5, np.float32)})

    def test_input_converts_with_dtype(self):
        with self.sess.graph() as g:
            ids = g.input([2, 0], dtype=I32)
            self.assertEqual(ids.dtype, I32)
            table = g.input(np.arange(6, dtype=np.float32).reshape(3, 2))
            np.testing.assert_allclose(g.run(table.get_rows(ids)), [[4, 5], [0, 1]])
            with self.assertRaisesRegex(ValueError, "needs the dimensions"):
                g.input(F32)
            with self.assertRaisesRegex(ValueError, "either data or a dtype"):
                g.input(np.ones(2, np.float32), 2)

    def test_feed_rejects_what_is_not_an_input_of_this_graph(self):
        v = self.sess.load_arrays({"v": np.ones(4, np.float32)})["v"]
        with self.sess.graph() as g:
            x = g.input(np.ones(4, np.float32), name="x")
            unused = g.input(F32, 4, name="unused")
            y = x.add(v)
            with self.assertRaisesRegex(ValueError, "is a variable; set it with .write"):
                g.run(y, feed={v: np.ones(4, np.float32)})
            with self.assertRaisesRegex(ValueError, "must be inputs of this graph"):
                g.run(y, feed={y: np.ones(4, np.float32)})
            with self.assertRaisesRegex(ValueError, "'unused' was fed but nothing computed"):
                g.run(y, feed={unused: np.ones(4, np.float32)})
            np.testing.assert_allclose(g.run(y), 2.0)  # an unused input needs no data

    def test_running_again_and_extending_the_graph(self):
        with self.sess.graph() as g:
            x = g.input(np.ones(4, np.float32))
            h = x * 2.0
            o = h + 1.0
            np.testing.assert_allclose(g.run(o), 3.0)
            np.testing.assert_allclose(g.run(o), 3.0)  # inputs are uploaded every time
            # Asking for an intermediate afterwards lays the graph out again.
            rh, ro = g.run(h, o)
            np.testing.assert_allclose(rh, 2.0)
            np.testing.assert_allclose(ro, 3.0)
            more = o * 10.0
            np.testing.assert_allclose(g.run(more, feed={x: np.zeros(4, np.float32)}), 10.0)
            self.assertIsNone(g.run())

    def test_successive_graphs_of_one_shape_with_other_fetches(self):
        # ggml reuses the previous layout for a graph of the same shape, in
        # which an intermediate was overwritten in place.
        def build(g):
            x = g.input(np.ones(64, np.float32))
            h = x * 2.0
            return h, (h + 1.0) * 3.0

        with self.sess.graph() as g:
            h, o = build(g)
            np.testing.assert_allclose(g.run(o), 9.0)
        with self.sess.graph() as g:
            h, o = build(g)
            rh, ro = g.run(h, o)
        np.testing.assert_allclose(rh, 2.0)
        np.testing.assert_allclose(ro, 9.0)

    def test_tensors_end_with_the_block_and_variables_do_not(self):
        v = self.sess.load_arrays({"v": np.ones(4, np.float32)})["v"]
        with self.sess.graph() as g:
            x = g.input(np.ones(4, np.float32))
            y = x.add(v)
            g.run(y)
        with self.assertRaises(GGMLError):
            y.to_numpy()
        with self.assertRaisesRegex(GGMLError, "this graph has ended"):
            g.run(y)
        with self.assertRaisesRegex(GGMLError, "this graph has ended"):
            g.input(np.ones(4, np.float32))
        np.testing.assert_allclose(v.to_numpy(), 1.0)
        with self.sess.graph() as g2:
            with self.assertRaisesRegex(GGMLError, "belongs to a graph that has ended"):
                g2.run(y)

    def test_one_graph_at_a_time_and_a_failed_block_leaves_the_session_usable(self):
        with self.sess.graph() as g:
            with self.assertRaisesRegex(GGMLError, "another graph is still open"):
                with self.sess.graph():
                    pass
            with self.assertRaisesRegex(GGMLError, "already used"):
                with g:
                    pass
        with self.assertRaisesRegex(GGMLError, "enter the graph first"):
            self.sess.graph().input(np.ones(2, np.float32))
        with self.assertRaises(ZeroDivisionError):
            with self.sess.graph() as g:
                g.input(np.ones(4, np.float32)) * 2.0
                1 / 0
        with self.sess.graph() as g:
            np.testing.assert_allclose(g.run(g.input(np.ones(4, np.float32)) * 2.0), 2.0)

    def test_fetching_a_variable_or_a_foreign_tensor_raises(self):
        v = self.sess.load_arrays({"v": np.ones(4, np.float32)})["v"]
        with Context(1 << 20, no_alloc=True) as other, self.sess.graph() as g:
            with self.assertRaisesRegex(ValueError, "'v' is a variable"):
                g.run(v)
            with self.assertRaisesRegex(ValueError, "not built in this graph"):
                g.run(other.arange(0, 4))
            with self.assertRaisesRegex(TypeError, "expected tensors"):
                g.run(3)

    def test_updates_write_into_variables_and_run_before_the_fetches(self):
        cache = self.sess.variable(F32, 8, name="cache")
        for step, value in enumerate((1.0, 2.0)):
            with self.sess.graph() as g:
                x = g.input(np.full(4, value, np.float32))
                g.update(x.cpy(cache.view_1d(4, 4 * 4 * step)))
                # Reads the slot written above: 0 unless the copy ran first.
                seen = cache.view_1d(4 * (step + 1), 0).sum()
                total = g.run(seen)
            np.testing.assert_allclose(total, [4.0, 12.0][step])
        np.testing.assert_allclose(cache.to_numpy(), [1, 1, 1, 1, 2, 2, 2, 2])
        with self.sess.graph() as g:
            g.update(g.input(np.ones(8, np.float32)).cpy(cache))
            self.assertIsNone(g.run())
        np.testing.assert_allclose(cache.to_numpy(), 1.0)

    def test_inplace_operations_accumulate_in_a_variable(self):
        state = self.sess.variable(F32, 4, name="state")
        for _ in range(3):
            with self.sess.graph() as g:
                g.update(state.add_inplace(g.input(np.ones(4, np.float32))))
                g.run()
        np.testing.assert_allclose(state.to_numpy(), 3.0)

    def test_reserve_sizes_the_memory_for_later_graphs(self):
        def build(g, n):
            return (g.input(np.ones(n, np.float32)) * 2.0 + 1.0).sqr()

        with self.sess.graph() as g:
            g.reserve(build(g, 4096))
        sizes = [self.sess.sched.buffer_size(b) for b in self.sess.backends]
        self.assertGreater(sum(sizes), 0)
        for n in (8, 64, 4096):
            with self.sess.graph() as g:
                np.testing.assert_allclose(g.run(build(g, n)), 9.0)
        self.assertEqual([self.sess.sched.buffer_size(b) for b in self.sess.backends], sizes)

    def test_a_graph_that_may_outgrow_graph_size_raises(self):
        with Session(device="cpu", graph_size=8) as sess:
            with sess.graph() as g:
                out = g.input(np.ones(4, np.float32))
                for _ in range(8):
                    out = out * 2.0
                with self.assertRaisesRegex(GGMLError, "larger graph_size"):
                    g.run(out)
            with sess.graph() as g:
                np.testing.assert_allclose(g.run(g.input(np.ones(4, np.float32)) * 2.0), 2.0)

    def test_low_level_objects_are_reachable(self):
        with self.sess.graph() as g:
            y = g.input(np.ones(4, np.float32)) * 2.0
            g.run(y)
            self.assertIs(g.cgraph.ctx, g.ctx)
            self.assertEqual(g.cgraph.n_nodes, 1)
            self.assertIn("n_nodes=1", repr(g))
        with self.assertRaises(AttributeError):
            g.no_such_operation


class ReviewedEdgeTests(SessionTestCase):
    def test_a_fetched_view_keeps_its_data(self):
        # ggml frees a view's source once the view's consumers are done unless
        # the source itself is an output.
        with self.sess.graph() as g:
            x = g.input(np.arange(8, dtype=np.float32))
            view = (x * 2.0).reshape_2d(4, 2)
            tail = view.sqr().cont().cont()
            got_view, got_tail = g.run(view, tail)
        want = (np.arange(8, dtype=np.float32) * 2).reshape(2, 4)
        np.testing.assert_array_equal(got_view, want)
        np.testing.assert_array_equal(got_tail, want ** 2)

    def test_a_permuted_view_cannot_be_fetched(self):
        a = np.arange(6, dtype=np.float32).reshape(2, 3)
        with self.sess.graph() as g:
            x = g.input(a)
            with self.assertRaisesRegex(ValueError, r"cont\(\)"):
                g.run(x.T)
            np.testing.assert_array_equal(g.run(x.T.cont()), a.T)

    def test_numpy_default_types_become_f32_and_i32(self):
        w = self.sess.load_arrays({"w": np.arange(6.0).reshape(3, 2), "i": [1, 2]})
        self.assertEqual(w["w"].dtype, F32)
        self.assertEqual(w["i"].dtype, I32)
        with self.sess.graph() as g:
            x = g.input([[1.0, 2.0]])
            rows = g.input([2, 0])
            self.assertEqual(x.dtype, F32)
            self.assertEqual(rows.dtype, I32)
            np.testing.assert_array_equal(g.run(w["w"].mul_mat(x)), [2.0, 8.0, 14.0])
            np.testing.assert_array_equal(
                g.run(w["w"].get_rows(rows)), [[4.0, 5.0], [0.0, 1.0]]
            )
            with self.assertRaisesRegex(ValueError, "I32"):
                g.input([1 << 40])
            self.assertEqual(g.input([1.0], dtype=F16).dtype, F16)

    def test_a_rejected_feed_keeps_the_previous_data(self):
        with self.sess.graph() as g:
            x = g.input(np.ones(4, dtype=np.float32))
            y = x * 3.0
            with self.assertRaises(ValueError):
                g.run(y, feed={x: np.ones(5, dtype=np.float32)})
            np.testing.assert_array_equal(g.run(y), np.full(4, 3.0, dtype=np.float32))

    def test_bad_variable_specs_raise(self):
        with self.assertRaisesRegex(ValueError, "not a tensor type"):
            self.sess.variables({"a": (999, 3)})
        with self.assertRaisesRegex(ValueError, "at least 1"):
            self.sess.variables({"a": (F32, 0)})
        with self.assertRaisesRegex(ValueError, "at least 1"):
            self.sess.load_arrays({"a": np.zeros((0, 2), dtype=np.float32)})
        with self.assertRaisesRegex(TypeError, "integers"):
            self.sess.variables({"a": (F32, 2.5)})
        # The session is still usable.
        self.assertEqual(self.sess.variable(F32, 2).ne[0], 2)

    def test_graph_size_counts_tensors_of_other_contexts(self):
        # Leafs from a context the session does not own count against the
        # capacity too; ggml would abort when they overflow it.
        with Session(device="cpu", graph_size=8) as sess, \
                Context.for_tensors(16) as ctx:
            leafs = [ctx.new_tensor_1d(F32, 2) for _ in range(12)]
            sess.backends[-1].alloc_ctx_tensors(ctx).clear(0)
            with sess.graph() as g:
                total = g.add(leafs[0], leafs[1])
                with self.assertRaisesRegex(GGMLError, "larger graph_size"):
                    g.run(total)

    def test_unused_variables_do_not_count_against_graph_size(self):
        with Session(device="cpu", graph_size=16) as sess:
            sess.variables({f"v{i}": (F32, 2) for i in range(20)})
            w = sess.load_arrays({"w": np.ones(2, dtype=np.float32)})
            with sess.graph() as g:
                np.testing.assert_array_equal(g.run(w["w"] * 2.0), [2.0, 2.0])


if __name__ == "__main__":
    unittest.main()

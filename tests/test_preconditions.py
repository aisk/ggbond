import unittest

from ggbond.ggml import (
    Context,
    F16,
    F32,
    GGMLError,
    GGUF,
    I32,
    graph_overhead_custom,
    tensor_overhead,
)


class OpShapeTests(unittest.TestCase):
    def setUp(self):
        self.ctx = Context(1 << 20, no_alloc=True)

    def tearDown(self):
        self.ctx.close()

    def t(self, *ne, dtype=F32):
        return self.ctx.new_tensor(dtype, *ne)

    def test_elementwise_ops_require_broadcastable_operands(self):
        a, b = self.t(4, 3), self.t(3, 3)
        for op in ("add", "sub", "mul", "div"):
            with self.assertRaisesRegex(ValueError, f"{op}: cannot broadcast"):
                getattr(a, op)(b)
        a.add(self.t(4, 1))  # broadcasting a row is fine

    def test_repeat_requires_a_multiple(self):
        with self.assertRaisesRegex(ValueError, "repeat: cannot tile"):
            self.t(3).repeat(self.t(4))
        self.t(2).repeat(self.t(4, 2))

    def test_mul_mat_requires_a_shared_inner_dim(self):
        with self.assertRaisesRegex(ValueError, r"mul_mat: .*a\.ne=\(3, 4, 1, 1\)"):
            self.t(3, 4).mul_mat(self.t(5, 2))
        with self.assertRaisesRegex(ValueError, "mul_mat"):
            self.t(3, 4, 2).mul_mat(self.t(3, 2, 3))
        self.assertEqual(self.t(3, 4).mul_mat(self.t(3, 2)).ne, (4, 2))

    def test_reshape_checks_count_and_contiguity(self):
        with self.assertRaisesRegex(ValueError, "cannot reshape 12 elements"):
            self.t(4, 3).reshape_2d(5, 2)
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            self.t(4, 3).transpose().reshape_1d(12)
        self.t(4, 3).reshape_1d(12)

    def test_cpy_requires_matching_element_counts(self):
        with self.assertRaisesRegex(ValueError, "cpy: element counts differ"):
            self.t(4).cpy(self.t(5))

    def test_get_rows_checks_index_dtype_and_shape(self):
        with self.assertRaisesRegex(ValueError, "must be I32"):
            self.t(4, 3).get_rows(self.t(2))
        with self.assertRaisesRegex(ValueError, "get_rows: needs"):
            self.t(4, 3, 2).get_rows(self.t(2, 3, dtype=I32))
        self.t(4, 3).get_rows(self.t(2, dtype=I32))

    def test_concat_checks_dim_dtype_and_shape(self):
        with self.assertRaisesRegex(ValueError, "dim must be"):
            self.t(4, 2).concat(self.t(4, 2), 4)
        with self.assertRaisesRegex(ValueError, "dtypes differ"):
            self.t(4, 2).concat(self.t(4, 2, dtype=F16), 1)
        with self.assertRaisesRegex(ValueError, "must match outside dim 1"):
            self.t(4, 2).concat(self.t(3, 2), 1)
        self.assertEqual(self.t(4, 2).concat(self.t(4, 3), 1).ne, (4, 5))


class ArenaTests(unittest.TestCase):
    def test_op_raises_when_the_context_is_full(self):
        with Context(tensor_overhead() * 2, no_alloc=True) as ctx:
            t = ctx.new_tensor_1d(F32, 4)
            t = t.relu()
            with self.assertRaisesRegex(GGMLError, "relu: context out of memory"):
                t.relu()

    def test_new_tensor_counts_data_in_allocating_contexts(self):
        with Context(tensor_overhead() + 64, no_alloc=False) as ctx:
            ctx.new_tensor_1d(F32, 16)  # exactly 64 bytes of data
        with Context(tensor_overhead() + 64, no_alloc=False) as ctx:
            with self.assertRaisesRegex(GGMLError, "new_tensor: context out of memory"):
                ctx.new_tensor_1d(F32, 17)

    def test_new_graph_raises_when_it_does_not_fit(self):
        with Context(graph_overhead_custom(8, False), no_alloc=True) as ctx:
            ctx.new_graph(size=8)
        # ggml_init rounds mem_size up to GGML_MEM_ALIGN, so fall short by a
        # whole alignment unit.
        with Context(graph_overhead_custom(8, False) - 16, no_alloc=True) as ctx:
            with self.assertRaisesRegex(GGMLError, "new_graph: context out of memory"):
                ctx.new_graph(size=8)

    def test_convolution_tensor_counts_match_ggml(self):
        # Every op reserves room for the tensors it creates; the convolutions
        # create several, so check the table against what ggml actually uses.
        cases = {
            "conv_1d": (lambda ctx: (ctx.new_tensor(F16, 3, 4, 8), ctx.new_tensor(F32, 32, 4)),
                        lambda k, x: k.conv_1d(x, 1, 1, 1), 5),
            "conv_2d": (lambda ctx: (ctx.new_tensor(F16, 3, 3, 4, 8), ctx.new_tensor(F32, 16, 16, 4)),
                        lambda k, x: k.conv_2d(x, 1, 1, 1, 1, 1, 1), 7),
        }
        for name, (inputs, op, n) in cases.items():
            with self.subTest(name), Context(1 << 20, no_alloc=True) as ctx:
                kernel, data = inputs(ctx)
                before = ctx.used_mem
                op(kernel, data)
                self.assertEqual(ctx.used_mem - before, n * tensor_overhead())


class GGUFGuardTests(unittest.TestCase):
    def setUp(self):
        self.gguf = GGUF.init_empty()
        self.gguf.set_val_u32("n", 3)
        self.gguf.set_arr_str("names", ["a", "b"])

    def tearDown(self):
        self.gguf.close()

    def test_typed_getters_check_index_and_type(self):
        with self.assertRaisesRegex(IndexError, "key index 5 out of range"):
            self.gguf.get_key(5)
        with self.assertRaisesRegex(GGMLError, "holds u32, not str"):
            self.gguf.get_val_str(0)
        with self.assertRaisesRegex(GGMLError, "get_val_u32"):
            self.gguf.get_val_u32(1)
        self.assertEqual(self.gguf.get_val_u32(0), 3)

    def test_array_getters_check_kind_and_bounds(self):
        with self.assertRaisesRegex(GGMLError, "get_arr_n"):
            self.gguf.get_arr_n(0)
        with self.assertRaisesRegex(IndexError, "element 2 out of range"):
            self.gguf.get_arr_str(1, 2)
        self.assertEqual(self.gguf.get_arr_str(1, 1), "b")

    def test_tensor_entries_are_checked_by_index_and_name(self):
        with self.assertRaisesRegex(IndexError, "tensor index 0 out of range"):
            self.gguf.get_tensor_name(0)
        with self.assertRaises(KeyError):
            self.gguf.set_tensor_type("missing", F32)
        with self.assertRaises(KeyError):
            self.gguf.set_tensor_data("missing", b"\0" * 4)

        with Context(1 << 20, no_alloc=False) as ctx:
            tensor = ctx.new_tensor_1d(F32, 4, name="w")
            self.gguf.add_tensor(tensor)
            with self.assertRaisesRegex(ValueError, "duplicate tensor name 'w'"):
                self.gguf.add_tensor(tensor)


if __name__ == "__main__":
    unittest.main()

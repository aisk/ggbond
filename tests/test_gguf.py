import os
import tempfile
import unittest

import numpy as np

from ggbond.ggml import Context, F32, GGMLError, GGUF
from ggbond.ggml.gguf import TYPE_INT32, type_name


class GGUFRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = os.path.join(self.directory.name, "model.gguf")

    def write(self):
        weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        with Context(1 << 20, no_alloc=False) as ctx:
            tensor = ctx.new_tensor_1d(F32, 4, name="w")
            with GGUF.init_empty() as writer:
                writer.set_val_str("general.architecture", "demo")
                writer.set_val_u32("demo.block_count", 7)
                writer.set_val_bool("demo.flag", True)
                writer.set_arr_data("demo.dims", TYPE_INT32, [3, 5, 7])
                writer.set_arr_str("demo.names", ["a", "bb"])
                writer.add_tensor(tensor)
                writer.set_tensor_data("w", weights.tobytes())
                writer.write_to_file(self.path)
        return weights

    def test_metadata_survives_a_round_trip(self):
        self.write()

        with GGUF.from_file(self.path) as gguf:
            self.addCleanup(gguf.context.close)

            self.assertEqual(gguf.value("general.architecture"), "demo")
            self.assertEqual(gguf.value("demo.block_count"), 7)
            self.assertIs(gguf.value("demo.flag"), True)
            np.testing.assert_array_equal(gguf.value("demo.dims"), [3, 5, 7])
            self.assertEqual(gguf.value("demo.names"), ["a", "bb"])

    def test_items_covers_every_key(self):
        self.write()

        with GGUF.from_file(self.path) as gguf:
            self.addCleanup(gguf.context.close)

            self.assertEqual(len(list(gguf.items())), gguf.n_kv)
            self.assertEqual(list(gguf.keys())[0], "general.architecture")

    def test_tensor_entry_is_written(self):
        weights = self.write()

        with GGUF.from_file(self.path) as gguf:
            self.addCleanup(gguf.context.close)

            self.assertEqual(gguf.n_tensors, 1)
            self.assertEqual(gguf.get_tensor_name(0), "w")
            self.assertEqual(gguf.get_tensor_size(0), weights.nbytes)

    def test_missing_key_raises_key_error(self):
        self.write()

        with GGUF.from_file(self.path) as gguf:
            self.addCleanup(gguf.context.close)

            with self.assertRaises(KeyError):
                gguf.value("demo.absent")

    def test_get_arr_data_refuses_string_arrays(self):
        self.write()

        with GGUF.from_file(self.path) as gguf:
            self.addCleanup(gguf.context.close)

            with self.assertRaisesRegex(GGMLError, "use get_arr_str"):
                gguf.get_arr_data(gguf.key_index("demo.names"))


class GGUFTypeNameTests(unittest.TestCase):
    def test_type_name_decodes(self):
        self.assertEqual(type_name(TYPE_INT32), "i32")


if __name__ == "__main__":
    unittest.main()

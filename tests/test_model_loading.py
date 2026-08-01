import io
import struct
import tempfile
import unittest

import ggml as _ggml

from examples import gpt2, magika
from ggbond.ggml import Backend


def _minimal_gpt2_file(*, trained_context=4, trailing=b"") -> bytes:
    header = struct.pack(
        "<I6i",
        _ggml.GGML_FILE_MAGIC,
        1,  # n_vocab
        trained_context,
        1,  # n_embd
        1,  # n_head
        1,  # n_layer
        0,  # ftype = F32
    )
    vocab = struct.pack("<iI", 1, 1) + b"a"
    return header + vocab + trailing


class ExactReadTests(unittest.TestCase):
    def test_magika_read_exact_rejects_short_data(self):
        with self.assertRaisesRegex(ValueError, "expected 4 bytes, got 2"):
            magika._read_exact(io.BytesIO(b"ab"), 4, "test data")


class GPT2ModelValidationTests(unittest.TestCase):
    def _write_model(self, data: bytes):
        tmp = tempfile.NamedTemporaryFile()
        tmp.write(data)
        tmp.flush()
        return tmp

    def test_rejects_context_larger_than_training_context(self):
        with self._write_model(_minimal_gpt2_file(trained_context=4)) as model_file:
            with self.assertRaisesRegex(ValueError, "trained size 4"):
                gpt2.gpt2_model_load(model_file.name, None, n_ctx=5)

    def test_rejects_truncated_tensor_header(self):
        with self._write_model(_minimal_gpt2_file(trailing=b"x")) as model_file:
            with Backend.cpu_init() as backend:
                with self.assertRaisesRegex(ValueError, "tensor header"):
                    gpt2.gpt2_model_load(model_file.name, backend, n_ctx=4)

    def test_rejects_missing_required_tensors(self):
        with self._write_model(_minimal_gpt2_file()) as model_file:
            with Backend.cpu_init() as backend:
                with self.assertRaisesRegex(ValueError, "missing .* required tensors"):
                    gpt2.gpt2_model_load(model_file.name, backend, n_ctx=4)


if __name__ == "__main__":
    unittest.main()

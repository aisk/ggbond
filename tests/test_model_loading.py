import io
import os
import tempfile
import unittest

from examples import gpt2 as gpt2_session
from examples.ggml import gpt2, magika
from ggbond import Session
from ggbond.ggml import Backend
from tests._gpt2_model import N_CTX, N_VOCAB, write_tiny_gpt2


class ExactReadTests(unittest.TestCase):
    def test_magika_read_exact_rejects_short_data(self):
        with self.assertRaisesRegex(ValueError, "expected 4 bytes, got 2"):
            magika._read_exact(io.BytesIO(b"ab"), 4, "test data")


class GPT2ModelValidationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "tiny.gguf")
        self.backend = Backend.cpu_init()
        self.addCleanup(self.backend.close)
        self.sess = Session(self.backend)
        self.addCleanup(self.sess.close)

    def _loaders(self):
        yield "ggml", lambda n_ctx: gpt2.gpt2_model_load(self.path, self.backend, n_ctx)
        yield "session", lambda n_ctx: gpt2_session.gpt2_model_load(self.path, self.sess, n_ctx)

    def test_rejects_context_larger_than_training_context(self):
        write_tiny_gpt2(self.path, trained_context=4)
        for name, load in self._loaders():
            with self.subTest(name), self.assertRaisesRegex(ValueError, "trained size 4"):
                load(5)

    def test_rejects_another_architecture(self):
        write_tiny_gpt2(self.path, architecture="llama")
        for name, load in self._loaders():
            with self.subTest(name), self.assertRaisesRegex(ValueError, "not a GPT-2 model.*'llama'"):
                load(N_CTX)

    def test_rejects_truncated_tensor_data(self):
        write_tiny_gpt2(self.path)
        with open(self.path, "r+b") as f:
            f.truncate(os.path.getsize(self.path) - 40)
        for name, load in self._loaders():
            with self.subTest(name), self.assertRaisesRegex(ValueError, "truncated"):
                load(N_CTX)

    def test_reads_hyperparameters_and_a_byte_level_vocab(self):
        write_tiny_gpt2(self.path)
        for name, load in self._loaders():
            with self.subTest(name):
                model, token_to_id, id_to_token = load(N_CTX)
                if name == "ggml":
                    self.addCleanup(model.close)
                self.assertEqual(
                    (model.n_vocab, model.n_ctx, model.n_embd, model.n_head, model.n_layer),
                    (N_VOCAB, N_CTX, 4, 2, 1),
                )
                self.assertEqual(model.eos_token, N_VOCAB - 1)
                self.assertEqual(
                    id_to_token, {0: b"ab", 1: b" cd", 2: "é".encode(), 3: b"<|endoftext|>"}
                )
                self.assertEqual(token_to_id[b" cd"], 1)
                self.assertEqual(gpt2.gpt_tokenize(token_to_id, "ab cdé"), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()

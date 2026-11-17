"""End-to-end forward pass of the gpt2 example over a synthetic tiny model.

Covers the graph-building path the validation tests skip, including the single
token case whose QKV tensor has a trailing 1-sized dimension.
"""

import os
import tempfile
import unittest

import numpy as np

from examples import gpt2 as gpt2_session
from examples.ggml import gpt2
from ggbond import Session
from ggbond.ggml import Backend, GAllocr
from tests._gpt2_model import N_CTX, N_VOCAB, write_tiny_gpt2


class GPT2ForwardTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.model_path = os.path.join(tmp.name, "tiny.gguf")
        write_tiny_gpt2(self.model_path)

        self.backend = Backend.cpu_init()
        self.addCleanup(self.backend.close)
        self.model, _, _ = gpt2.gpt2_model_load(
            self.model_path, self.backend, n_ctx=N_CTX
        )
        self.addCleanup(self.model.close)
        self.allocator = GAllocr.from_backend(self.backend)
        self.addCleanup(self.allocator.close)

    def test_prompt_then_single_token_produce_finite_logits(self):
        prompt_logits = gpt2.gpt2_eval(
            self.model, self.backend, self.allocator, 0, [0, 1, 2]
        )
        self.assertEqual(prompt_logits.shape, (N_VOCAB,))
        self.assertTrue(np.all(np.isfinite(prompt_logits)))

        # N == 1: the QKV tensor's second dimension is 1, so its row stride is
        # only visible through .nb.
        next_logits = gpt2.gpt2_eval(
            self.model, self.backend, self.allocator, 3, [3]
        )
        self.assertEqual(next_logits.shape, (N_VOCAB,))
        self.assertTrue(np.all(np.isfinite(next_logits)))

    def test_a_batched_prompt_matches_token_by_token_decoding(self):
        batched = gpt2.gpt2_eval(self.model, self.backend, self.allocator, 0, [0, 1, 2])

        # Re-run the same tokens one at a time through a fresh KV cache.
        model, _, _ = gpt2.gpt2_model_load(
            self.model_path, self.backend, n_ctx=N_CTX
        )
        self.addCleanup(model.close)
        stepwise = None
        for position, token in enumerate([0, 1, 2]):
            stepwise = gpt2.gpt2_eval(
                model, self.backend, self.allocator, position, [token]
            )

        np.testing.assert_allclose(stepwise, batched, atol=1e-4)

    def test_the_session_example_matches_the_ggml_example(self):
        tokens = [0, 1, 2, 3, 1, 0, 2]
        expected = [gpt2.gpt2_eval(self.model, self.backend, self.allocator, 0, tokens[:3])]
        for position, token in enumerate(tokens[3:], start=3):
            expected.append(
                gpt2.gpt2_eval(self.model, self.backend, self.allocator, position, [token])
            )

        with Session(device="cpu", graph_size=gpt2_session.GPT2_MAX_NODES) as sess:
            model, _, _ = gpt2_session.gpt2_model_load(self.model_path, sess, N_CTX)
            # The worst-case reservation main() makes, then smaller graphs.
            with sess.graph() as g:
                *_, logits = gpt2_session.gpt2_graph(model, g, N_CTX - 4, 4)
                g.reserve(logits)
            sizes = [sess.sched.buffer_size(b) for b in sess.backends]

            got = [gpt2_session.gpt2_eval(model, sess, 0, tokens[:3])]
            for position, token in enumerate(tokens[3:], start=3):
                got.append(gpt2_session.gpt2_eval(model, sess, position, [token]))

            self.assertEqual([sess.sched.buffer_size(b) for b in sess.backends], sizes)
            # The key/value cache the graphs wrote is the same as well.
            np.testing.assert_array_equal(
                model.memory_k.to_numpy(), self.model.memory_k.to_numpy()
            )
        for step, (a, b) in enumerate(zip(expected, got)):
            np.testing.assert_array_equal(a, b, err_msg=f"step {step}")


if __name__ == "__main__":
    unittest.main()

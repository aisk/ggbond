"""End-to-end forward pass of the gpt2 example over a synthetic tiny model.

Covers the graph-building path the validation tests skip, including the single
token case whose QKV tensor has a trailing 1-sized dimension.
"""

import struct
import tempfile
import unittest

import numpy as np

import ggml as _ggml

from examples import gpt2 as gpt2_session
from examples.ggml import gpt2
from ggbond import Session
from ggbond.ggml import Backend, GAllocr

N_VOCAB = 4
N_CTX = 8
N_EMBD = 4
N_HEAD = 2
N_LAYER = 1


def _tensor_entry(name: str, ne: "list[int]", values: np.ndarray) -> bytes:
    encoded = name.encode()
    header = struct.pack("<3i", len(ne), len(encoded), 0)  # 0 = F32
    dims = b"".join(struct.pack("<i", n) for n in ne)
    return header + dims + encoded + values.astype(np.float32).tobytes()


def _tiny_gpt2_file() -> bytes:
    rng = np.random.default_rng(0)

    def weights(*ne):
        # ne is ggml order; the file stores elements in that same layout.
        return rng.standard_normal(int(np.prod(ne))).astype(np.float32) * 0.1

    parts = [
        struct.pack("<I6i", _ggml.GGML_FILE_MAGIC,
                    N_VOCAB, N_CTX, N_EMBD, N_HEAD, N_LAYER, 0),
        struct.pack("<i", N_VOCAB),
    ]
    for i in range(N_VOCAB):
        token = f"t{i}".encode()
        parts.append(struct.pack("<I", len(token)) + token)

    entries = [
        ("model/ln_f/g", [N_EMBD]),
        ("model/ln_f/b", [N_EMBD]),
        ("model/wte", [N_EMBD, N_VOCAB]),
        ("model/wpe", [N_EMBD, N_CTX]),
        ("model/h0/ln_1/g", [N_EMBD]),
        ("model/h0/ln_1/b", [N_EMBD]),
        ("model/h0/ln_2/g", [N_EMBD]),
        ("model/h0/ln_2/b", [N_EMBD]),
        ("model/h0/attn/c_attn/w", [N_EMBD, 3 * N_EMBD]),
        ("model/h0/attn/c_attn/b", [3 * N_EMBD]),
        ("model/h0/attn/c_proj/w", [N_EMBD, N_EMBD]),
        ("model/h0/attn/c_proj/b", [N_EMBD]),
        ("model/h0/mlp/c_fc/w", [N_EMBD, 4 * N_EMBD]),
        ("model/h0/mlp/c_fc/b", [4 * N_EMBD]),
        ("model/h0/mlp/c_proj/w", [4 * N_EMBD, N_EMBD]),
        ("model/h0/mlp/c_proj/b", [N_EMBD]),
    ]
    for name, ne in entries:
        parts.append(_tensor_entry(name, ne, weights(*ne)))

    return b"".join(parts)


class GPT2ForwardTests(unittest.TestCase):
    def setUp(self):
        self.model_file = tempfile.NamedTemporaryFile(suffix=".bin")
        self.addCleanup(self.model_file.close)
        self.model_file.write(_tiny_gpt2_file())
        self.model_file.flush()

        self.backend = Backend.cpu_init()
        self.addCleanup(self.backend.close)
        self.model, _, _ = gpt2.gpt2_model_load(
            self.model_file.name, self.backend, n_ctx=N_CTX
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
            self.model_file.name, self.backend, n_ctx=N_CTX
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
            model, _, _ = gpt2_session.gpt2_model_load(self.model_file.name, sess, N_CTX)
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

"""GPT-2 inference on the high-level ``ggbond`` API.

A port of ggml's ``gpt-2`` ``main-backend`` example that reads a GGUF model.
:meth:`Session.load_gguf` turns the file into a dict of weights plus its
metadata, which carries the hyperparameters and the vocabulary.

A :class:`Session` picks the device. The weights and the key/value cache are
*variables*: created once, kept for the whole run. Each evaluation opens a
graph (``with sess.graph() as g:``), because its shapes depend on how many
tokens are fed and how many are already cached, builds the transformer in it,
and runs it for the logits. The same program on the 1:1 ggml classes is
``examples/ggml/gpt2.py``.

Usage::

    python examples/gpt2.py -m models/gpt2.Q8_0.gguf -p "Hello"
    python examples/gpt2.py -m models/gpt2.Q8_0.gguf -b gpu

Get a model from Hugging Face, where llama.cpp's conversions of GPT-2 are
published (any quantization works; the tensor and metadata names are the ones
llama.cpp's ``convert_hf_to_gguf.py`` writes)::

    curl -L -o models/gpt2.Q8_0.gguf \\
        https://huggingface.co/mradermacher/gpt2-GGUF/resolve/main/gpt2.Q8_0.gguf
"""

import argparse
import codecs
import math
import random
import re
import time

import numpy as np

from ggbond import F32, I32, GraphScope, Session

GPT2_MAX_NODES = 4096


# ============================================================================
# Tokenizer + sampling (ported from common.cpp)
# ============================================================================

def gpt_tokenize(token_to_id: dict[bytes, int], text: str) -> list[int]:
    """Greedy longest-match BPE tokenizer matching the C++ ``gpt_tokenize``.

    Splits the text into GPT-2-style pieces (each keeps its leading space), then
    greedily matches the longest known prefix against the byte-keyed vocab.
    """
    pat = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""",
        re.UNICODE,
    )
    tokens: list[int] = []
    for word in pat.findall(text):
        b = word.encode("utf-8")
        i, n = 0, len(b)
        while i < n:
            matched = False
            for j in range(n, i, -1):  # longest prefix first
                sub = b[i:j]
                if sub in token_to_id:
                    tokens.append(token_to_id[sub])
                    i = j
                    matched = True
                    break
            if not matched:  # unknown byte -- skip it
                i += 1
    return tokens


def gpt_sample_top_k_top_p(
    logits: np.ndarray, top_k: int, top_p: float, temp: float, rng: random.Random
) -> int:
    """Top-k / top-p sampling from a logits vector."""
    n_vocab = len(logits)
    if temp <= 0:
        return int(np.argmax(logits))

    scaled = logits / temp
    if top_k > 0:
        top_k = min(top_k, n_vocab)
        indices = np.argpartition(scaled, -top_k)[-top_k:]
    else:
        indices = np.arange(n_vocab)

    top_logits = scaled[indices]
    exps = np.exp(top_logits - np.max(top_logits))
    probs = exps / np.sum(exps)

    order = np.argsort(-probs)
    sorted_probs = probs[order]
    sorted_indices = indices[order]

    if top_p < 1.0:
        cumsum = np.cumsum(sorted_probs)
        cutoff = int(np.searchsorted(cumsum, top_p)) + 1
        sorted_probs = sorted_probs[:cutoff]
        sorted_indices = sorted_indices[:cutoff]
        sorted_probs = sorted_probs / np.sum(sorted_probs)

    r = rng.random()
    acc = 0.0
    for p, idx in zip(sorted_probs, sorted_indices):
        acc += p
        if r < acc:
            return int(idx)
    return int(sorted_indices[-1])


# ============================================================================
# Model
# ============================================================================

def _gpt2_byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's byte-level alphabet.

    The tokenizer stores every byte of a token as one printable character (a
    space is ``Ġ``), and that is how a GGUF lists the vocabulary.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    )
    decoder = {chr(b): b for b in printable}
    shifted = (b for b in range(256) if b not in printable)
    decoder.update({chr(256 + n): b for n, b in enumerate(shifted)})
    return decoder


def gpt2_vocab(gguf) -> "tuple[dict[bytes, int], dict[int, bytes]]":
    """The GGUF's vocabulary as raw bytes: (token_to_id, id_to_token).

    Many tokens are not valid UTF-8 on their own, so the vocab is keyed by
    bytes and matched on bytes.
    """
    decoder = _gpt2_byte_decoder()
    id_to_token = {}
    for i, token in enumerate(gguf["tokenizer.ggml.tokens"]):
        try:
            id_to_token[i] = bytes(decoder[ch] for ch in token)
        except KeyError:  # not byte-level encoded: keep the text as it is
            id_to_token[i] = token.encode("utf-8")
    return {token: i for i, token in id_to_token.items()}, id_to_token


def _gpt2_hparams(gguf, n_ctx: int) -> dict:
    """The numbers the graph needs, read from the GGUF metadata and checked."""
    arch = gguf.get("general.architecture")
    if arch != "gpt2":
        raise ValueError(f"not a GPT-2 model: general.architecture is {arch!r}")
    n_ctx_train = gguf["gpt2.context_length"]
    n_embd = gguf["gpt2.embedding_length"]
    n_head = gguf["gpt2.attention.head_count"]
    if n_embd % n_head != 0:
        raise ValueError(f"model embedding size {n_embd} is not divisible by {n_head} heads")
    if not (0 < n_ctx <= n_ctx_train):
        raise ValueError(
            f"requested context size must be between 1 and the trained size {n_ctx_train}, got {n_ctx}"
        )
    return dict(
        n_ctx=n_ctx,  # the *user* context length; the file has the trained one
        n_embd=n_embd,
        n_head=n_head,
        n_layer=gguf["gpt2.block_count"],
        eps=gguf.get("gpt2.attention.layer_norm_epsilon", 1e-5),
        eos_token=gguf.get("tokenizer.ggml.eos_token_id", 50256),
    )


class GPT2Model:
    """The model's variables (weights, key/value cache) and hyperparameters."""

    def __init__(self, weights, memory_k, memory_v, *, n_ctx, n_embd, n_head, n_layer, eps, eos_token):
        self.weights = weights  # name -> Variable, named as in the GGUF
        self.memory_k = memory_k
        self.memory_v = memory_v
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.eps = eps
        self.eos_token = eos_token
        self.n_vocab = weights["token_embd.weight"].ne[1]
        # llama.cpp's converter writes the tied LM head out as its own tensor;
        # fall back on the token embedding for a file that does not.
        self.lm_head = weights.get("output.weight", weights["token_embd.weight"])


def gpt2_model_load(fname: str, sess: Session, n_ctx: int):
    """Load a GPT-2 GGUF. Returns (model, token_to_id, id_to_token)."""
    print(f"gpt2_model_load: loading model from '{fname}'")

    # every tensor of the file becomes a variable on the session's device
    weights, gguf = sess.load_gguf(fname)
    hparams = _gpt2_hparams(gguf, n_ctx)
    for key, value in hparams.items():
        print(f"gpt2_model_load: {key:9} = {value}")
    size_mb = sum(t.nbytes for t in weights.values()) / (1024.0 * 1024.0)
    print(f"gpt2_model_load: weights size = {size_mb:6.2f} MB")

    # key + value memory: variables the graphs write
    n_elements = hparams["n_embd"] * hparams["n_layer"] * n_ctx
    memory = sess.variables({"memory_k": (F32, n_elements), "memory_v": (F32, n_elements)})
    kv_mb = sum(t.nbytes for t in memory.values()) / (1024.0 * 1024.0)
    print(f"gpt2_model_load: memory size  = {kv_mb:6.2f} MB")

    model = GPT2Model(weights, memory["memory_k"], memory["memory_v"], **hparams)
    return (model, *gpt2_vocab(gguf))


# ============================================================================
# Graph
# ============================================================================

def gpt2_graph(model: GPT2Model, g: GraphScope, n_past: int, n_tokens: int):
    """Build the forward pass in the graph ``g``. Returns (embd, position, logits)."""
    N = n_tokens
    w = model.weights
    n_embd, n_layer, n_ctx, n_head = model.n_embd, model.n_layer, model.n_ctx, model.n_head

    # Weights and the KV cache are variables; an expression on one
    # (``w.mul_mat(x)``, ``memory_k.view_1d(...)``) becomes part of this graph.
    memory_k = model.memory_k
    memory_v = model.memory_v
    k_esz = memory_k.element_size  # bytes per element (F32 -> 4)

    embd = g.input(I32, N, name="embd")
    position = g.input(I32, N, name="position")

    def norm(x, name):  # layer norm with the gain and bias called ``name``
        return x.norm(model.eps).mul(w[f"{name}.weight"]).add(w[f"{name}.bias"])

    def linear(x, name):
        return w[f"{name}.weight"].mul_mat(x).add(w[f"{name}.bias"])

    # token + position embedding
    inpL = w["token_embd.weight"].get_rows(embd) + w["position_embd.weight"].get_rows(position)

    for il in range(n_layer):
        blk = f"blk.{il}"

        # ln_1
        cur = norm(inpL, f"{blk}.attn_norm")

        # attention: fused QKV projection -> [3*n_embd, N]
        cur = linear(cur, f"{blk}.attn_qkv")

        # row stride of the contiguous [3*n_embd, N] QKV tensor
        nb1 = cur.nb[1]
        Qcur = cur.view_2d(n_embd, N, nb1, 0 * 4 * n_embd)
        Kcur = cur.view_2d(n_embd, N, nb1, 1 * 4 * n_embd)
        Vcur = cur.view_2d(n_embd, N, nb1, 2 * 4 * n_embd)

        # store the new keys/values into the KV cache
        if N >= 1:
            # g.update registers the copies now, so they run before the reads below
            k = memory_k.view_1d(N * n_embd, k_esz * n_embd * (il * n_ctx + n_past))
            v = memory_v.view_1d(N * n_embd, k_esz * n_embd * (il * n_ctx + n_past))
            g.update(Kcur.cpy(k))
            g.update(Vcur.cpy(v))

        # Q = [64, N, 12]
        Q = Qcur.cont_3d(n_embd // n_head, n_head, N).permute(0, 2, 1, 3)

        # K = [64, n_past + N, 12]
        K = (
            memory_k.view_1d((n_past + N) * n_embd, il * n_ctx * k_esz * n_embd)
            .reshape_3d(n_embd // n_head, n_head, n_past + N)
            .permute(0, 2, 1, 3)
        )

        # KQ = K * Q, scaled, causally masked, softmaxed
        KQ = K.mul_mat(Q)
        KQ_scaled = KQ.scale(1.0 / math.sqrt(n_embd / n_head))
        KQ_masked = KQ_scaled.diag_mask_inf(n_past)
        KQ_soft_max = KQ_masked.soft_max()

        # V^T = [n_past + N, 64, 12]
        V_trans = (
            memory_v.view_1d((n_past + N) * n_embd, il * n_ctx * k_esz * n_embd)
            .reshape_3d(n_embd // n_head, n_head, n_past + N)
            .permute(1, 2, 0, 3)
            .cont_3d(n_past + N, n_embd // n_head, n_head)
        )

        KQV = V_trans.mul_mat(KQ_soft_max)
        cur = KQV.permute(0, 2, 1, 3).cont_2d(n_embd, N)

        # attention output projection
        cur = linear(cur, f"{blk}.attn_output")

        # residual
        cur = cur.add(inpL)
        inpFF = cur

        # ln_2 + MLP
        cur = norm(inpFF, f"{blk}.ffn_norm")
        cur = linear(cur, f"{blk}.ffn_up").gelu()
        cur = linear(cur, f"{blk}.ffn_down")

        # residual
        inpL = cur.add(inpFF)

    # final norm
    inpL = norm(inpL, "output_norm")

    # logits = lm_head * inpL
    logits = model.lm_head.mul_mat(inpL)
    logits.name = "logits"
    return embd, position, logits


# ============================================================================
# Evaluation
# ============================================================================

def gpt2_eval(model: GPT2Model, sess: Session, n_past: int, embd_inp) -> np.ndarray:
    """Run the transformer and return logits for the last token."""
    N = len(embd_inp)
    n_vocab = model.n_vocab
    # A graph per evaluation: its shapes depend on N and n_past. Everything
    # built in the block ends with it; the weights and the KV cache stay.
    with sess.graph() as g:
        embd, position, logits = gpt2_graph(model, g, n_past, N)
        out = g.run(logits, feed={
            embd: np.asarray(embd_inp, dtype=np.int32),
            position: np.arange(n_past, n_past + N, dtype=np.int32),
        })

    # logits hold [n_vocab, N] row-major (n_vocab fastest); return just the
    # last token's row, mirroring the C++ which reads that slice directly.
    return out.reshape(N, n_vocab)[N - 1].copy()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GPT-2 inference on the ggbond.ggml API")
    parser.add_argument("-m", "--model", default="models/gpt2.Q8_0.gguf",
                        help="Path to a GPT-2 model in GGUF format")
    parser.add_argument("-p", "--prompt", default="Hello, my name is", help="Input prompt")
    parser.add_argument("-n", "--n-predict", type=int, default=200, help="Tokens to predict")
    parser.add_argument("--n-ctx", type=int, default=1024, help="Context size")
    parser.add_argument("--n-batch", type=int, default=32, help="Prompt batch size")
    parser.add_argument("-t", "--threads", type=int, default=None,
                        help="CPU threads (default: physical cores)")
    parser.add_argument("-b", "--backend", default=None,
                        help="Device name or type, e.g. cpu, gpu or one of: "
                             + ", ".join(Session.devices()) + " (default: best available)")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 = random)")
    args = parser.parse_args()

    t_main_start = time.perf_counter()

    seed = args.seed if args.seed >= 0 else random.randint(0, 2**31 - 1)
    print(f"main: seed = {seed}")
    rng = random.Random(seed)

    with Session(device=args.backend, n_threads=args.threads, graph_size=GPT2_MAX_NODES) as sess:
        t_start = time.perf_counter()
        model, token_to_id, id_to_token = gpt2_model_load(args.model, sess, args.n_ctx)
        t_load = time.perf_counter() - t_start

        # size the compute memory once, for the worst-case graph
        n_tokens = min(model.n_ctx, args.n_batch)
        n_past_wc = model.n_ctx - n_tokens
        with sess.graph() as g:
            *_, logits_wc = gpt2_graph(model, g, n_past_wc, n_tokens)
            g.reserve(logits_wc)
        mem = sum(sess.sched.buffer_size(b) for b in sess.backends)
        print(f"main: compute buffer size: {mem / (1024.0 * 1024.0):.2f} MB")

        # tokenize the prompt
        embd_inp = gpt_tokenize(token_to_id, args.prompt)
        n_predict = min(args.n_predict, model.n_ctx - len(embd_inp))

        print(f"main: prompt: '{args.prompt}'")
        print(f"main: number of tokens in prompt = {len(embd_inp)}, "
              f"first 8 tokens: {embd_inp[:8]}")
        print()

        n_past = 0
        t_sample = 0.0
        t_predict = 0.0
        logits = None
        embd: list[int] = []
        # tokens carry raw bytes that may split a multi-byte UTF-8 char across
        # token boundaries; feed them through one incremental decoder.
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        i = 0
        while i < len(embd_inp) + n_predict:
            # predict
            if len(embd) > 0:
                t0 = time.perf_counter()
                logits = gpt2_eval(model, sess, n_past, embd)
                t_predict += time.perf_counter() - t0

            n_past += len(embd)
            embd.clear()

            if i >= len(embd_inp):
                # sample the next token
                t0 = time.perf_counter()
                token_id = gpt_sample_top_k_top_p(
                    logits, args.top_k, args.top_p, args.temp, rng
                )
                t_sample += time.perf_counter() - t0
                embd.append(token_id)
            else:
                # still consuming the prompt, in batches of n_batch
                while i < len(embd_inp):
                    embd.append(embd_inp[i])
                    i += 1
                    if len(embd) >= args.n_batch:
                        break
                i -= 1

            for token_id in embd:
                piece = decoder.decode(id_to_token.get(token_id, b""))
                print(piece, end="", flush=True)

            if embd[-1] == model.eos_token:  # end-of-text
                break

            i += 1

        t_main_end = time.perf_counter()
        print("\n")
        print(f"main:     load time = {t_load * 1000:8.2f} ms")
        print(f"main:   sample time = {t_sample * 1000:8.2f} ms")
        print(f"main:  predict time = {t_predict * 1000:8.2f} ms"
              f" / {t_predict * 1000 / max(n_past, 1):.2f} ms per token")
        print(f"main:    total time = {(t_main_end - t_main_start) * 1000:8.2f} ms")


if __name__ == "__main__":
    main()

"""GPT-2 inference on the ``ggbond.ggml`` OO wrapper.

A port of ggml's ``gpt-2`` ``main-backend`` example to the ``ggbond.ggml`` API
that reads a GGUF model: :class:`GGUF` parses the header and fills a metadata
context with the file's tensors, a backend buffer is allocated for them, and
the tensor data is uploaded straight from the mapped file. Hyperparameters and
the vocabulary come from the GGUF metadata.

The two-phase ggml flow is explicit throughout: per token we reset one
metadata :class:`Context`, build a fresh graph in it, let a reused
:class:`GAllocr` allocate its working memory, upload the input ids, compute, and
read the last token's logits back into numpy.

Usage::

    python examples/ggml/gpt2.py -m models/gpt2.Q8_0.gguf -p "Hello"
    python examples/ggml/gpt2.py -m models/gpt2.Q8_0.gguf -b metal

Get a model from Hugging Face, where llama.cpp's conversions of GPT-2 are
published (any quantization works; the tensor and metadata names are the ones
llama.cpp's ``convert_hf_to_gguf.py`` writes)::

    curl -L -o models/gpt2.Q8_0.gguf \\
        https://huggingface.co/mradermacher/gpt2-GGUF/resolve/main/gpt2.Q8_0.gguf
"""

import argparse
import codecs
import math
import mmap
import random
import re
import time

import numpy as np

from ggbond.ggml import GGUF, Backend, Context, GAllocr, Tensor, F32, I32

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
    """Owns the weight + KV contexts so they outlive every graph built from them,
    and the context each evaluation builds its graph in."""

    def __init__(self, ctx_w: Context):
        self.ctx_w = ctx_w  # the context gguf_init_from_file filled with tensor metadata
        self.ctx_kv: Context | None = None
        # Holds one evaluation's graph at a time; reset before each one.
        self.ctx_graph = Context.for_tensors(0, graph_size=GPT2_MAX_NODES)
        self.weights: dict[str, Tensor] = {}  # named as in the GGUF
        self.lm_head = self.memory_k = self.memory_v = None
        self.n_vocab = self.n_ctx = self.n_embd = self.n_head = self.n_layer = 0
        self.eps = 1e-5
        self.eos_token = 50256

    def close(self):
        # Both contexts must outlive any graph referencing their tensors; they
        # are freed independently of the backend buffers that hold the data.
        self.ctx_graph.close()
        if self.ctx_kv is not None:
            self.ctx_kv.close()
        self.ctx_w.close()


def gpt2_model_load(fname: str, backend: Backend, n_ctx: int):
    """Load a GPT-2 GGUF. Returns (model, token_to_id, id_to_token)."""
    print(f"gpt2_model_load: loading model from '{fname}'")

    # The GGUF handle is only the parsed header: it is closed when loading is
    # done, while the metadata context it filled stays with the model.
    with GGUF.from_file(fname, no_alloc=True) as gguf:
        model = GPT2Model(gguf.context)
        try:
            hparams = _gpt2_hparams(gguf, n_ctx)
            for key, value in hparams.items():
                print(f"gpt2_model_load: {key:9} = {value}")
                setattr(model, key, value)

            # -- weights: one backend buffer for every tensor, filled from the file --
            buf_w = backend.alloc_ctx_tensors(model.ctx_w)
            print(f"gpt2_model_load: backend buffer size = {buf_w.size / (1024.0 * 1024.0):6.2f} MB")
            with open(fname, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                view = memoryview(mapped)
                try:
                    for i in range(gguf.n_tensors):
                        name = gguf.get_tensor_name(i)
                        tensor = model.ctx_w.get_tensor(name)
                        start = gguf.data_offset + gguf.get_tensor_offset(i)
                        end = start + tensor.nbytes
                        if end > len(view):
                            raise ValueError(f"truncated model: tensor '{name}' ends at byte {end} of {len(view)}")
                        tensor.write_raw(view[start:end])
                        model.weights[name] = tensor
                finally:
                    view.release()

            model.n_vocab = model.weights["token_embd.weight"].ne[1]
            # llama.cpp's converter writes the tied LM head out as its own
            # tensor; fall back on the token embedding for a file that does not.
            model.lm_head = model.weights.get("output.weight", model.weights["token_embd.weight"])

            # -- key + value memory (uses the *user* n_ctx) --
            n_elements = model.n_embd * model.n_layer * n_ctx
            model.ctx_kv = Context.for_tensors(2)
            model.memory_k = model.ctx_kv.new_tensor_1d(F32, n_elements)
            model.memory_v = model.ctx_kv.new_tensor_1d(F32, n_elements)
            # A fresh backend buffer is uninitialized, and the graph reads the
            # whole cache range before the mask hides the unwritten part.
            buf_kv = backend.alloc_ctx_tensors(model.ctx_kv)
            buf_kv.clear(0)
            print(f"gpt2_model_load: memory size = {buf_kv.size / (1024.0 * 1024.0):8.2f} MB")

            return (model, *gpt2_vocab(gguf))
        except BaseException:
            model.close()
            raise


# ============================================================================
# Graph
# ============================================================================

def gpt2_graph(model: GPT2Model, ctx: Context, n_past: int, n_tokens: int):
    """Build the forward graph in ``ctx``. Returns (graph, embd, position, logits)."""
    N = n_tokens
    w = model.weights
    n_embd, n_layer, n_ctx, n_head = model.n_embd, model.n_layer, model.n_ctx, model.n_head

    # Weights and the KV cache live in the model's own contexts, which are
    # sized only for their metadata. Ops whose first operand is one of them go
    # through ``ctx`` (``ctx.mul_mat(w, x)`` is ``ggml_mul_mat(ctx, w, x)``) so
    # the result nodes land in this graph's context.
    memory_k = model.memory_k
    memory_v = model.memory_v
    k_esz = memory_k.element_size  # bytes per element (F32 -> 4)

    graph = ctx.new_graph(size=GPT2_MAX_NODES)

    embd = ctx.new_tensor_1d(I32, N, name="embd").set_input()
    position = ctx.new_tensor_1d(I32, N, name="position").set_input()

    def norm(x, name):  # layer norm with the gain and bias called ``name``
        return x.norm(model.eps).mul(w[f"{name}.weight"]).add(w[f"{name}.bias"])

    def linear(x, name):
        return ctx.mul_mat(w[f"{name}.weight"], x).add(w[f"{name}.bias"])

    # token + position embedding
    inpL = ctx.get_rows(w["token_embd.weight"], embd).add(ctx.get_rows(w["position_embd.weight"], position))

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
            k = ctx.view_1d(memory_k, N * n_embd, k_esz * n_embd * (il * n_ctx + n_past))
            v = ctx.view_1d(memory_v, N * n_embd, k_esz * n_embd * (il * n_ctx + n_past))
            graph.build_forward_expand(Kcur.cpy(k))
            graph.build_forward_expand(Vcur.cpy(v))

        # Q = [64, N, 12]
        Q = Qcur.cont_3d(n_embd // n_head, n_head, N).permute(0, 2, 1, 3)

        # K = [64, n_past + N, 12]
        K = (
            ctx.view_1d(memory_k, (n_past + N) * n_embd, il * n_ctx * k_esz * n_embd)
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
            ctx.view_1d(memory_v, (n_past + N) * n_embd, il * n_ctx * k_esz * n_embd)
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
    logits = ctx.mul_mat(model.lm_head, inpL)
    logits.name = "logits"
    logits.set_output()

    graph.build_forward_expand(logits)
    return graph, embd, position, logits


# ============================================================================
# Evaluation
# ============================================================================

def gpt2_eval(model, backend, allocr, n_past, embd_inp) -> np.ndarray:
    """Run the transformer and return logits for the last token."""
    N = len(embd_inp)
    n_vocab = model.n_vocab
    # Reuse one metadata context for every evaluation: ggml_reset empties it,
    # which also retires the previous evaluation's tensors and graph.
    ctx = model.ctx_graph
    ctx.reset()
    graph, embd, position, logits = gpt2_graph(model, ctx, n_past, N)

    allocr.alloc_graph(graph)

    embd.write(np.asarray(embd_inp, dtype=np.int32))
    position.write(np.arange(n_past, n_past + N, dtype=np.int32))

    backend.compute(graph)

    # logits hold [n_vocab, N] row-major (n_vocab fastest); return just the
    # last token's row, mirroring the C++ which reads that slice directly.
    return logits.to_numpy().reshape(N, n_vocab)[N - 1].copy()


# ============================================================================
# Main
# ============================================================================

def _init_backend(kind: str, device: int) -> Backend:
    kind = kind.lower()
    if kind == "cpu":
        return Backend.cpu_init()
    if kind == "cuda":
        return Backend.cuda_init(device)
    if kind == "metal":
        return Backend.metal_init()
    if kind == "hip":
        return Backend.hip_init(device)
    raise ValueError(f"unknown backend '{kind}' (cpu|cuda|metal|hip)")


def main():
    parser = argparse.ArgumentParser(description="GPT-2 inference on the ggbond.ggml API")
    parser.add_argument("-m", "--model", default="models/gpt2.Q8_0.gguf",
                        help="Path to a GPT-2 model in GGUF format")
    parser.add_argument("-p", "--prompt", default="Hello, my name is", help="Input prompt")
    parser.add_argument("-n", "--n-predict", type=int, default=200, help="Tokens to predict")
    parser.add_argument("--n-ctx", type=int, default=1024, help="Context size")
    parser.add_argument("--n-batch", type=int, default=32, help="Prompt batch size")
    parser.add_argument("-t", "--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("-b", "--backend", default="cpu", help="Backend (cpu|cuda|metal|hip)")
    parser.add_argument("--device", type=int, default=0, help="GPU device index")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 = random)")
    args = parser.parse_args()

    t_main_start = time.perf_counter()

    seed = args.seed if args.seed >= 0 else random.randint(0, 2**31 - 1)
    print(f"main: seed = {seed}")
    rng = random.Random(seed)

    backend = _init_backend(args.backend, args.device)
    if backend.is_cpu:
        backend.set_n_threads(args.threads)

    allocr = None
    model = None
    try:
        t_start = time.perf_counter()
        model, token_to_id, id_to_token = gpt2_model_load(args.model, backend, args.n_ctx)
        t_load = time.perf_counter() - t_start

        # compute-buffer allocator: reserve once for the worst-case graph
        allocr = GAllocr.from_backend(backend)
        n_tokens = min(model.n_ctx, args.n_batch)
        n_past_wc = model.n_ctx - n_tokens
        gworst, *_ = gpt2_graph(model, model.ctx_graph, n_past_wc, n_tokens)
        allocr.reserve(gworst)
        mem_mb = allocr.buffer_size(0) / (1024.0 * 1024.0)
        print(f"main: compute buffer size: {mem_mb:.2f} MB")

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
                logits = gpt2_eval(model, backend, allocr, n_past, embd)
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
    finally:
        if allocr is not None:
            allocr.close()
        if model is not None:
            model.close()
        backend.close()


if __name__ == "__main__":
    main()

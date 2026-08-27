"""GPT-2 inference on the ``ggbond.ggml`` OO wrapper.

A faithful port of ggml's ``gpt-2`` ``main-backend`` example to the
``ggbond.ggml`` API. It reads the *legacy* ggml model format (``ggml-model.bin``
produced by ggml's convert script -- magic ``ggml``), not GGUF: hparams, the
inline vocab, and the named weights are parsed straight from the file with plain
Python I/O, then uploaded into backend-resident tensors.

The two-phase ggml flow is explicit throughout: per token we build a fresh
metadata graph in its own throwaway :class:`Context`, let a reused
:class:`GAllocr` allocate its working memory, upload the input ids, compute, and
read the last token's logits back into numpy.

Usage::

    python examples/gpt2.py -m models/gpt-2-117M/ggml-model.bin -p "Hello"
    python examples/gpt2.py -m models/gpt-2-117M/ggml-model.bin -b cuda

Get a model with ggml's gpt-2 ``convert-ckpt-to-ggml.py`` / ``download-model.sh``
(the legacy ``.bin`` format this example reads).
"""

import argparse
import codecs
import math
import random
import re
import struct
import time

import numpy as np

import ggml as _ggml  # legacy-format constants + ftype->type conversion only
from ggbond.ggml import Backend, Context, GAllocr, Tensor, DType, F32, I32

GPT2_MAX_NODES = 4096


def _read_exact(f, size: int, what: str) -> bytes:
    data = f.read(size)
    if len(data) != size:
        raise ValueError(f"truncated model while reading {what}: expected {size} bytes, got {len(data)}")
    return data


# ============================================================================
# Tokenizer + sampling (model-format independent; ported from common.cpp)
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

class GPT2HParams:
    def __init__(self, n_vocab, n_ctx, n_embd, n_head, n_layer, eps=1e-5):
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.eps = eps


class GPT2Layer:
    __slots__ = (
        "ln_1_g", "ln_1_b", "ln_2_g", "ln_2_b",
        "c_attn_attn_w", "c_attn_attn_b", "c_attn_proj_w", "c_attn_proj_b",
        "c_mlp_fc_w", "c_mlp_fc_b", "c_mlp_proj_w", "c_mlp_proj_b",
    )


class GPT2Model:
    """Owns the weight + KV contexts so they outlive every graph built from them."""

    def __init__(self):
        self.hparams = None
        self.ln_f_g = self.ln_f_b = None
        self.wte = self.wpe = self.lm_head = None
        self.layers: list[GPT2Layer] = []
        self.memory_k = self.memory_v = None
        self.ctx_w: Context | None = None
        self.ctx_kv: Context | None = None

    def close(self):
        # Both contexts must outlive any graph referencing their tensors; they
        # are freed independently of the backend buffers that hold the data.
        if self.ctx_kv is not None:
            self.ctx_kv.close()
        if self.ctx_w is not None:
            self.ctx_w.close()


def gpt2_model_load(fname: str, backend: Backend, n_ctx: int):
    """Load a legacy ggml GPT-2 model. Returns (model, token_to_id, id_to_token)."""
    print(f"gpt2_model_load: loading model from '{fname}'")

    with open(fname, "rb") as f:
        # -- verify magic --
        (magic,) = struct.unpack("<I", _read_exact(f, 4, "magic"))
        if magic != _ggml.GGML_FILE_MAGIC:
            raise ValueError(f"invalid model file '{fname}' (bad magic)")

        # -- hparams -- (n_ctx here is the *trained* context length)
        n_vocab, n_ctx_train, n_embd, n_head, n_layer, ftype = struct.unpack(
            "<6i", _read_exact(f, 24, "hyperparameters")
        )
        if min(n_vocab, n_ctx_train, n_embd, n_head, n_layer) <= 0:
            raise ValueError("model contains invalid non-positive hyperparameters")
        if n_embd % n_head != 0:
            raise ValueError(f"model embedding size {n_embd} is not divisible by {n_head} heads")
        if not (0 < n_ctx <= n_ctx_train):
            raise ValueError(
                f"requested context size must be between 1 and the trained size {n_ctx_train}, got {n_ctx}"
            )
        qntvr = ftype // _ggml.GGML_QNT_VERSION_FACTOR
        print(f"gpt2_model_load: n_vocab = {n_vocab}")
        print(f"gpt2_model_load: n_ctx   = {n_ctx_train}")
        print(f"gpt2_model_load: n_embd  = {n_embd}")
        print(f"gpt2_model_load: n_head  = {n_head}")
        print(f"gpt2_model_load: n_layer = {n_layer}")
        print(f"gpt2_model_load: ftype   = {ftype}")
        print(f"gpt2_model_load: qntvr   = {qntvr}")
        ftype %= _ggml.GGML_QNT_VERSION_FACTOR

        # type of the big tensors (F16 / quantized) vs the always-F32 small ones
        wtype = DType(_ggml.ggml_ftype_to_ggml_type(ftype))

        # -- vocab (read inline, byte-level-encoded UTF-8 strings) --
        (n_vocab2,) = struct.unpack("<i", _read_exact(f, 4, "vocabulary size"))
        if n_vocab2 != n_vocab:
            raise ValueError(f"bad vocab size {n_vocab2} != {n_vocab}")
        # Tokens are stored as the *raw bytes* of each piece (the convert script
        # decodes GPT-2's byte-level encoding back to bytes), so many are not
        # valid UTF-8 on their own. Key the vocab by bytes and match on bytes.
        token_to_id: dict[bytes, int] = {}
        id_to_token: dict[int, bytes] = {}
        for i in range(n_vocab):
            (length,) = struct.unpack("<I", _read_exact(f, 4, f"token {i} length"))
            word = _read_exact(f, length, f"token {i}")
            token_to_id[word] = i
            id_to_token[i] = word

        model = GPT2Model()
        model.hparams = GPT2HParams(n_vocab, n_ctx, n_embd, n_head, n_layer)

        # -- create the weight tensors (metadata only; no_alloc context) --
        n_tensors = 2 + 6 + 12 * n_layer
        ctx_w = Context.for_tensors(n_tensors)
        model.ctx_w = ctx_w

        tensors: dict[str, Tensor] = {}

        model.ln_f_g = ctx_w.new_tensor_1d(F32, n_embd)
        model.ln_f_b = ctx_w.new_tensor_1d(F32, n_embd)
        model.wte = ctx_w.new_tensor_2d(wtype, n_embd, n_vocab)
        model.wpe = ctx_w.new_tensor_2d(F32, n_embd, n_ctx_train)
        model.lm_head = ctx_w.new_tensor_2d(wtype, n_embd, n_vocab)

        tensors["model/ln_f/g"] = model.ln_f_g
        tensors["model/ln_f/b"] = model.ln_f_b
        tensors["model/wte"] = model.wte
        tensors["model/wpe"] = model.wpe
        tensors["model/lm_head"] = model.lm_head

        for i in range(n_layer):
            layer = GPT2Layer()
            layer.ln_1_g = ctx_w.new_tensor_1d(F32, n_embd)
            layer.ln_1_b = ctx_w.new_tensor_1d(F32, n_embd)
            layer.ln_2_g = ctx_w.new_tensor_1d(F32, n_embd)
            layer.ln_2_b = ctx_w.new_tensor_1d(F32, n_embd)
            layer.c_attn_attn_w = ctx_w.new_tensor_2d(wtype, n_embd, 3 * n_embd)
            layer.c_attn_attn_b = ctx_w.new_tensor_1d(F32, 3 * n_embd)
            layer.c_attn_proj_w = ctx_w.new_tensor_2d(wtype, n_embd, n_embd)
            layer.c_attn_proj_b = ctx_w.new_tensor_1d(F32, n_embd)
            layer.c_mlp_fc_w = ctx_w.new_tensor_2d(wtype, n_embd, 4 * n_embd)
            layer.c_mlp_fc_b = ctx_w.new_tensor_1d(F32, 4 * n_embd)
            layer.c_mlp_proj_w = ctx_w.new_tensor_2d(wtype, 4 * n_embd, n_embd)
            layer.c_mlp_proj_b = ctx_w.new_tensor_1d(F32, n_embd)
            model.layers.append(layer)

            p = f"model/h{i}"
            tensors[f"{p}/ln_1/g"] = layer.ln_1_g
            tensors[f"{p}/ln_1/b"] = layer.ln_1_b
            tensors[f"{p}/ln_2/g"] = layer.ln_2_g
            tensors[f"{p}/ln_2/b"] = layer.ln_2_b
            tensors[f"{p}/attn/c_attn/w"] = layer.c_attn_attn_w
            tensors[f"{p}/attn/c_attn/b"] = layer.c_attn_attn_b
            tensors[f"{p}/attn/c_proj/w"] = layer.c_attn_proj_w
            tensors[f"{p}/attn/c_proj/b"] = layer.c_attn_proj_b
            tensors[f"{p}/mlp/c_fc/w"] = layer.c_mlp_fc_w
            tensors[f"{p}/mlp/c_fc/b"] = layer.c_mlp_fc_b
            tensors[f"{p}/mlp/c_proj/w"] = layer.c_mlp_proj_w
            tensors[f"{p}/mlp/c_proj/b"] = layer.c_mlp_proj_b

        # allocate a backend buffer for all weight tensors
        buf_w = backend.alloc_ctx_tensors(ctx_w)
        size_mb = buf_w.size / (1024.0 * 1024.0)
        print(f"gpt2_model_load: backend buffer size = {size_mb:6.2f} MB")

        # -- key + value memory (uses the *user* n_ctx) --
        n_mem = n_layer * n_ctx
        n_elements = n_embd * n_mem
        ctx_kv = Context.for_tensors(2)
        model.ctx_kv = ctx_kv
        model.memory_k = ctx_kv.new_tensor_1d(F32, n_elements, name="memory_k")
        model.memory_v = ctx_kv.new_tensor_1d(F32, n_elements, name="memory_v")
        buf_kv = backend.alloc_ctx_tensors(ctx_kv)
        kv_mb = buf_kv.size / (1024.0 * 1024.0)
        print(f"gpt2_model_load: memory size = {kv_mb:8.2f} MB, n_mem = {n_mem}")

        # -- load weights --
        total_size = 0
        has_lm_head = False
        loaded_names = set()
        while True:
            header = f.read(12)
            if not header:
                break  # EOF
            if len(header) != 12:
                raise ValueError(
                    f"truncated model while reading tensor header: expected 12 bytes, got {len(header)}"
                )
            n_dims, length, ttype = struct.unpack("<3i", header)
            if n_dims not in (1, 2):
                raise ValueError(f"invalid tensor dimension count {n_dims}")
            if length < 0:
                raise ValueError(f"invalid negative tensor name length {length}")
            ne = [1, 1]
            nelements = 1
            for i in range(n_dims):
                (ne[i],) = struct.unpack(
                    "<i", _read_exact(f, 4, f"tensor dimension {i}")
                )
                if ne[i] <= 0:
                    raise ValueError(f"invalid tensor dimension {ne[i]}")
                nelements *= ne[i]
            name = _read_exact(f, length, "tensor name").decode("utf-8")

            if name not in tensors:
                raise ValueError(f"unknown tensor '{name}' in model file")
            if name in loaded_names:
                raise ValueError(f"duplicate tensor '{name}' in model file")
            tensor = tensors[name]
            tensor.name = name

            if ttype != int(tensor.dtype):
                raise ValueError(
                    f"tensor '{name}' has type {ttype}, expected {int(tensor.dtype)} ({tensor.dtype.name})"
                )

            if tensor.nelements != nelements:
                raise ValueError(f"tensor '{name}' has wrong size in model file")
            te = tensor.ne
            if te[0] != ne[0] or (len(te) > 1 and te[1] != ne[1]):
                raise ValueError(
                    f"tensor '{name}' has wrong shape: got {list(te)}, expected {ne[:n_dims]}"
                )

            tensor.set_raw(_read_exact(f, tensor.nbytes, f"tensor '{name}' data"))
            loaded_names.add(name)

            # GPT-2 ties the LM head to the token embedding (wte) unless the file
            # carries an explicit lm_head.
            if name == "model/wte" and not has_lm_head:
                model.lm_head = tensor
            if name == "model/lm_head":
                has_lm_head = True
                model.lm_head = tensor

            total_size += tensor.nbytes

        required_names = set(tensors)
        required_names.discard("model/lm_head")  # optional: tied to model/wte
        missing = sorted(required_names - loaded_names)
        if missing:
            preview = ", ".join(missing[:5])
            suffix = " ..." if len(missing) > 5 else ""
            raise ValueError(f"model is missing {len(missing)} required tensors: {preview}{suffix}")

        print(f"gpt2_model_load: model size  = {total_size / 1024.0 / 1024.0:8.2f} MB")

    return model, token_to_id, id_to_token


# ============================================================================
# Graph
# ============================================================================

def gpt2_graph(model: GPT2Model, ctx: Context, n_past: int, n_tokens: int):
    """Build the forward graph in ``ctx``. Returns (graph, embd, position, logits)."""
    N = n_tokens
    hp = model.hparams
    n_embd, n_layer, n_ctx, n_head = hp.n_embd, hp.n_layer, hp.n_ctx, hp.n_head

    # Rebind weight/KV tensors to this graph's context so op-result nodes land
    # here (the weight context is sized only for weight metadata).
    R = ctx.ref

    memory_k = R(model.memory_k)
    memory_v = R(model.memory_v)
    k_esz = memory_k.element_size  # bytes per element (F32 -> 4)

    graph = ctx.new_graph(size=GPT2_MAX_NODES)

    embd = ctx.new_tensor_1d(I32, N, name="embd").set_input()
    position = ctx.new_tensor_1d(I32, N, name="position").set_input()

    # wte + wpe
    inpL = R(model.wte).get_rows(embd).add(R(model.wpe).get_rows(position))

    for il in range(n_layer):
        layer = model.layers[il]

        # ln_1
        cur = inpL.norm(hp.eps).mul(R(layer.ln_1_g)).add(R(layer.ln_1_b))

        # attention: fused QKV projection -> [3*n_embd, N]
        cur = R(layer.c_attn_attn_w).mul_mat(cur).add(R(layer.c_attn_attn_b))

        # row stride of the contiguous [3*n_embd, N] QKV tensor
        nb1 = cur.nb4[1]
        Qcur = cur.view_2d(n_embd, N, nb1, 0 * 4 * n_embd)
        Kcur = cur.view_2d(n_embd, N, nb1, 1 * 4 * n_embd)
        Vcur = cur.view_2d(n_embd, N, nb1, 2 * 4 * n_embd)

        # store the new keys/values into the KV cache
        if N >= 1:
            k = memory_k.view_1d(N * n_embd, k_esz * n_embd * (il * n_ctx + n_past))
            v = memory_v.view_1d(N * n_embd, k_esz * n_embd * (il * n_ctx + n_past))
            graph.build_forward_expand(Kcur.cpy(k))
            graph.build_forward_expand(Vcur.cpy(v))

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
        cur = R(layer.c_attn_proj_w).mul_mat(cur).add(R(layer.c_attn_proj_b))

        # residual
        cur = cur.add(inpL)
        inpFF = cur

        # ln_2 + MLP
        cur = inpFF.norm(hp.eps).mul(R(layer.ln_2_g)).add(R(layer.ln_2_b))
        cur = R(layer.c_mlp_fc_w).mul_mat(cur).add(R(layer.c_mlp_fc_b)).gelu()
        cur = R(layer.c_mlp_proj_w).mul_mat(cur).add(R(layer.c_mlp_proj_b))

        # residual
        inpL = cur.add(inpFF)

    # final norm
    inpL = inpL.norm(hp.eps).mul(R(model.ln_f_g)).add(R(model.ln_f_b))

    # logits = lm_head * inpL
    logits = R(model.lm_head).mul_mat(inpL)
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
    n_vocab = model.hparams.n_vocab
    with Context.for_tensors(0, graph_size=GPT2_MAX_NODES) as ctx:
        graph, embd, position, logits = gpt2_graph(model, ctx, n_past, N)

        allocr.alloc_graph(graph)

        embd.set(np.asarray(embd_inp, dtype=np.int32))
        position.set(np.arange(n_past, n_past + N, dtype=np.int32))

        backend.compute(graph)
        backend.synchronize()

        # logits hold [n_vocab, N] row-major (n_vocab fastest); return just the
        # last token's row, mirroring the C++ which reads that slice directly.
        flat = logits.get(np.empty(logits.nelements, dtype=np.float32))
        return flat[(N - 1) * n_vocab: N * n_vocab].copy()


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
    parser.add_argument("-m", "--model", default="models/gpt-2-117M/ggml-model.bin",
                        help="Path to a legacy ggml GPT-2 model (.bin)")
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
        n_tokens = min(model.hparams.n_ctx, args.n_batch)
        n_past_wc = model.hparams.n_ctx - n_tokens
        with Context.for_tensors(0, graph_size=GPT2_MAX_NODES) as wctx:
            gworst, *_ = gpt2_graph(model, wctx, n_past_wc, n_tokens)
            allocr.reserve(gworst)
        mem_mb = allocr.buffer_size(0) / (1024.0 * 1024.0)
        print(f"main: compute buffer size: {mem_mb:.2f} MB")

        # tokenize the prompt
        embd_inp = gpt_tokenize(token_to_id, args.prompt)
        n_predict = min(args.n_predict, model.hparams.n_ctx - len(embd_inp))

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

            if embd[-1] == 50256:  # end-of-text
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

"""
GPT-2 inference using ggbond Session API.

Ported from vendor/ggml/examples/gpt-2/main-backend.cpp

Usage:
    python examples/ggbond_gpt2.py -m models/gpt-2-117M/ggml-model-f32.bin -p "Hello, world"

Download model:
    cd vendor/ggml
    python examples/gpt-2/download-model.py 124M
    python examples/gpt-2/convert-ckpt-to-ggml.py models/gpt-2-117M/ 0
"""

import argparse
import math
import re
import struct
import sys
import random

import numpy as np

import ggbond
from ggbond import ggml
from ggbond.context import Context

GPT2_MAX_NODES = 4096


# ============================================================================
# Tokenizer
# ============================================================================

def gpt_tokenize(vocab_token_to_id: dict[str, int], text: str) -> list[int]:
    """BPE tokenizer matching the C++ gpt_tokenize implementation."""
    pat = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""",
        re.UNICODE,
    )
    words = pat.findall(text)

    tokens: list[int] = []
    for word in words:
        i = 0
        while i < len(word):
            best_len = 0
            best_id = -1
            for j in range(len(word), i, -1):
                sub = word[i:j]
                if sub in vocab_token_to_id:
                    best_len = j - i
                    best_id = vocab_token_to_id[sub]
                    break
            if best_id == -1:
                i += 1
            else:
                tokens.append(best_id)
                i += best_len
    return tokens


def gpt_sample_top_k_top_p(
    logits: np.ndarray,
    top_k: int,
    top_p: float,
    temp: float,
    rng: random.Random,
) -> int:
    """Top-k / top-p sampling from logits."""
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

    max_logit = np.max(top_logits)
    exps = np.exp(top_logits - max_logit)
    probs = exps / np.sum(exps)

    sorted_order = np.argsort(-probs)
    sorted_probs = probs[sorted_order]
    sorted_indices = indices[sorted_order]

    if top_p < 1.0:
        cumsum = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumsum, top_p) + 1
        sorted_probs = sorted_probs[:cutoff]
        sorted_indices = sorted_indices[:cutoff]
        sorted_probs = sorted_probs / np.sum(sorted_probs)

    r = rng.random()
    cumsum = 0.0
    for i, p in enumerate(sorted_probs):
        cumsum += p
        if r < cumsum:
            return int(sorted_indices[i])
    return int(sorted_indices[-1])


# ============================================================================
# Model structures
# ============================================================================

class GPT2HParams:
    def __init__(self):
        self.n_vocab = 50257
        self.n_ctx = 1024
        self.n_embd = 768
        self.n_head = 12
        self.n_layer = 12
        self.ftype = 1
        self.eps = 1e-5


class GPT2Layer:
    def __init__(self):
        self.ln_1_g = None
        self.ln_1_b = None
        self.ln_2_g = None
        self.ln_2_b = None
        self.c_attn_attn_w = None
        self.c_attn_attn_b = None
        self.c_attn_proj_w = None
        self.c_attn_proj_b = None
        self.c_mlp_fc_w = None
        self.c_mlp_fc_b = None
        self.c_mlp_proj_w = None
        self.c_mlp_proj_b = None


class GPT2Model:
    def __init__(self):
        self.hparams = GPT2HParams()
        self.ln_f_g = None
        self.ln_f_b = None
        self.wte = None
        self.wpe = None
        self.lm_head = None
        self.layers: list[GPT2Layer] = []
        self.memory_k = None
        self.memory_v = None
        self.tensors: dict[str, object] = {}


# ============================================================================
# Model loading
# ============================================================================

def gpt2_model_load(
    fname: str, model: GPT2Model, session: ggbond.Session, n_ctx: int, n_gpu_layers: int
) -> tuple[dict[str, int], dict[int, str]]:
    """Load GPT-2 model from binary file. Returns (token_to_id, id_to_token)."""
    print(f"gpt2_model_load: loading model from '{fname}'")

    with open(fname, "rb") as f:
        (magic,) = struct.unpack("<I", f.read(4))
        if magic != ggml.FILE_MAGIC:
            raise ValueError(
                f"invalid model file '{fname}' (bad magic: {magic:#x}, "
                f"expected {ggml.FILE_MAGIC:#x})"
            )

        hp = model.hparams
        hp.n_vocab, hp.n_ctx, hp.n_embd, hp.n_head, hp.n_layer, hp.ftype = (
            struct.unpack("<6i", f.read(24))
        )

        qntvr = hp.ftype // ggml.QNT_VERSION_FACTOR
        print(f"gpt2_model_load: n_vocab = {hp.n_vocab}")
        print(f"gpt2_model_load: n_ctx   = {hp.n_ctx}")
        print(f"gpt2_model_load: n_embd  = {hp.n_embd}")
        print(f"gpt2_model_load: n_head  = {hp.n_head}")
        print(f"gpt2_model_load: n_layer = {hp.n_layer}")
        print(f"gpt2_model_load: ftype   = {hp.ftype}")
        print(f"gpt2_model_load: qntvr   = {qntvr}")

        hp.ftype %= ggml.QNT_VERSION_FACTOR

        (n_vocab_file,) = struct.unpack("<i", f.read(4))
        if n_vocab_file != hp.n_vocab:
            raise ValueError(
                f"invalid model file '{fname}' "
                f"(bad vocab size {n_vocab_file} != {hp.n_vocab})"
            )

        token_to_id: dict[str, int] = {}
        id_to_token: dict[int, str] = {}
        for i in range(hp.n_vocab):
            (length,) = struct.unpack("<I", f.read(4))
            word = f.read(length).decode("utf-8", errors="replace")
            token_to_id[word] = i
            id_to_token[i] = word

        wtype = ggml.Type(ggml.ftype_to_ggml_type(hp.ftype))

        n_tensors = 2 + 6 + 12 * hp.n_layer
        ctx_w = session.context(n_tensors=n_tensors)

        n_embd = hp.n_embd
        n_layer = hp.n_layer

        model.ln_f_g = ctx_w.new_tensor(ggml.Type.F32, n_embd)
        model.ln_f_b = ctx_w.new_tensor(ggml.Type.F32, n_embd)

        model.wte = ctx_w.new_tensor(wtype, n_embd, hp.n_vocab)
        model.wpe = ctx_w.new_tensor(ggml.Type.F32, n_embd, hp.n_ctx)
        model.lm_head = ctx_w.new_tensor(wtype, n_embd, hp.n_vocab)

        model.tensors["model/ln_f/g"] = model.ln_f_g
        model.tensors["model/ln_f/b"] = model.ln_f_b
        model.tensors["model/wte"] = model.wte
        model.tensors["model/wpe"] = model.wpe
        model.tensors["model/lm_head"] = model.lm_head

        model.layers = []
        for i in range(n_layer):
            layer = GPT2Layer()

            layer.ln_1_g = ctx_w.new_tensor(ggml.Type.F32, n_embd)
            layer.ln_1_b = ctx_w.new_tensor(ggml.Type.F32, n_embd)
            layer.ln_2_g = ctx_w.new_tensor(ggml.Type.F32, n_embd)
            layer.ln_2_b = ctx_w.new_tensor(ggml.Type.F32, n_embd)

            layer.c_attn_attn_w = ctx_w.new_tensor(wtype, n_embd, 3 * n_embd)
            layer.c_attn_attn_b = ctx_w.new_tensor(ggml.Type.F32, 3 * n_embd)
            layer.c_attn_proj_w = ctx_w.new_tensor(wtype, n_embd, n_embd)
            layer.c_attn_proj_b = ctx_w.new_tensor(ggml.Type.F32, n_embd)

            layer.c_mlp_fc_w = ctx_w.new_tensor(wtype, n_embd, 4 * n_embd)
            layer.c_mlp_fc_b = ctx_w.new_tensor(ggml.Type.F32, 4 * n_embd)
            layer.c_mlp_proj_w = ctx_w.new_tensor(wtype, 4 * n_embd, n_embd)
            layer.c_mlp_proj_b = ctx_w.new_tensor(ggml.Type.F32, n_embd)

            model.layers.append(layer)

            prefix = f"model/h{i}"
            model.tensors[f"{prefix}/ln_1/g"] = layer.ln_1_g
            model.tensors[f"{prefix}/ln_1/b"] = layer.ln_1_b
            model.tensors[f"{prefix}/ln_2/g"] = layer.ln_2_g
            model.tensors[f"{prefix}/ln_2/b"] = layer.ln_2_b
            model.tensors[f"{prefix}/attn/c_attn/w"] = layer.c_attn_attn_w
            model.tensors[f"{prefix}/attn/c_attn/b"] = layer.c_attn_attn_b
            model.tensors[f"{prefix}/attn/c_proj/w"] = layer.c_attn_proj_w
            model.tensors[f"{prefix}/attn/c_proj/b"] = layer.c_attn_proj_b
            model.tensors[f"{prefix}/mlp/c_fc/w"] = layer.c_mlp_fc_w
            model.tensors[f"{prefix}/mlp/c_fc/b"] = layer.c_mlp_fc_b
            model.tensors[f"{prefix}/mlp/c_proj/w"] = layer.c_mlp_proj_w
            model.tensors[f"{prefix}/mlp/c_proj/b"] = layer.c_mlp_proj_b

        buffer_w = session.alloc(ctx_w)

        print(
            f"gpt2_model_load: backend buffer size = "
            f"{ggml.backend_buffer_get_size(buffer_w) / 1024 / 1024:.2f} MB"
        )

        model.hparams.n_ctx = n_ctx

        ctx_kv = session.context(n_tensors=2)

        n_mem = n_layer * n_ctx
        n_elements = n_embd * n_mem

        model.memory_k = ctx_kv.new_tensor(ggml.Type.F32, n_elements)
        model.memory_v = ctx_kv.new_tensor(ggml.Type.F32, n_elements)

        buffer_kv = session.alloc(ctx_kv)

        memory_size = ggml.backend_buffer_get_size(buffer_kv)
        print(
            f"gpt2_model_load: memory size = {memory_size / 1024 / 1024:.2f} MB, "
            f"n_mem = {n_mem}"
        )

        total_size = 0
        has_lm_head = False

        while True:
            header = f.read(12)
            if len(header) < 12:
                break

            n_dims, length, ttype = struct.unpack("<iii", header)

            ne = [1, 1]
            for d in range(n_dims):
                (ne[d],) = struct.unpack("<i", f.read(4))

            name = f.read(length).decode("utf-8")

            if name not in model.tensors:
                raise ValueError(f"unknown tensor '{name}' in model file")

            tensor = model.tensors[name]
            ggml.set_name(tensor, name)

            n_bytes = ggml.nbytes(tensor)
            data = np.frombuffer(f.read(n_bytes), dtype=np.uint8)
            ggml.backend_tensor_set(tensor, data, 0, n_bytes)

            if name == "model/wte" and not has_lm_head:
                model.lm_head = tensor

            if name == "model/lm_head":
                has_lm_head = True

            total_size += n_bytes

        print(f"gpt2_model_load: model size  = {total_size / 1024 / 1024:.2f} MB")

    return token_to_id, id_to_token


# ============================================================================
# Graph building
# ============================================================================

def gpt2_graph(model: GPT2Model, n_past: int, n_tokens: int):
    """Build the GPT-2 computation graph."""
    N = n_tokens
    hp = model.hparams
    n_embd = hp.n_embd
    n_layer = hp.n_layer
    n_ctx = hp.n_ctx
    n_head = hp.n_head

    ctx = Context.for_graph(max_nodes=GPT2_MAX_NODES, grads=False)

    gf = ctx.new_graph(max_nodes=GPT2_MAX_NODES, grads=False)

    embd = ctx.new_tensor(ggml.Type.I32, N, name="embd")
    ggml.set_input(embd)

    position = ctx.new_tensor(ggml.Type.I32, N, name="position")
    ggml.set_input(position)

    # wte + wpe
    inpL = ggml.add(
        ctx.raw,
        ggml.get_rows(ctx.raw, model.wte, embd),
        ggml.get_rows(ctx.raw, model.wpe, position),
    )

    for il in range(n_layer):
        layer = model.layers[il]

        cur = ggml.norm(ctx.raw, inpL, hp.eps)
        cur = ggml.add(ctx.raw, ggml.mul(ctx.raw, cur, layer.ln_1_g), layer.ln_1_b)

        cur = ggml.mul_mat(ctx.raw, layer.c_attn_attn_w, cur)
        cur = ggml.add(ctx.raw, cur, layer.c_attn_attn_b)

        nb1 = ggml.tensor_nb(cur, 1)
        Qcur = ggml.view_2d(ctx.raw, cur, n_embd, N, nb1, 0 * 4 * n_embd)
        Kcur = ggml.view_2d(ctx.raw, cur, n_embd, N, nb1, 1 * 4 * n_embd)
        Vcur = ggml.view_2d(ctx.raw, cur, n_embd, N, nb1, 2 * 4 * n_embd)

        if N >= 1:
            k = ggml.view_1d(
                ctx.raw, model.memory_k, N * n_embd,
                ggml.element_size(model.memory_k) * n_embd * (il * n_ctx + n_past),
            )
            v = ggml.view_1d(
                ctx.raw, model.memory_v, N * n_embd,
                ggml.element_size(model.memory_v) * n_embd * (il * n_ctx + n_past),
            )
            ggml.build_forward_expand(gf, ggml.cpy(ctx.raw, Kcur, k))
            ggml.build_forward_expand(gf, ggml.cpy(ctx.raw, Vcur, v))

        Q = ggml.permute(
            ctx.raw,
            ggml.cont_3d(ctx.raw, Qcur, n_embd // n_head, n_head, N),
            0, 2, 1, 3,
        )

        K = ggml.permute(
            ctx.raw,
            ggml.reshape_3d(
                ctx.raw,
                ggml.view_1d(
                    ctx.raw, model.memory_k, (n_past + N) * n_embd,
                    il * n_ctx * ggml.element_size(model.memory_k) * n_embd,
                ),
                n_embd // n_head, n_head, n_past + N,
            ),
            0, 2, 1, 3,
        )

        KQ = ggml.mul_mat(ctx.raw, K, Q)
        KQ_scaled = ggml.scale(ctx.raw, KQ, 1.0 / math.sqrt(n_embd / n_head))
        KQ_masked = ggml.diag_mask_inf(ctx.raw, KQ_scaled, n_past)
        KQ_soft_max = ggml.soft_max(ctx.raw, KQ_masked)

        V_trans = ggml.cont_3d(
            ctx.raw,
            ggml.permute(
                ctx.raw,
                ggml.reshape_3d(
                    ctx.raw,
                    ggml.view_1d(
                        ctx.raw, model.memory_v, (n_past + N) * n_embd,
                        il * n_ctx * ggml.element_size(model.memory_v) * n_embd,
                    ),
                    n_embd // n_head, n_head, n_past + N,
                ),
                1, 2, 0, 3,
            ),
            n_past + N, n_embd // n_head, n_head,
        )

        KQV = ggml.mul_mat(ctx.raw, V_trans, KQ_soft_max)
        KQV_merged = ggml.permute(ctx.raw, KQV, 0, 2, 1, 3)
        cur = ggml.cont_2d(ctx.raw, KQV_merged, n_embd, N)

        cur = ggml.mul_mat(ctx.raw, layer.c_attn_proj_w, cur)
        cur = ggml.add(ctx.raw, cur, layer.c_attn_proj_b)

        cur = ggml.add(ctx.raw, cur, inpL)

        inpFF = cur

        cur = ggml.norm(ctx.raw, inpFF, hp.eps)
        cur = ggml.add(ctx.raw, ggml.mul(ctx.raw, cur, layer.ln_2_g), layer.ln_2_b)

        cur = ggml.mul_mat(ctx.raw, layer.c_mlp_fc_w, cur)
        cur = ggml.add(ctx.raw, cur, layer.c_mlp_fc_b)

        cur = ggml.gelu(ctx.raw, cur)

        cur = ggml.mul_mat(ctx.raw, layer.c_mlp_proj_w, cur)
        cur = ggml.add(ctx.raw, cur, layer.c_mlp_proj_b)

        inpL = ggml.add(ctx.raw, cur, inpFF)

    inpL = ggml.norm(ctx.raw, inpL, hp.eps)
    inpL = ggml.add(ctx.raw, ggml.mul(ctx.raw, inpL, model.ln_f_g), model.ln_f_b)

    inpL = ggml.mul_mat(ctx.raw, model.lm_head, inpL)
    ggml.set_name(inpL, "logits")
    ggml.set_output(inpL)

    ggml.build_forward_expand(gf, inpL)

    return gf, ctx


# ============================================================================
# Evaluation
# ============================================================================

def gpt2_eval(
    model: GPT2Model,
    session: ggbond.Session,
    n_threads: int,
    n_past: int,
    embd_inp: list[int],
) -> np.ndarray:
    """Evaluate the transformer and return logits for the last token."""
    N = len(embd_inp)
    n_vocab = model.hparams.n_vocab

    gf, graph_ctx = gpt2_graph(model, n_past, N)

    session.runner.compute(gf, inputs={
        "embd": np.array(embd_inp, dtype=np.int32),
        "position": np.arange(n_past, n_past + N, dtype=np.int32),
    })

    logits = session.get_slice(gf, "logits",
                               offset=n_vocab * (N - 1) * 4, count=n_vocab)

    return logits


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GPT-2 inference with ggbond Session API")
    parser.add_argument(
        "-m", "--model", type=str,
        default="models/gpt-2-117M/ggml-model.bin",
        help="Path to model file",
    )
    parser.add_argument("-p", "--prompt", type=str, default="Hello", help="Input prompt")
    parser.add_argument("-n", "--n-predict", type=int, default=200, help="Number of tokens to predict")
    parser.add_argument("--n-ctx", type=int, default=1024, help="Context size")
    parser.add_argument("--n-batch", type=int, default=32, help="Batch size for prompt processing")
    parser.add_argument("-t", "--threads", type=int, default=4, help="Number of threads")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="Number of layers to offload to GPU")
    args = parser.parse_args()

    ggml.time_init()
    ggml.log_set_default()
    t_main_start_us = ggml.time_us()

    seed = args.seed if args.seed >= 0 else random.randint(0, 2**31)
    print(f"main: seed = {seed}")
    rng = random.Random(seed)

    t_start_us = ggml.time_us()
    model = GPT2Model()

    with ggbond.Session("cpu", n_threads=args.threads) as s:
        token_to_id, id_to_token = gpt2_model_load(
            args.model, model, s, args.n_ctx, args.n_gpu_layers
        )
        t_load_us = ggml.time_us() - t_start_us

        # Reserve memory for worst case
        n_tokens = min(model.hparams.n_ctx, args.n_batch)
        n_past_worst = model.hparams.n_ctx - n_tokens
        gf_worst, ctx_worst = gpt2_graph(model, n_past_worst, n_tokens)
        s.reserve(gf_worst)
        del ctx_worst

        mem_size = s.runner.buffer_size
        print(
            f"main: compute buffer size: {mem_size / 1024 / 1024:.2f} MB",
            file=sys.stderr,
        )

        # Tokenize prompt
        embd_inp = gpt_tokenize(token_to_id, args.prompt)
        n_predict = min(args.n_predict, model.hparams.n_ctx - len(embd_inp))

        print(f"main: prompt: '{args.prompt}'")
        print(
            f"main: number of tokens in prompt = {len(embd_inp)}, "
            f"first 8 tokens: {embd_inp[:8]}"
        )
        print()

        n_past = 0
        t_sample_us = 0
        t_predict_us = 0

        logits = None
        embd: list[int] = []

        i = 0
        while i < len(embd_inp) + n_predict:
            if len(embd) > 0:
                t_start = ggml.time_us()
                logits = gpt2_eval(
                    model, s, args.threads, n_past, embd
                )
                t_predict_us += ggml.time_us() - t_start

            n_past += len(embd)
            embd.clear()

            if i >= len(embd_inp):
                t_start_sample = ggml.time_us()
                token_id = gpt_sample_top_k_top_p(
                    logits, args.top_k, args.top_p, args.temp, rng
                )
                t_sample_us += ggml.time_us() - t_start_sample
                embd.append(token_id)
            else:
                while i < len(embd_inp):
                    embd.append(embd_inp[i])
                    i += 1
                    if len(embd) >= args.n_batch:
                        break
                i -= 1

            for token_id in embd:
                print(id_to_token.get(token_id, ""), end="", flush=True)

            if embd[-1] == 50256:
                break

            i += 1

        # Report timing
        t_main_end_us = ggml.time_us()
        print("\n")
        print(f"main:     load time = {t_load_us / 1000:.2f} ms")
        print(f"main:   sample time = {t_sample_us / 1000:.2f} ms")
        print(
            f"main:  predict time = {t_predict_us / 1000:.2f} ms"
            f" / {t_predict_us / 1000 / max(n_past, 1):.2f} ms per token"
        )
        print(f"main:    total time = {(t_main_end_us - t_main_start_us) / 1000:.2f} ms")


if __name__ == "__main__":
    main()

"""
GPT-2 inference using ggbond Session API with GGUF format.

Usage:
    python examples/gpt2.py -m models/gpt2-117M.gguf -p "Hello, world"

Download/Convert model:
    # Use existing GGUF conversion tools (e.g., llama.cpp)
"""

import argparse
import math
import re
import sys
import random
from dataclasses import dataclass

import numpy as np

import ggbond
from ggbond import ggml
from ggbond.graph import GAllocr, Graph

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

@dataclass
class GPT2HParams:
    n_vocab: int = 50257
    n_ctx: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    eps: float = 1e-5


@dataclass
class GPT2Layer:
    ln_1_g: ggbond.Tensor
    ln_1_b: ggbond.Tensor
    ln_2_g: ggbond.Tensor
    ln_2_b: ggbond.Tensor
    c_attn_attn_w: ggbond.Tensor
    c_attn_attn_b: ggbond.Tensor
    c_attn_proj_w: ggbond.Tensor
    c_attn_proj_b: ggbond.Tensor
    c_mlp_fc_w: ggbond.Tensor
    c_mlp_fc_b: ggbond.Tensor
    c_mlp_proj_w: ggbond.Tensor
    c_mlp_proj_b: ggbond.Tensor


@dataclass
class GPT2Model:
    hparams: GPT2HParams
    ln_f_g: ggbond.Tensor
    ln_f_b: ggbond.Tensor
    wte: ggbond.Tensor
    wpe: ggbond.Tensor
    lm_head: ggbond.Tensor
    layers: list[GPT2Layer]
    memory_k: ggbond.Tensor
    memory_v: ggbond.Tensor


# ============================================================================
# Model loading
# ============================================================================

def gpt2_model_load(
    fname: str, session: ggbond.Session, n_ctx: int
) -> tuple[GPT2Model, dict[str, int], dict[int, str]]:
    """Load GPT-2 model from GGUF file. Returns (model, token_to_id, id_to_token)."""
    print(f"gpt2_model_load: loading model from '{fname}'")

    # Initialize GGUF context to read metadata
    gguf_ctx, _ = ggml.gguf_init_from_file(fname, no_alloc=True)

    # Read hyperparameters from metadata
    # GGUF stores these as uint32 (type=4) or float32 (type=6)
    def get_u32(key: str) -> int | None:
        key_id = ggml.gguf_find_key(gguf_ctx, key)
        if key_id < 0:
            return None
        return ggml.gguf_get_val_u32(gguf_ctx, key_id)

    def get_f32(key: str) -> float | None:
        key_id = ggml.gguf_find_key(gguf_ctx, key)
        if key_id < 0:
            return None
        return ggml.gguf_get_val_f32(gguf_ctx, key_id)

    n_vocab = get_u32("gpt2.vocab_size") or 50257
    n_embd = get_u32("gpt2.embedding_length") or 768
    n_head = get_u32("gpt2.attention.head_count") or 12
    n_layer = get_u32("gpt2.block_count") or 12
    eps = get_f32("gpt2.attention.layer_norm_epsilon") or 1e-5

    print(f"gpt2_model_load: n_vocab = {n_vocab}")
    print(f"gpt2_model_load: n_embd  = {n_embd}")
    print(f"gpt2_model_load: n_head  = {n_head}")
    print(f"gpt2_model_load: n_layer = {n_layer}")

    # Read vocabulary
    token_key_id = ggml.gguf_find_key(gguf_ctx, "tokenizer.ggml.tokens")
    if token_key_id >= 0:
        n_tokens = ggml.gguf_get_arr_n(gguf_ctx, token_key_id)
        token_to_id: dict[str, int] = {}
        id_to_token: dict[int, str] = {}
        for i in range(n_tokens):
            token = ggml.gguf_get_arr_str(gguf_ctx, token_key_id, i)
            token_to_id[token] = i
            id_to_token[i] = token
    else:
        raise ValueError("GGUF file missing tokenizer.ggml.tokens metadata")

    # Free GGUF context (metadata read complete)
    ggml.gguf_free(gguf_ctx)

    # Load weight tensors (weights loaded directly to backend)
    weights = session.load_gguf(fname)
    print(f"gpt2_model_load: loaded {len(weights)} tensors")

    hp = GPT2HParams(
        n_vocab=n_vocab,
        n_ctx=n_ctx,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        eps=eps,
    )

    ln_f_g = weights["output_norm.weight"]
    ln_f_b = weights["output_norm.bias"]
    wte = weights["token_embd.weight"]
    wpe = weights["position_embd.weight"]
    lm_head = weights["output.weight"]

    layers: list[GPT2Layer] = []
    for i in range(n_layer):
        layer = GPT2Layer(
            ln_1_g=weights[f"blk.{i}.attn_norm.weight"],
            ln_1_b=weights[f"blk.{i}.attn_norm.bias"],
            ln_2_g=weights[f"blk.{i}.ffn_norm.weight"],
            ln_2_b=weights[f"blk.{i}.ffn_norm.bias"],
            c_attn_attn_w=weights[f"blk.{i}.attn_qkv.weight"],
            c_attn_attn_b=weights[f"blk.{i}.attn_qkv.bias"],
            c_attn_proj_w=weights[f"blk.{i}.attn_output.weight"],
            c_attn_proj_b=weights[f"blk.{i}.attn_output.bias"],
            c_mlp_fc_w=weights[f"blk.{i}.ffn_up.weight"],
            c_mlp_fc_b=weights[f"blk.{i}.ffn_up.bias"],
            c_mlp_proj_w=weights[f"blk.{i}.ffn_down.weight"],
            c_mlp_proj_b=weights[f"blk.{i}.ffn_down.bias"],
        )
        layers.append(layer)

    # Allocate KV cache
    n_mem = n_layer * n_ctx
    n_elements = n_embd * n_mem
    memory_k = session.empty(ggml.Type.F32, n_elements, name="memory_k")
    memory_v = session.empty(ggml.Type.F32, n_elements, name="memory_v")

    model = GPT2Model(
        hparams=hp,
        ln_f_g=ln_f_g,
        ln_f_b=ln_f_b,
        wte=wte,
        wpe=wpe,
        lm_head=lm_head,
        layers=layers,
        memory_k=memory_k,
        memory_v=memory_v,
    )

    return model, token_to_id, id_to_token


# ============================================================================
# Graph building
# ============================================================================

def _t(tensor):
    """Extract the underlying ggml.Tensor pointer from a high-level Tensor."""
    return tensor._ggml_tensor


def gpt2_graph(model: GPT2Model, n_past: int, n_tokens: int):
    """Build the GPT-2 computation graph."""
    N = n_tokens
    hp = model.hparams
    n_embd = hp.n_embd
    n_layer = hp.n_layer
    n_ctx = hp.n_ctx
    n_head = hp.n_head

    g = Graph(max_nodes=GPT2_MAX_NODES)

    embd = g.new_tensor(ggml.Type.I32, N, name="embd")
    ggml.set_input(embd)

    position = g.new_tensor(ggml.Type.I32, N, name="position")
    ggml.set_input(position)

    # wte + wpe
    inpL = g.add(
        g.get_rows(_t(model.wte), embd),
        g.get_rows(_t(model.wpe), position),
    )

    for il in range(n_layer):
        layer = model.layers[il]

        cur = g.norm(inpL, hp.eps)
        cur = g.add(g.mul(cur, _t(layer.ln_1_g)), _t(layer.ln_1_b))

        cur = g.mul_mat(_t(layer.c_attn_attn_w), cur)
        cur = g.add(cur, _t(layer.c_attn_attn_b))

        nb1 = ggml.tensor_nb(cur, 1)
        Qcur = g.view_2d(cur, n_embd, N, nb1, 0 * 4 * n_embd)
        Kcur = g.view_2d(cur, n_embd, N, nb1, 1 * 4 * n_embd)
        Vcur = g.view_2d(cur, n_embd, N, nb1, 2 * 4 * n_embd)

        if N >= 1:
            k = g.view_1d(
                _t(model.memory_k), N * n_embd,
                ggml.element_size(_t(model.memory_k)) * n_embd * (il * n_ctx + n_past),
            )
            v = g.view_1d(
                _t(model.memory_v), N * n_embd,
                ggml.element_size(_t(model.memory_v)) * n_embd * (il * n_ctx + n_past),
            )
            g.build_forward(g.cpy(Kcur, k))
            g.build_forward(g.cpy(Vcur, v))

        Q = g.permute(
            g.cont_3d(Qcur, n_embd // n_head, n_head, N),
            0, 2, 1, 3,
        )

        K = g.permute(
            g.reshape(
                g.view_1d(
                    _t(model.memory_k), (n_past + N) * n_embd,
                    il * n_ctx * ggml.element_size(_t(model.memory_k)) * n_embd,
                ),
                n_embd // n_head, n_head, n_past + N,
            ),
            0, 2, 1, 3,
        )

        KQ = g.mul_mat(K, Q)
        KQ_scaled = g.scale(KQ, 1.0 / math.sqrt(n_embd / n_head))
        KQ_masked = g.diag_mask_inf(KQ_scaled, n_past)
        KQ_soft_max = g.soft_max(KQ_masked)

        V_trans = g.cont_3d(
            g.permute(
                g.reshape(
                    g.view_1d(
                        _t(model.memory_v), (n_past + N) * n_embd,
                        il * n_ctx * ggml.element_size(_t(model.memory_v)) * n_embd,
                    ),
                    n_embd // n_head, n_head, n_past + N,
                ),
                1, 2, 0, 3,
            ),
            n_past + N, n_embd // n_head, n_head,
        )

        KQV = g.mul_mat(V_trans, KQ_soft_max)
        KQV_merged = g.permute(KQV, 0, 2, 1, 3)
        cur = g.cont_2d(KQV_merged, n_embd, N)

        cur = g.mul_mat(_t(layer.c_attn_proj_w), cur)
        cur = g.add(cur, _t(layer.c_attn_proj_b))

        cur = g.add(cur, inpL)

        inpFF = cur

        cur = g.norm(inpFF, hp.eps)
        cur = g.add(g.mul(cur, _t(layer.ln_2_g)), _t(layer.ln_2_b))

        cur = g.mul_mat(_t(layer.c_mlp_fc_w), cur)
        cur = g.add(cur, _t(layer.c_mlp_fc_b))

        cur = g.gelu(cur)

        cur = g.mul_mat(_t(layer.c_mlp_proj_w), cur)
        cur = g.add(cur, _t(layer.c_mlp_proj_b))

        inpL = g.add(cur, inpFF)

    inpL = g.norm(inpL, hp.eps)
    inpL = g.add(g.mul(inpL, _t(model.ln_f_g)), _t(model.ln_f_b))

    inpL = g.mul_mat(_t(model.lm_head), inpL)
    ggml.set_name(inpL, "logits")
    ggml.set_output(inpL)

    g.build_forward(inpL)

    return g


# ============================================================================
# Evaluation
# ============================================================================

def gpt2_eval(
    model: GPT2Model,
    session: ggbond.Session,
    n_past: int,
    embd_inp: list[int],
) -> np.ndarray:
    """Evaluate the transformer and return logits for the last token."""
    N = len(embd_inp)
    n_vocab = model.hparams.n_vocab

    g = gpt2_graph(model, n_past, N)

    allocr = GAllocr(session.backend)
    allocr.reserve(g.raw)
    allocr.alloc(g.raw)

    embd_data = np.array(embd_inp, dtype=np.int32)
    pos_data = np.arange(n_past, n_past + N, dtype=np.int32)

    session.backend.tensor_set(ggml.graph_get_tensor(g.raw, "embd"), embd_data)
    session.backend.tensor_set(ggml.graph_get_tensor(g.raw, "position"), pos_data)

    session.backend.compute(g.raw)

    logits_t = ggml.graph_get_tensor(g.raw, "logits")
    logits = session.backend.tensor_get_slice(
        logits_t, n_vocab * (N - 1) * 4, n_vocab,
    )

    allocr.close()
    g.close()

    return logits


# ============================================================================
# Token decoding
# ============================================================================

def decode_token(token: str) -> str:
    """Decode GGUF token string (replaces special Unicode chars with actual chars)."""
    # GGUF uses special Unicode chars for whitespace:
    # 'Ġ' (U+0120) -> space
    # 'Ċ' (U+010A) -> newline
    # 'ĉ' (U+0109) -> tab (possibly)
    return token.replace('Ġ', ' ').replace('Ċ', '\n').replace('ĉ', '\t')


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GPT-2 inference with ggbond Session API")
    parser.add_argument(
        "-m", "--model", type=str,
        default="models/gpt2-117M.gguf",
        help="Path to GGUF model file",
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
    args = parser.parse_args()

    t_main_start_us = ggml.time_us()

    seed = args.seed if args.seed >= 0 else random.randint(0, 2**31)
    print(f"main: seed = {seed}")
    rng = random.Random(seed)

    t_start_us = ggml.time_us()

    with ggbond.Session("cpu", n_threads=args.threads) as s:
        model, token_to_id, id_to_token = gpt2_model_load(args.model, s, args.n_ctx)
        t_load_us = ggml.time_us() - t_start_us

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
                logits = gpt2_eval(model, s, n_past, embd)
                t_predict_us += ggml.time_us() - t_start

            n_past += len(embd)
            embd.clear()

            if i >= len(embd_inp):
                assert logits is not None
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
                print(decode_token(id_to_token.get(token_id, "")), end="", flush=True)

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

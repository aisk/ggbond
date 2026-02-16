"""
GPT-2 inference using ggbond Session API.

Ported from vendor/ggml/examples/gpt-2/main-backend.cpp

Usage:
    python examples/gpt2.py -m models/gpt-2-117M/ggml-model-f32.bin -p "Hello, world"

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


# ============================================================================
# Model loading
# ============================================================================

def gpt2_model_load(
    fname: str, model: GPT2Model, session: ggbond.Session, n_ctx: int
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

        n_embd = hp.n_embd
        n_layer = hp.n_layer

        # Build a name→(dtype, shape) map for all expected tensors
        tensor_info: dict[str, tuple] = {}
        tensor_info["model/ln_f/g"] = (ggml.Type.F32, (n_embd,))
        tensor_info["model/ln_f/b"] = (ggml.Type.F32, (n_embd,))
        tensor_info["model/wte"] = (wtype, (n_embd, hp.n_vocab))
        tensor_info["model/wpe"] = (ggml.Type.F32, (n_embd, hp.n_ctx))
        tensor_info["model/lm_head"] = (wtype, (n_embd, hp.n_vocab))

        for i in range(n_layer):
            prefix = f"model/h{i}"
            tensor_info[f"{prefix}/ln_1/g"] = (ggml.Type.F32, (n_embd,))
            tensor_info[f"{prefix}/ln_1/b"] = (ggml.Type.F32, (n_embd,))
            tensor_info[f"{prefix}/ln_2/g"] = (ggml.Type.F32, (n_embd,))
            tensor_info[f"{prefix}/ln_2/b"] = (ggml.Type.F32, (n_embd,))
            tensor_info[f"{prefix}/attn/c_attn/w"] = (wtype, (n_embd, 3 * n_embd))
            tensor_info[f"{prefix}/attn/c_attn/b"] = (ggml.Type.F32, (3 * n_embd,))
            tensor_info[f"{prefix}/attn/c_proj/w"] = (wtype, (n_embd, n_embd))
            tensor_info[f"{prefix}/attn/c_proj/b"] = (ggml.Type.F32, (n_embd,))
            tensor_info[f"{prefix}/mlp/c_fc/w"] = (wtype, (n_embd, 4 * n_embd))
            tensor_info[f"{prefix}/mlp/c_fc/b"] = (ggml.Type.F32, (4 * n_embd,))
            tensor_info[f"{prefix}/mlp/c_proj/w"] = (wtype, (4 * n_embd, n_embd))
            tensor_info[f"{prefix}/mlp/c_proj/b"] = (ggml.Type.F32, (n_embd,))

        # Read weights from file and create tensors via session.tensor()
        tensors: dict[str, ggbond.Tensor] = {}
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

            if name not in tensor_info:
                raise ValueError(f"unknown tensor '{name}' in model file")

            dtype, shape = tensor_info[name]

            # Calculate byte count from ggml type info
            # For quantized types we read raw bytes; for native types use numpy
            type_size = ggml.type_size(ggml.Type(ttype))
            block_size = ggml.blck_size(ggml.Type(ttype))
            n_elements = 1
            for d in range(n_dims):
                n_elements *= ne[d]
            n_bytes = n_elements * type_size // block_size

            raw_data = f.read(n_bytes)
            t = session.tensor(raw_data, dtype=dtype, shape=shape, name=name)
            tensors[name] = t

            if name == "model/wte" and not has_lm_head:
                tensors["model/lm_head"] = t

            if name == "model/lm_head":
                has_lm_head = True

            total_size += n_bytes

        print(f"gpt2_model_load: model size  = {total_size / 1024 / 1024:.2f} MB")

        # Override n_ctx
        model.hparams.n_ctx = n_ctx

        # Assign model weight references
        model.ln_f_g = tensors["model/ln_f/g"]
        model.ln_f_b = tensors["model/ln_f/b"]
        model.wte = tensors["model/wte"]
        model.wpe = tensors["model/wpe"]
        model.lm_head = tensors["model/lm_head"]

        model.layers = []
        for i in range(n_layer):
            layer = GPT2Layer()
            prefix = f"model/h{i}"
            layer.ln_1_g = tensors[f"{prefix}/ln_1/g"]
            layer.ln_1_b = tensors[f"{prefix}/ln_1/b"]
            layer.ln_2_g = tensors[f"{prefix}/ln_2/g"]
            layer.ln_2_b = tensors[f"{prefix}/ln_2/b"]
            layer.c_attn_attn_w = tensors[f"{prefix}/attn/c_attn/w"]
            layer.c_attn_attn_b = tensors[f"{prefix}/attn/c_attn/b"]
            layer.c_attn_proj_w = tensors[f"{prefix}/attn/c_proj/w"]
            layer.c_attn_proj_b = tensors[f"{prefix}/attn/c_proj/b"]
            layer.c_mlp_fc_w = tensors[f"{prefix}/mlp/c_fc/w"]
            layer.c_mlp_fc_b = tensors[f"{prefix}/mlp/c_fc/b"]
            layer.c_mlp_proj_w = tensors[f"{prefix}/mlp/c_proj/w"]
            layer.c_mlp_proj_b = tensors[f"{prefix}/mlp/c_proj/b"]
            model.layers.append(layer)

        # Allocate KV cache
        n_mem = n_layer * n_ctx
        n_elements = n_embd * n_mem
        model.memory_k = session.empty(ggml.Type.F32, n_elements, name="memory_k")
        model.memory_v = session.empty(ggml.Type.F32, n_elements, name="memory_v")

    return token_to_id, id_to_token


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
            args.model, model, s, args.n_ctx
        )
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

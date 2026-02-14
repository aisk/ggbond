"""
GPT-2 inference using ggbond Python bindings.

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
import platform
import re
import struct
import sys
import random

import numpy as np

from ggbond import ggml

GPT2_MAX_NODES = 4096


# ============================================================================
# Tokenizer
# ============================================================================

def gpt_tokenize(vocab_token_to_id: dict[str, int], text: str) -> list[int]:
    """BPE tokenizer matching the C++ gpt_tokenize implementation."""
    # Regex pattern from common.cpp
    pat = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""",
        re.UNICODE,
    )
    words = pat.findall(text)

    tokens: list[int] = []
    for word in words:
        # Greedy longest-match tokenization
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
                # Skip unknown character
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

    # Temperature scaling
    if temp <= 0:
        return int(np.argmax(logits))

    scaled = logits / temp

    # Top-k: get indices of top_k largest values
    if top_k > 0:
        top_k = min(top_k, n_vocab)
        indices = np.argpartition(scaled, -top_k)[-top_k:]
    else:
        indices = np.arange(n_vocab)

    top_logits = scaled[indices]

    # Softmax
    max_logit = np.max(top_logits)
    exps = np.exp(top_logits - max_logit)
    probs = exps / np.sum(exps)

    # Sort by probability descending for top-p
    sorted_order = np.argsort(-probs)
    sorted_probs = probs[sorted_order]
    sorted_indices = indices[sorted_order]

    # Top-p filtering
    if top_p < 1.0:
        cumsum = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumsum, top_p) + 1
        sorted_probs = sorted_probs[:cutoff]
        sorted_indices = sorted_indices[:cutoff]
        # Re-normalize
        sorted_probs = sorted_probs / np.sum(sorted_probs)

    # Sample
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
        self.ctx_w = None
        self.ctx_kv = None
        self.backend = None
        self.buffer_w = None
        self.buffer_kv = None
        self.tensors: dict[str, object] = {}


# ============================================================================
# Model loading
# ============================================================================

def gpt2_model_load(
    fname: str, model: GPT2Model, n_ctx: int, n_gpu_layers: int
) -> dict[int, str]:
    """Load GPT-2 model from binary file. Returns id_to_token vocab dict."""
    print(f"gpt2_model_load: loading model from '{fname}'")

    with open(fname, "rb") as f:
        # Verify magic
        (magic,) = struct.unpack("<I", f.read(4))
        if magic != ggml.FILE_MAGIC:
            raise ValueError(
                f"invalid model file '{fname}' (bad magic: {magic:#x}, "
                f"expected {ggml.FILE_MAGIC:#x})"
            )

        # Load hparams
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

        # Load vocab
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

        # Determine weight type
        wtype = ggml.Type(ggml.ftype_to_ggml_type(hp.ftype))

        # Create context for weight tensors (no_alloc)
        n_tensors = 2 + 6 + 12 * hp.n_layer
        ctx_w = ggml.context_init(
            ggml.tensor_overhead() * n_tensors, no_alloc=True
        )
        model.ctx_w = ctx_w

        # Initialize backend
        if n_gpu_layers > 0 and platform.system() == "Darwin":
            print("gpt2_model_load: using Metal backend", file=sys.stderr)
            model.backend = ggml.backend_metal_init()

        if not model.backend:
            print("gpt2_model_load: using CPU backend", file=sys.stderr)
            model.backend = ggml.backend_cpu_init()

        if not model.backend:
            raise RuntimeError("failed to initialize backend")

        # Create tensors
        n_embd = hp.n_embd
        n_layer = hp.n_layer

        model.ln_f_g = ggml.new_tensor_1d(ctx_w, ggml.Type.F32, n_embd)
        model.ln_f_b = ggml.new_tensor_1d(ctx_w, ggml.Type.F32, n_embd)

        model.wte = ggml.new_tensor_2d(ctx_w, wtype, n_embd, hp.n_vocab)
        model.wpe = ggml.new_tensor_2d(ctx_w, ggml.Type.F32, n_embd, hp.n_ctx)
        model.lm_head = ggml.new_tensor_2d(ctx_w, wtype, n_embd, hp.n_vocab)

        model.tensors["model/ln_f/g"] = model.ln_f_g
        model.tensors["model/ln_f/b"] = model.ln_f_b
        model.tensors["model/wte"] = model.wte
        model.tensors["model/wpe"] = model.wpe
        model.tensors["model/lm_head"] = model.lm_head

        model.layers = []
        for i in range(n_layer):
            layer = GPT2Layer()

            layer.ln_1_g = ggml.new_tensor_1d(ctx_w, ggml.Type.F32, n_embd)
            layer.ln_1_b = ggml.new_tensor_1d(ctx_w, ggml.Type.F32, n_embd)
            layer.ln_2_g = ggml.new_tensor_1d(ctx_w, ggml.Type.F32, n_embd)
            layer.ln_2_b = ggml.new_tensor_1d(ctx_w, ggml.Type.F32, n_embd)

            layer.c_attn_attn_w = ggml.new_tensor_2d(
                ctx_w, wtype, n_embd, 3 * n_embd
            )
            layer.c_attn_attn_b = ggml.new_tensor_1d(
                ctx_w, ggml.Type.F32, 3 * n_embd
            )
            layer.c_attn_proj_w = ggml.new_tensor_2d(
                ctx_w, wtype, n_embd, n_embd
            )
            layer.c_attn_proj_b = ggml.new_tensor_1d(
                ctx_w, ggml.Type.F32, n_embd
            )

            layer.c_mlp_fc_w = ggml.new_tensor_2d(
                ctx_w, wtype, n_embd, 4 * n_embd
            )
            layer.c_mlp_fc_b = ggml.new_tensor_1d(
                ctx_w, ggml.Type.F32, 4 * n_embd
            )
            layer.c_mlp_proj_w = ggml.new_tensor_2d(
                ctx_w, wtype, 4 * n_embd, n_embd
            )
            layer.c_mlp_proj_b = ggml.new_tensor_1d(
                ctx_w, ggml.Type.F32, n_embd
            )

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

        # Allocate weight tensors in backend buffer
        model.buffer_w = ggml.backend_alloc_ctx_tensors(ctx_w, model.backend)

        print(
            f"gpt2_model_load: backend buffer size = "
            f"{ggml.backend_buffer_get_size(model.buffer_w) / 1024 / 1024:.2f} MB"
        )

        # Override training n_ctx with user-provided value
        model.hparams.n_ctx = n_ctx

        # Key + value memory
        n_tensors_kv = 2
        ctx_kv = ggml.context_init(
            ggml.tensor_overhead() * n_tensors_kv, no_alloc=True
        )
        model.ctx_kv = ctx_kv

        n_mem = n_layer * n_ctx
        n_elements = n_embd * n_mem

        model.memory_k = ggml.new_tensor_1d(ctx_kv, ggml.Type.F32, n_elements)
        model.memory_v = ggml.new_tensor_1d(ctx_kv, ggml.Type.F32, n_elements)

        model.buffer_kv = ggml.backend_alloc_ctx_tensors(
            ctx_kv, model.backend
        )

        memory_size = ggml.backend_buffer_get_size(model.buffer_kv)
        print(
            f"gpt2_model_load: memory size = {memory_size / 1024 / 1024:.2f} MB, "
            f"n_mem = {n_mem}"
        )

        # Load weights
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

            # Read tensor data
            data = np.frombuffer(f.read(n_bytes), dtype=np.uint8)
            ggml.backend_tensor_set(tensor, data, 0, n_bytes)

            # GPT-2 models share the WTE tensor as the LM head
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

    buf_size = (
        ggml.tensor_overhead() * GPT2_MAX_NODES
        + ggml.graph_overhead_custom(GPT2_MAX_NODES, False)
    )
    ctx = ggml.context_init(buf_size, no_alloc=True)

    gf = ggml.new_graph_custom(ctx, GPT2_MAX_NODES, False)

    embd = ggml.new_tensor_1d(ctx, ggml.Type.I32, N)
    ggml.set_name(embd, "embd")
    ggml.set_input(embd)

    position = ggml.new_tensor_1d(ctx, ggml.Type.I32, N)
    ggml.set_name(position, "position")
    ggml.set_input(position)

    # wte + wpe
    inpL = ggml.add(
        ctx,
        ggml.get_rows(ctx, model.wte, embd),
        ggml.get_rows(ctx, model.wpe, position),
    )

    for il in range(n_layer):
        layer = model.layers[il]

        # Layer norm 1
        cur = ggml.norm(ctx, inpL, hp.eps)
        cur = ggml.add(ctx, ggml.mul(ctx, cur, layer.ln_1_g), layer.ln_1_b)

        # Attention: cur = attn_w * cur + attn_b
        cur = ggml.mul_mat(ctx, layer.c_attn_attn_w, cur)
        cur = ggml.add(ctx, cur, layer.c_attn_attn_b)

        # Self-attention
        nb1 = ggml.tensor_nb(cur, 1)
        Qcur = ggml.view_2d(ctx, cur, n_embd, N, nb1, 0 * 4 * n_embd)
        Kcur = ggml.view_2d(ctx, cur, n_embd, N, nb1, 1 * 4 * n_embd)
        Vcur = ggml.view_2d(ctx, cur, n_embd, N, nb1, 2 * 4 * n_embd)

        # Store K, V to memory
        if N >= 1:
            k = ggml.view_1d(
                ctx,
                model.memory_k,
                N * n_embd,
                ggml.element_size(model.memory_k) * n_embd * (il * n_ctx + n_past),
            )
            v = ggml.view_1d(
                ctx,
                model.memory_v,
                N * n_embd,
                ggml.element_size(model.memory_v) * n_embd * (il * n_ctx + n_past),
            )
            ggml.build_forward_expand(gf, ggml.cpy(ctx, Kcur, k))
            ggml.build_forward_expand(gf, ggml.cpy(ctx, Vcur, v))

        # Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
        Q = ggml.permute(
            ctx,
            ggml.cont_3d(ctx, Qcur, n_embd // n_head, n_head, N),
            0, 2, 1, 3,
        )

        # K from memory
        K = ggml.permute(
            ctx,
            ggml.reshape_3d(
                ctx,
                ggml.view_1d(
                    ctx,
                    model.memory_k,
                    (n_past + N) * n_embd,
                    il * n_ctx * ggml.element_size(model.memory_k) * n_embd,
                ),
                n_embd // n_head,
                n_head,
                n_past + N,
            ),
            0, 2, 1, 3,
        )

        # K * Q
        KQ = ggml.mul_mat(ctx, K, Q)

        # KQ_scaled = KQ / sqrt(n_embd / n_head)
        KQ_scaled = ggml.scale(ctx, KQ, 1.0 / math.sqrt(n_embd / n_head))

        # Causal mask
        KQ_masked = ggml.diag_mask_inf(ctx, KQ_scaled, n_past)

        # Softmax
        KQ_soft_max = ggml.soft_max(ctx, KQ_masked)

        # V_trans from memory
        V_trans = ggml.cont_3d(
            ctx,
            ggml.permute(
                ctx,
                ggml.reshape_3d(
                    ctx,
                    ggml.view_1d(
                        ctx,
                        model.memory_v,
                        (n_past + N) * n_embd,
                        il * n_ctx * ggml.element_size(model.memory_v) * n_embd,
                    ),
                    n_embd // n_head,
                    n_head,
                    n_past + N,
                ),
                1, 2, 0, 3,
            ),
            n_past + N,
            n_embd // n_head,
            n_head,
        )

        # KQV = V_trans^T * KQ_soft_max
        KQV = ggml.mul_mat(ctx, V_trans, KQ_soft_max)

        # KQV_merged = KQV.permute(0, 2, 1, 3)
        KQV_merged = ggml.permute(ctx, KQV, 0, 2, 1, 3)

        # cur = KQV_merged.contiguous().view(n_embd, N)
        cur = ggml.cont_2d(ctx, KQV_merged, n_embd, N)

        # Projection
        cur = ggml.mul_mat(ctx, layer.c_attn_proj_w, cur)
        cur = ggml.add(ctx, cur, layer.c_attn_proj_b)

        # Residual
        cur = ggml.add(ctx, cur, inpL)

        inpFF = cur

        # Feed-forward: layer norm 2
        cur = ggml.norm(ctx, inpFF, hp.eps)
        cur = ggml.add(ctx, ggml.mul(ctx, cur, layer.ln_2_g), layer.ln_2_b)

        # FC
        cur = ggml.mul_mat(ctx, layer.c_mlp_fc_w, cur)
        cur = ggml.add(ctx, cur, layer.c_mlp_fc_b)

        # GELU
        cur = ggml.gelu(ctx, cur)

        # Projection
        cur = ggml.mul_mat(ctx, layer.c_mlp_proj_w, cur)
        cur = ggml.add(ctx, cur, layer.c_mlp_proj_b)

        # Residual
        inpL = ggml.add(ctx, cur, inpFF)

    # Final layer norm
    inpL = ggml.norm(ctx, inpL, hp.eps)
    inpL = ggml.add(ctx, ggml.mul(ctx, inpL, model.ln_f_g), model.ln_f_b)

    # LM head
    inpL = ggml.mul_mat(ctx, model.lm_head, inpL)
    ggml.set_name(inpL, "logits")
    ggml.set_output(inpL)

    ggml.build_forward_expand(gf, inpL)

    # Note: ctx is intentionally not freed here - the graph references it.
    # In the C++ version, ggml_free(ctx) is called but the graph data persists
    # because it was allocated in the context's memory buffer.
    # In Python, we return the ctx to keep it alive while the graph is in use.
    return gf, ctx


# ============================================================================
# Evaluation
# ============================================================================

def gpt2_eval(
    model: GPT2Model,
    allocr,
    n_threads: int,
    n_past: int,
    embd_inp: list[int],
) -> np.ndarray:
    """Evaluate the transformer and return logits for the last token."""
    N = len(embd_inp)
    n_vocab = model.hparams.n_vocab

    gf, graph_ctx = gpt2_graph(model, n_past, N)

    # Allocate graph tensors
    ggml.gallocr_alloc_graph(allocr, gf)

    # Set graph inputs
    embd = ggml.graph_get_tensor(gf, "embd")
    embd_data = np.array(embd_inp, dtype=np.int32)
    ggml.backend_tensor_set(embd, embd_data, 0, N * 4)

    position = ggml.graph_get_tensor(gf, "position")
    pos_data = np.array([n_past + i for i in range(N)], dtype=np.int32)
    ggml.backend_tensor_set(position, pos_data, 0, N * 4)

    # Set backend options
    if ggml.backend_is_cpu(model.backend):
        ggml.backend_cpu_set_n_threads(model.backend, n_threads)

    # Compute
    ggml.backend_graph_compute(model.backend, gf)

    # Get logits for the last token only
    logits_tensor = ggml.graph_get_tensor(gf, "logits")
    logits = np.empty(n_vocab, dtype=np.float32)
    ggml.backend_tensor_get(
        logits_tensor, logits, n_vocab * (N - 1) * 4, n_vocab * 4
    )

    ggml.context_free(graph_ctx)

    return logits


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GPT-2 inference with ggbond")
    parser.add_argument(
        "-m", "--model", type=str,
        default="models/gpt-2-117M/ggml-model.bin",
        help="Path to model file",
    )
    parser.add_argument(
        "-p", "--prompt", type=str, default="Hello",
        help="Input prompt",
    )
    parser.add_argument(
        "-n", "--n-predict", type=int, default=200,
        help="Number of tokens to predict",
    )
    parser.add_argument(
        "--n-ctx", type=int, default=1024,
        help="Context size",
    )
    parser.add_argument(
        "--n-batch", type=int, default=32,
        help="Batch size for prompt processing",
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=4,
        help="Number of threads",
    )
    parser.add_argument(
        "--top-k", type=int, default=40,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9,
        help="Top-p sampling",
    )
    parser.add_argument(
        "--temp", type=float, default=1.0,
        help="Temperature",
    )
    parser.add_argument(
        "--seed", type=int, default=-1,
        help="Random seed (-1 for random)",
    )
    parser.add_argument(
        "--n-gpu-layers", type=int, default=0,
        help="Number of layers to offload to GPU",
    )
    args = parser.parse_args()

    ggml.time_init()
    t_main_start_us = ggml.time_us()

    seed = args.seed if args.seed >= 0 else random.randint(0, 2**31)
    print(f"main: seed = {seed}")
    rng = random.Random(seed)

    # Load model
    t_start_us = ggml.time_us()
    model = GPT2Model()
    token_to_id, id_to_token = gpt2_model_load(
        args.model, model, args.n_ctx, args.n_gpu_layers
    )
    t_load_us = ggml.time_us() - t_start_us

    # Create graph allocator
    allocr = ggml.gallocr_new(
        ggml.backend_get_default_buffer_type(model.backend)
    )

    # Reserve memory for worst case
    n_tokens = min(model.hparams.n_ctx, args.n_batch)
    n_past_worst = model.hparams.n_ctx - n_tokens
    gf_worst, ctx_worst = gpt2_graph(model, n_past_worst, n_tokens)
    ggml.gallocr_reserve(allocr, gf_worst)
    ggml.context_free(ctx_worst)

    mem_size = ggml.gallocr_get_buffer_size(allocr, 0)
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
        # Predict
        if len(embd) > 0:
            t_start = ggml.time_us()
            logits = gpt2_eval(
                model, allocr, args.threads, n_past, embd
            )
            t_predict_us += ggml.time_us() - t_start

        n_past += len(embd)
        embd.clear()

        if i >= len(embd_inp):
            # Sample next token
            t_start_sample = ggml.time_us()
            token_id = gpt_sample_top_k_top_p(
                logits, args.top_k, args.top_p, args.temp, rng
            )
            t_sample_us += ggml.time_us() - t_start_sample

            embd.append(token_id)
        else:
            # Still processing the input prompt
            while i < len(embd_inp):
                embd.append(embd_inp[i])
                i += 1
                if len(embd) >= args.n_batch:
                    break
            # Adjust: the outer loop will increment i by 1
            i -= 1

        # Display text
        for token_id in embd:
            print(id_to_token.get(token_id, ""), end="", flush=True)

        # End of text token
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

    # Cleanup
    ggml.context_free(model.ctx_w)
    ggml.gallocr_free(allocr)
    ggml.backend_buffer_free(model.buffer_w)
    ggml.backend_buffer_free(model.buffer_kv)
    ggml.backend_free(model.backend)


if __name__ == "__main__":
    main()

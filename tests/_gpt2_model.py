"""A tiny synthetic GPT-2 in GGUF form, named as llama.cpp's converter names it."""

import numpy as np

from ggbond.ggml import GGUF, Context, F32

N_VOCAB = 4
N_CTX = 8
N_EMBD = 4
N_HEAD = 2
N_LAYER = 1

# In GPT-2's byte-level alphabet: "Ġ" is a space and "Ã©" the two bytes of "é".
TOKENS = ["ab", "Ġcd", "Ã©", "<|endoftext|>"]


def write_tiny_gpt2(path, *, architecture="gpt2", trained_context=N_CTX) -> None:
    rng = np.random.default_rng(0)
    entries = [
        ("token_embd.weight", (N_EMBD, N_VOCAB)),
        ("position_embd.weight", (N_EMBD, trained_context)),
        ("output_norm.weight", (N_EMBD,)),
        ("output_norm.bias", (N_EMBD,)),
    ]
    for i in range(N_LAYER):
        entries += [
            (f"blk.{i}.attn_norm.weight", (N_EMBD,)),
            (f"blk.{i}.attn_norm.bias", (N_EMBD,)),
            (f"blk.{i}.attn_qkv.weight", (N_EMBD, 3 * N_EMBD)),
            (f"blk.{i}.attn_qkv.bias", (3 * N_EMBD,)),
            (f"blk.{i}.attn_output.weight", (N_EMBD, N_EMBD)),
            (f"blk.{i}.attn_output.bias", (N_EMBD,)),
            (f"blk.{i}.ffn_norm.weight", (N_EMBD,)),
            (f"blk.{i}.ffn_norm.bias", (N_EMBD,)),
            (f"blk.{i}.ffn_up.weight", (N_EMBD, 4 * N_EMBD)),
            (f"blk.{i}.ffn_up.bias", (4 * N_EMBD,)),
            (f"blk.{i}.ffn_down.weight", (4 * N_EMBD, N_EMBD)),
            (f"blk.{i}.ffn_down.bias", (N_EMBD,)),
        ]

    with GGUF.init_empty() as out, Context(1 << 20, no_alloc=False) as ctx:
        out.set_val_str("general.architecture", architecture)
        out.set_val_u32("gpt2.context_length", trained_context)
        out.set_val_u32("gpt2.embedding_length", N_EMBD)
        out.set_val_u32("gpt2.block_count", N_LAYER)
        out.set_val_u32("gpt2.attention.head_count", N_HEAD)
        out.set_val_f32("gpt2.attention.layer_norm_epsilon", 1e-5)
        out.set_arr_str("tokenizer.ggml.tokens", TOKENS)
        out.set_val_u32("tokenizer.ggml.eos_token_id", N_VOCAB - 1)
        for name, ne in entries:  # ne is ggml order
            values = rng.standard_normal(int(np.prod(ne))).astype(np.float32) * 0.1
            out.add_tensor(ctx.new_tensor(F32, *ne, name=name))
            out.set_tensor_data(name, values.tobytes())
        out.write_to_file(str(path))

"""CLIP inference using ggbond Graph API with GGUF format.

Usage:
    python examples/clip.py -m models/clip.gguf

Weight name convention (GGUF keys — llama.cpp / clip.cpp format):
    Vision:  v.blk.{i}.{ln1,ln2,attn_q,attn_k,attn_v,attn_out,ffn_down,ffn_up}.{weight,bias}
             v.class_embd, v.patch_embd.weight, v.position_embd.weight
             v.pre_ln.weight/bias, v.post_ln.weight/bias
             visual_projection.weight
    Text:    t.blk.{i}.{ln1,ln2,attn_q,attn_k,attn_v,attn_out,ffn_down,ffn_up}.{weight,bias}
             t.token_embd.weight, t.position_embd.weight
             t.post_ln.weight/bias
             text_projection.weight

MLP naming convention in this GGUF (confusingly swapped from what you'd expect):
    ffn_down.weight [n_embd, 4*n_embd]  → expand:   n_embd → 4*n_embd  (fc / up-proj)
    ffn_up.weight   [4*n_embd, n_embd]  → contract: 4*n_embd → n_embd  (proj / down-proj)

Weight storage convention (llama.cpp style):
    Linear weights stored as [K, M] (K=input_dim, M=output_dim) so that
    mul_mat(W, x) with x=[K, N] gives [M, N].
"""

import argparse
import math
from dataclasses import dataclass

import numpy as np

import ggbond
from ggbond import ggml
from ggbond.graph import GAllocr, Graph

CLIP_MAX_NODES = 8192


# ============================================================================
# Model structures
# ============================================================================

@dataclass
class ClipHparams:
    # Vision encoder
    image_size: int = 224
    patch_size: int = 32
    vision_width: int = 768
    vision_layers: int = 12
    vision_heads: int = 12
    # Text encoder
    vocab_size: int = 49408
    context_length: int = 77
    text_width: int = 512
    text_layers: int = 12
    text_heads: int = 8
    # Shared projection
    embed_dim: int = 512
    eps: float = 1e-5


@dataclass
class ClipVisionLayer:
    ln_1_w: ggml.Tensor
    ln_1_b: ggml.Tensor
    attn_q_w: ggml.Tensor    # [vision_width, vision_width]
    attn_q_b: ggml.Tensor    # [vision_width]
    attn_k_w: ggml.Tensor    # [vision_width, vision_width]
    attn_k_b: ggml.Tensor    # [vision_width]
    attn_v_w: ggml.Tensor    # [vision_width, vision_width]
    attn_v_b: ggml.Tensor    # [vision_width]
    attn_out_w: ggml.Tensor  # [vision_width, vision_width]
    attn_out_b: ggml.Tensor  # [vision_width]
    ln_2_w: ggml.Tensor
    ln_2_b: ggml.Tensor
    mlp_fc_w: ggml.Tensor    # ffn_down.weight: expand  [width, 4*width]
    mlp_fc_b: ggml.Tensor    # ffn_down.bias:           [4*width]
    mlp_proj_w: ggml.Tensor  # ffn_up.weight:  contract [4*width, width]
    mlp_proj_b: ggml.Tensor  # ffn_up.bias:             [width]


@dataclass
class ClipTextLayer:
    ln_1_w: ggml.Tensor
    ln_1_b: ggml.Tensor
    attn_q_w: ggml.Tensor
    attn_q_b: ggml.Tensor
    attn_k_w: ggml.Tensor
    attn_k_b: ggml.Tensor
    attn_v_w: ggml.Tensor
    attn_v_b: ggml.Tensor
    attn_out_w: ggml.Tensor
    attn_out_b: ggml.Tensor
    ln_2_w: ggml.Tensor
    ln_2_b: ggml.Tensor
    mlp_fc_w: ggml.Tensor
    mlp_fc_b: ggml.Tensor
    mlp_proj_w: ggml.Tensor
    mlp_proj_b: ggml.Tensor


class ClipModel:
    """Holds CLIP model weights and exposes encode_image / encode_text."""

    def __init__(
        self,
        hparams: ClipHparams,
        session: ggbond.Session,
        # Vision weights
        visual_cls: ggml.Tensor,        # v.class_embd               [vision_width]
        visual_pos: ggml.Tensor,        # v.position_embd.weight     [vision_width, N_patches+1]
        visual_conv1: ggml.Tensor,      # v.patch_embd.weight        [patch, patch, 3, vision_width]
        visual_ln_pre_w: ggml.Tensor,
        visual_ln_pre_b: ggml.Tensor,
        visual_ln_post_w: ggml.Tensor,
        visual_ln_post_b: ggml.Tensor,
        visual_proj: ggml.Tensor,       # visual_projection.weight   [vision_width, embed_dim]
        vision_layers: list,
        # Text weights
        token_embd: ggml.Tensor,        # t.token_embd.weight        [text_width, vocab_size]
        text_pos: ggml.Tensor,          # t.position_embd.weight     [text_width, context_length]
        ln_final_w: ggml.Tensor,
        ln_final_b: ggml.Tensor,
        text_proj: ggml.Tensor,         # text_projection.weight     [text_width, embed_dim]
        logit_scale: ggml.Tensor | None,
        text_layers: list,
    ):
        self.hparams = hparams
        self._session = session
        self.visual_cls = visual_cls
        self.visual_pos = visual_pos
        self.visual_conv1 = visual_conv1
        self.visual_ln_pre_w = visual_ln_pre_w
        self.visual_ln_pre_b = visual_ln_pre_b
        self.visual_ln_post_w = visual_ln_post_w
        self.visual_ln_post_b = visual_ln_post_b
        self.visual_proj = visual_proj
        self.vision_layers = vision_layers
        self.token_embd = token_embd
        self.text_pos = text_pos
        self.ln_final_w = ln_final_w
        self.ln_final_b = ln_final_b
        self.text_proj = text_proj
        self.logit_scale = logit_scale
        self.text_layers = text_layers

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Encode image [C, H, W] float32 → embedding [embed_dim]."""
        return clip_encode_image(self, self._session, image)

    def encode_text(self, token_ids: np.ndarray) -> np.ndarray:
        """Encode token IDs [context_length] int32 → embedding [embed_dim]."""
        return clip_encode_text(self, self._session, token_ids)

    def __call__(
        self, image: np.ndarray, text_tokens: list
    ) -> tuple:
        """Compute image-text similarity logits.

        Args:
            image:       numpy [C, H, W] float32
            text_tokens: list of numpy [context_length] int32 arrays

        Returns:
            (logits_per_image, logits_per_text) as 2-D numpy arrays
        """
        img_feat = self.encode_image(image).reshape(1, -1)
        text_feats = np.stack([self.encode_text(t) for t in text_tokens])

        # Normalize
        img_feat = img_feat / (np.linalg.norm(img_feat, axis=1, keepdims=True) + 1e-8)
        text_feats = text_feats / (np.linalg.norm(text_feats, axis=1, keepdims=True) + 1e-8)

        # logit_scale
        if self.logit_scale is not None:
            ls_buf = np.empty(1, dtype=np.float32)
            ggml.backend_tensor_get(self.logit_scale, ls_buf, 0, 4)
            logit_scale = float(np.exp(ls_buf[0]))
        else:
            logit_scale = 100.0   # OpenCLIP default ≈ exp(4.605)

        logits_per_image = logit_scale * (img_feat @ text_feats.T)
        logits_per_text = logits_per_image.T
        return logits_per_image, logits_per_text


# ============================================================================
# Model loading
# ============================================================================

def _t(tensor):
    """Extract raw ggml.Tensor pointer from a high-level Tensor."""
    return tensor._ggml_tensor


def _c(g: Graph, t: ggml.Tensor) -> ggml.Tensor:
    """Cast tensor to F32 if needed (e.g. F16 weights used in element-wise ops)."""
    if ggml.tensor_type(t) != int(ggml.Type.F32):
        return ggml.cast(g.ctx, t, ggml.Type.F32)
    return t


def clip_model_load(fname: str, session: ggbond.Session) -> ClipModel:
    """Load CLIP model from GGUF file (llama.cpp / clip.cpp format)."""
    print(f"clip_model_load: loading from '{fname}'")
    gguf = session.load_gguf(fname)
    w = gguf.weights
    meta = gguf.meta
    print(f"clip_model_load: loaded {len(w)} tensors")

    # --- Hparams from metadata ---
    hp = ClipHparams()
    hp.image_size     = meta.get("clip.vision.image_size",           hp.image_size)
    hp.patch_size     = meta.get("clip.vision.patch_size",           hp.patch_size)
    hp.vision_width   = meta.get("clip.vision.embedding_length",     hp.vision_width)
    hp.vision_layers  = meta.get("clip.vision.block_count",          hp.vision_layers)
    hp.vision_heads   = meta.get("clip.vision.attention.head_count", hp.vision_heads)
    hp.vocab_size     = meta.get("clip.text.vocab_size",             hp.vocab_size)
    hp.context_length = meta.get("clip.text.context_length",         hp.context_length)
    hp.text_width     = meta.get("clip.text.embedding_length",       hp.text_width)
    hp.text_layers    = meta.get("clip.text.block_count",            hp.text_layers)
    hp.text_heads     = meta.get("clip.text.attention.head_count",   hp.text_heads)
    hp.embed_dim      = meta.get("clip.vision.projection_dim",       hp.embed_dim)
    v_eps = meta.get("clip.vision.attention.layer_norm_epsilon", None)
    if v_eps is not None:
        hp.eps = v_eps

    # Fallback: infer from tensor shapes when metadata is absent
    if "v.class_embd" in w:
        hp.vision_width = ggml.tensor_ne(_t(w["v.class_embd"]), 0)
    if "v.position_embd.weight" in w:
        hp.vision_width = ggml.tensor_ne(_t(w["v.position_embd.weight"]), 0)
    if "t.token_embd.weight" in w:
        hp.text_width = ggml.tensor_ne(_t(w["t.token_embd.weight"]), 0)
        hp.vocab_size = ggml.tensor_ne(_t(w["t.token_embd.weight"]), 1)
    if "t.position_embd.weight" in w:
        hp.text_width     = ggml.tensor_ne(_t(w["t.position_embd.weight"]), 0)
        hp.context_length = ggml.tensor_ne(_t(w["t.position_embd.weight"]), 1)
    if "visual_projection.weight" in w:
        hp.embed_dim = ggml.tensor_ne(_t(w["visual_projection.weight"]), 1)

    print(f"clip_model_load: image_size    = {hp.image_size}")
    print(f"clip_model_load: patch_size    = {hp.patch_size}")
    print(f"clip_model_load: vision_width  = {hp.vision_width}")
    print(f"clip_model_load: vision_layers = {hp.vision_layers}")
    print(f"clip_model_load: vision_heads  = {hp.vision_heads}")
    print(f"clip_model_load: vocab_size    = {hp.vocab_size}")
    print(f"clip_model_load: context_len   = {hp.context_length}")
    print(f"clip_model_load: text_width    = {hp.text_width}")
    print(f"clip_model_load: text_layers   = {hp.text_layers}")
    print(f"clip_model_load: text_heads    = {hp.text_heads}")
    print(f"clip_model_load: embed_dim     = {hp.embed_dim}")

    # --- Vision transformer blocks ---
    vision_layers = []
    for i in range(hp.vision_layers):
        pfx = f"v.blk.{i}."
        vision_layers.append(ClipVisionLayer(
            ln_1_w     = _t(w[pfx + "ln1.weight"]),
            ln_1_b     = _t(w[pfx + "ln1.bias"]),
            attn_q_w   = _t(w[pfx + "attn_q.weight"]),
            attn_q_b   = _t(w[pfx + "attn_q.bias"]),
            attn_k_w   = _t(w[pfx + "attn_k.weight"]),
            attn_k_b   = _t(w[pfx + "attn_k.bias"]),
            attn_v_w   = _t(w[pfx + "attn_v.weight"]),
            attn_v_b   = _t(w[pfx + "attn_v.bias"]),
            attn_out_w = _t(w[pfx + "attn_out.weight"]),
            attn_out_b = _t(w[pfx + "attn_out.bias"]),
            ln_2_w     = _t(w[pfx + "ln2.weight"]),
            ln_2_b     = _t(w[pfx + "ln2.bias"]),
            mlp_fc_w   = _t(w[pfx + "ffn_down.weight"]),   # expand:   width → 4*width
            mlp_fc_b   = _t(w[pfx + "ffn_down.bias"]),
            mlp_proj_w = _t(w[pfx + "ffn_up.weight"]),     # contract: 4*width → width
            mlp_proj_b = _t(w[pfx + "ffn_up.bias"]),
        ))

    # --- Text transformer blocks ---
    text_layers = []
    for i in range(hp.text_layers):
        pfx = f"t.blk.{i}."
        text_layers.append(ClipTextLayer(
            ln_1_w     = _t(w[pfx + "ln1.weight"]),
            ln_1_b     = _t(w[pfx + "ln1.bias"]),
            attn_q_w   = _t(w[pfx + "attn_q.weight"]),
            attn_q_b   = _t(w[pfx + "attn_q.bias"]),
            attn_k_w   = _t(w[pfx + "attn_k.weight"]),
            attn_k_b   = _t(w[pfx + "attn_k.bias"]),
            attn_v_w   = _t(w[pfx + "attn_v.weight"]),
            attn_v_b   = _t(w[pfx + "attn_v.bias"]),
            attn_out_w = _t(w[pfx + "attn_out.weight"]),
            attn_out_b = _t(w[pfx + "attn_out.bias"]),
            ln_2_w     = _t(w[pfx + "ln2.weight"]),
            ln_2_b     = _t(w[pfx + "ln2.bias"]),
            mlp_fc_w   = _t(w[pfx + "ffn_down.weight"]),
            mlp_fc_b   = _t(w[pfx + "ffn_down.bias"]),
            mlp_proj_w = _t(w[pfx + "ffn_up.weight"]),
            mlp_proj_b = _t(w[pfx + "ffn_up.bias"]),
        ))

    logit_scale = _t(w["logit_scale"]) if "logit_scale" in w else None

    return ClipModel(
        hparams          = hp,
        session          = session,
        visual_cls       = _t(w["v.class_embd"]),
        visual_pos       = _t(w["v.position_embd.weight"]),
        visual_conv1     = _t(w["v.patch_embd.weight"]),
        visual_ln_pre_w  = _t(w["v.pre_ln.weight"]),
        visual_ln_pre_b  = _t(w["v.pre_ln.bias"]),
        visual_ln_post_w = _t(w["v.post_ln.weight"]),
        visual_ln_post_b = _t(w["v.post_ln.bias"]),
        visual_proj      = _t(w["visual_projection.weight"]),
        vision_layers    = vision_layers,
        token_embd       = _t(w["t.token_embd.weight"]),
        text_pos         = _t(w["t.position_embd.weight"]),
        ln_final_w       = _t(w["t.post_ln.weight"]),
        ln_final_b       = _t(w["t.post_ln.bias"]),
        text_proj        = _t(w["text_projection.weight"]),
        logit_scale      = logit_scale,
        text_layers      = text_layers,
    )


# ============================================================================
# Graph building
# ============================================================================

def _attn_block(
    g: Graph,
    layer,
    inpL: ggml.Tensor,
    N: int,
    n_head: int,
    eps: float,
    use_attn_mask: bool,
) -> ggml.Tensor:
    """Residual attention block for CLIP (no KV cache, separate Q/K/V projections).

    Supports both vision (use_attn_mask=False) and text (use_attn_mask=True) encoders.
    QuickGELU activation: gelu_quick(x) = x * sigmoid(1.702 * x).
    """
    # Ensure F32 activations (get_rows on F16 table yields F16; norm outputs F32
    # but the incoming residual may still be F16 from positional embedding add)
    inpL = _c(g, inpL)

    n_embd = ggml.tensor_ne(inpL, 0)
    head_dim = n_embd // n_head

    # LayerNorm 1
    cur = g.norm(inpL, eps)
    cur = g.add(g.mul(cur, _c(g, layer.ln_1_w)), _c(g, layer.ln_1_b))

    # Separate Q, K, V projections: [n_embd, n_embd] × [n_embd, N] → [n_embd, N]
    Qcur = g.add(g.mul_mat(layer.attn_q_w, cur), _c(g, layer.attn_q_b))
    Kcur = g.add(g.mul_mat(layer.attn_k_w, cur), _c(g, layer.attn_k_b))
    Vcur = g.add(g.mul_mat(layer.attn_v_w, cur), _c(g, layer.attn_v_b))

    # Reshape to [head_dim, n_head, N] then permute to [head_dim, N, n_head]
    Q = g.permute(g.cont_3d(Qcur, head_dim, n_head, N), 0, 2, 1, 3)
    K = g.permute(g.cont_3d(Kcur, head_dim, n_head, N), 0, 2, 1, 3)
    # V_trans: [N, head_dim, n_head] for batched matmul with KQ_soft_max
    V_trans = g.cont_3d(
        g.permute(g.cont_3d(Vcur, head_dim, n_head, N), 1, 2, 0, 3),
        N, head_dim, n_head,
    )

    # Attention scores: mul_mat(K, Q) → [N, N, n_head]
    KQ = g.mul_mat(K, Q)
    KQ_scaled = g.scale(KQ, 1.0 / math.sqrt(head_dim))
    if use_attn_mask:
        KQ_scaled = g.diag_mask_inf(KQ_scaled, 0)   # causal mask (n_past=0)
    KQ_soft_max = g.soft_max(KQ_scaled)

    # Weighted sum: mul_mat(V_trans, KQ_soft_max) → [head_dim, N, n_head]
    KQV = g.mul_mat(V_trans, KQ_soft_max)
    KQV_merged = g.permute(KQV, 0, 2, 1, 3)
    cur = g.cont_2d(KQV_merged, n_embd, N)

    # Output projection
    cur = g.mul_mat(layer.attn_out_w, cur)
    cur = g.add(cur, _c(g, layer.attn_out_b))

    cur = g.add(cur, inpL)   # residual

    inpFF = cur

    # LayerNorm 2
    cur = g.norm(inpFF, eps)
    cur = g.add(g.mul(cur, _c(g, layer.ln_2_w)), _c(g, layer.ln_2_b))

    # MLP: expand + QuickGELU + contract
    cur = g.mul_mat(layer.mlp_fc_w, cur)      # [width, 4*width] × [width, N] → [4*width, N]
    cur = g.add(cur, _c(g, layer.mlp_fc_b))
    cur = ggml.gelu_quick(g.ctx, cur)          # QuickGELU: x * sigmoid(1.702 * x)
    cur = g.mul_mat(layer.mlp_proj_w, cur)    # [4*width, width] × [4*width, N] → [width, N]
    cur = g.add(cur, _c(g, layer.mlp_proj_b))

    return g.add(cur, inpFF)   # residual


def clip_vision_graph(model: ClipModel) -> Graph:
    """Build the CLIP vision encoder computation graph."""
    hp = model.hparams
    grid_size = hp.image_size // hp.patch_size
    N_patches = grid_size * grid_size
    N = N_patches + 1   # patches + class token

    g = Graph(max_nodes=CLIP_MAX_NODES)

    # Input: GGML [W, H, C, 1] — uploaded from numpy [C, H, W] via transpose
    img_t = g.new_tensor(ggml.Type.F32, hp.image_size, hp.image_size, 3, 1, name="image")
    ggml.set_input(img_t)

    # Patch embedding via conv2d
    # v.patch_embd.weight: [patch_size, patch_size, 3, vision_width]
    # conv2d output:       [grid_size, grid_size, vision_width, 1]
    patches_4d = ggml.conv_2d(
        g.ctx, model.visual_conv1, img_t,
        hp.patch_size, hp.patch_size,   # strides
        0, 0,                           # padding
        1, 1,                           # dilation
    )

    # Permute [gs, gs, width, 1] → [width, gs, gs, 1], then reshape to [width, N_patches]
    patches_perm = g.permute(patches_4d, 2, 0, 1, 3)
    patches_wn = g.cont_2d(patches_perm, hp.vision_width, N_patches)

    # Prepend class token: concat([width,1], [width,N_patches], dim=1) → [width, N]
    cls_2d = ggml.reshape_2d(g.ctx, _c(g, model.visual_cls), hp.vision_width, 1)
    cur = g.concat(cls_2d, patches_wn, dim=1)

    # Add positional embedding [width, N]
    cur = g.add(cur, _c(g, model.visual_pos))

    # ln_pre
    cur = g.norm(cur, hp.eps)
    cur = g.add(g.mul(cur, _c(g, model.visual_ln_pre_w)), _c(g, model.visual_ln_pre_b))

    # Vision transformer blocks (no causal mask)
    for layer in model.vision_layers:
        cur = _attn_block(g, layer, cur, N, hp.vision_heads, hp.eps, use_attn_mask=False)

    # Extract CLS token (column 0): view_2d → [width, 1]
    nb1 = ggml.tensor_nb(cur, 1)
    cls_out = g.view_2d(cur, hp.vision_width, 1, nb1, 0)

    # ln_post
    cls_out = g.norm(cls_out, hp.eps)
    cls_out = g.add(g.mul(cls_out, _c(g, model.visual_ln_post_w)), _c(g, model.visual_ln_post_b))

    # Visual projection: [vision_width, embed_dim] × [vision_width, 1] → [embed_dim, 1]
    out = g.mul_mat(model.visual_proj, cls_out)
    ggml.set_name(out, "image_feat")
    ggml.set_output(out)
    g.build_forward(out)

    return g


def clip_text_graph(model: ClipModel, eot_idx: int) -> Graph:
    """Build the CLIP text encoder computation graph."""
    hp = model.hparams
    N = hp.context_length

    g = Graph(max_nodes=CLIP_MAX_NODES)

    # Input token IDs [N]
    tokens_t = g.new_tensor(ggml.Type.I32, N, name="tokens")
    ggml.set_input(tokens_t)

    # Token embedding lookup [text_width, vocab_size] × indices[N] → [text_width, N]
    # get_rows inherits the source type (F16 if token_embd is F16); cast to F32
    inpL = _c(g, g.get_rows(model.token_embd, tokens_t))

    # Add positional embedding [text_width, context_length]
    cur = g.add(inpL, _c(g, model.text_pos))

    # Text transformer blocks (with causal mask)
    for layer in model.text_layers:
        cur = _attn_block(g, layer, cur, N, hp.text_heads, hp.eps, use_attn_mask=True)

    # Final LayerNorm (t.post_ln)
    cur = g.norm(cur, hp.eps)
    cur = g.add(g.mul(cur, _c(g, model.ln_final_w)), _c(g, model.ln_final_b))

    # Extract EOT token at eot_idx: view [text_width, 1]
    nb1 = ggml.tensor_nb(cur, 1)
    eot_feat = g.view_2d(cur, hp.text_width, 1, nb1, eot_idx * nb1)

    # Text projection: [text_width, embed_dim] × [text_width, 1] → [embed_dim, 1]
    out = g.mul_mat(model.text_proj, eot_feat)
    ggml.set_name(out, "text_feat")
    ggml.set_output(out)
    g.build_forward(out)

    return g


# ============================================================================
# Encode functions
# ============================================================================

def clip_encode_image(
    model: ClipModel, session: ggbond.Session, image: np.ndarray
) -> np.ndarray:
    """Encode an image to a feature vector.

    Args:
        image: numpy [C, H, W] float32 in the range expected by the model.

    Returns:
        1-D numpy array of shape [embed_dim] (unnormalized).
    """
    g = clip_vision_graph(model)

    allocr = GAllocr(session.backend)
    allocr.reserve(g.raw)
    allocr.alloc(g.raw)

    # Upload image: numpy [C, H, W] → GGML [W, H, C] by transposing axes (2,1,0)
    img_data = np.ascontiguousarray(image.transpose(2, 1, 0), dtype=np.float32)
    session.backend.tensor_set(ggml.graph_get_tensor(g.raw, "image"), img_data)

    session.backend.compute(g.raw)

    feat_t = ggml.graph_get_tensor(g.raw, "image_feat")
    feat = session.backend.tensor_get(feat_t)

    allocr.close()
    g.close()

    return feat


def clip_encode_text(
    model: ClipModel, session: ggbond.Session, token_ids: np.ndarray
) -> np.ndarray:
    """Encode tokenized text to a feature vector.

    Args:
        token_ids: int32 array of shape [context_length] (padded with 0s).
                   The EOT token position is found via argmax.

    Returns:
        1-D numpy array of shape [embed_dim] (unnormalized).
    """
    # EOT token is at argmax of the token ID sequence
    eot_idx = int(np.argmax(token_ids))

    g = clip_text_graph(model, eot_idx)

    allocr = GAllocr(session.backend)
    allocr.reserve(g.raw)
    allocr.alloc(g.raw)

    tokens = np.array(token_ids, dtype=np.int32)
    session.backend.tensor_set(ggml.graph_get_tensor(g.raw, "tokens"), tokens)

    session.backend.compute(g.raw)

    feat_t = ggml.graph_get_tensor(g.raw, "text_feat")
    feat = session.backend.tensor_get(feat_t)

    allocr.close()
    g.close()

    return feat


# ============================================================================
# Image preprocessing
# ============================================================================

_CLIP_MEAN = np.array([0.48145466, 0.4578275,  0.40821073], dtype=np.float32)
_CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def preprocess_image(path: str, size: int = 224) -> np.ndarray:
    """Load and preprocess an image for CLIP.

    Returns float32 array of shape [C, H, W] normalized with CLIP mean/std.
    """
    from PIL import Image
    img = Image.open(path).convert("RGB")
    w, h = img.size
    # Resize shortest side to `size`, keep aspect ratio
    if w < h:
        new_w, new_h = size, int(h * size / w)
    else:
        new_w, new_h = int(w * size / h), size
    img = img.resize((new_w, new_h), Image.BICUBIC)
    # Center crop to size × size
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    arr = np.array(img, dtype=np.float32) / 255.0   # [H, W, C] in [0,1]
    arr = arr.transpose(2, 0, 1)                     # → [C, H, W]
    arr = (arr - _CLIP_MEAN[:, None, None]) / _CLIP_STD[:, None, None]
    return arr


def tokenize(texts: list[str], context_length: int = 77) -> np.ndarray:
    """Tokenize a list of strings using open_clip's BPE tokenizer.

    Returns int32 array of shape [len(texts), context_length].
    Requires: pip install open_clip_torch
    """
    try:
        import open_clip
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        tokens = tokenizer(texts)          # torch.Tensor [N, 77]
        return tokens.numpy().astype(np.int32)
    except ImportError as e:
        raise ImportError(
            "Text tokenization requires open_clip_torch: pip install open_clip_torch"
        ) from e


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CLIP inference with ggbond Graph API")
    parser.add_argument("-m", "--model", type=str, default="models/clip.gguf",
                        help="Path to GGUF model file")
    parser.add_argument("-i", "--image", type=str, default=None,
                        help="Path to image file (JPEG/PNG/...)")
    parser.add_argument("-p", "--prompts", type=str, nargs="+",
                        default=["a photo of a cat", "a photo of a dog",
                                 "a photo of a person", "a landscape photo"],
                        help="Text prompts to rank against the image")
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu", "cuda", "metal", "hip"],
                        help="Backend to use")
    parser.add_argument("-t", "--threads", type=int, default=4,
                        help="Number of CPU threads")
    args = parser.parse_args()

    t_start = ggml.time_us()

    with ggbond.Session(args.backend, n_threads=args.threads) as session:
        model = clip_model_load(args.model, session)
        t_load = ggml.time_us() - t_start
        print(f"\nmain: model loaded in {t_load / 1000:.1f} ms")

        hp = model.hparams

        # --- Image ---
        if args.image is not None:
            print(f"\nmain: loading image '{args.image}'")
            image = preprocess_image(args.image, hp.image_size)
        else:
            print("\nmain: no image provided, using random noise (use -i to pass a real image)")
            rng = np.random.default_rng(42)
            image = rng.standard_normal((3, hp.image_size, hp.image_size)).astype(np.float32)

        t_img_start = ggml.time_us()
        img_feat = clip_encode_image(model, session, image)
        t_img = ggml.time_us() - t_img_start
        img_norm = img_feat / (np.linalg.norm(img_feat) + 1e-8)
        print(f"main: image encoding : {t_img / 1000:.1f} ms  shape={img_feat.shape}")

        # --- Text prompts ---
        print(f"\nmain: encoding {len(args.prompts)} text prompt(s)")
        try:
            all_tokens = tokenize(args.prompts, hp.context_length)  # [N, 77]
        except ImportError as e:
            print(f"  warning: {e}")
            print("  falling back to dummy tokens (install open_clip_torch for real results)")
            all_tokens = np.zeros((len(args.prompts), hp.context_length), dtype=np.int32)
            for i in range(len(args.prompts)):
                all_tokens[i, 0] = 49406   # BOS
                all_tokens[i, 1] = 49407   # EOT

        text_feats = []
        for i, prompt in enumerate(args.prompts):
            t0 = ggml.time_us()
            feat = clip_encode_text(model, session, all_tokens[i])
            dt = ggml.time_us() - t0
            feat_norm = feat / (np.linalg.norm(feat) + 1e-8)
            text_feats.append(feat_norm)
            print(f"  [{i}] {dt/1000:5.1f} ms  \"{prompt}\"")

        # --- Similarity ranking ---
        text_feats = np.stack(text_feats)          # [N, embed_dim]
        sims = (text_feats @ img_norm).tolist()    # [N]
        ranked = sorted(zip(sims, args.prompts), reverse=True)

        print("\nmain: similarity ranking (image vs. text prompts):")
        for rank, (sim, prompt) in enumerate(ranked):
            bar = "#" * int(max(sim, 0) * 40)
            print(f"  #{rank+1}  {sim:+.4f}  {bar:<40}  \"{prompt}\"")


if __name__ == "__main__":
    main()

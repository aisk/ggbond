"""Offline NAM A1 WaveNet inference on the high-level ``ggbond`` API.

A port of Neural Amp Modeler's "A1" (feed-forward WaveNet) offline renderer.
It reads a ``.nam`` file (JSON: architecture + config + a flat ``weights``
array), rebuilds the WaveNet as a ggml graph, and renders a WAV through it.

The weights are variables of a :class:`Session` (declared together, then filled
from the flat array). One graph is opened for the render: the padded audio is
its input, the forward pass is written with ggml's own operations, and
``g.run`` returns the mono result as a numpy array. ``examples/ggml/nam.py`` is
the same program on the low-level :mod:`ggbond.ggml` layer.

Convolutions use ``ggml_im2col`` (F32) + ``ggml_mul_mat`` -- the same
decomposition as ``ggml_conv_1d``, but keeping the kernel in F32 instead of the
builtin's forced F16, so the render matches the F32 PyTorch reference.

Scope: the standard A1 subset -- one or more layer arrays, ``groups == 1``,
``layer1x1`` active, ``head1x1`` inactive, ``head`` rechannel kernel size 1, no
separate output head, no ``condition_dsp``. Activations tanh/relu/sigmoid/
identity and the legacy ``gated`` (primary * sigmoid) pairing are supported.

Usage::

    python examples/nam.py model.nam input.wav output.wav
    python examples/nam.py model.nam input.wav output.wav -b cpu -t 8
"""

from __future__ import annotations

import argparse
import json
import wave

import numpy as np

from ggbond import F32, GraphScope, Session, Tensor, Variable

NAM_MAX_NODES = 8192  # graph_size: tensors one render graph may hold


class NamError(RuntimeError):
    pass


# ============================================================================
# Graph-side ops
# ============================================================================

def conv1d(kernel: Variable, x: Tensor, dilation: int = 1) -> Tensor:
    """Valid (no-pad, stride-1) 1D convolution, F32 throughout.

    ``kernel`` ne = ``(K, IC, OC)`` (reversed torch ``[OC, IC, K]``), ``x`` ne =
    ``(L, IC)``. Returns ne = ``(OL, OC)`` with ``OL = L - (K-1)*dilation``.
    Mirrors ``ggml_conv_1d`` but with an F32 im2col so F32 kernels are kept.
    """
    # (kernel, data, s0=1, s1, p0, p1, d0=dilation, d1, is_2d, dst_type)
    # ``kernel`` is a variable: its ops land in the graph that is open.
    im = kernel.im2col(x, 1, 0, 0, 0, dilation, 0, False, F32)  # ne = (IC*K, OL, N)
    ick, ol, n = im.ne[0], im.ne[1], im.ne[2]
    k, ic, oc = kernel.ne[0], kernel.ne[1], kernel.ne[2]
    im2 = im.reshape_2d(ick, n * ol)
    k2 = kernel.reshape_2d(k * ic, oc)
    return im2.mul_mat(k2).reshape_3d(ol, oc, n)  # (OL, OC, N=1)


def slice_tail(t: Tensor, n: int) -> Tensor:
    """Keep the last ``n`` frames along the length axis (ne0), all channels.

    Right-aligned trim used everywhere NAM writes ``x[:, :, -n:]``. Returns a
    contiguous copy so it can feed the next convolution.
    """
    length, channels = t.ne[0], t.ne[1]
    if n == length:
        return t
    if n > length:
        raise NamError(f"slice_tail: want {n} frames but tensor has {length}")
    off = (length - n) * t.nb[0]
    return t.view_2d(n, channels, t.nb[1], off).cont()


def _act(t: Tensor, name: str) -> Tensor:
    name = name.lower()
    if name == "tanh":
        return t.tanh()
    if name == "relu":
        return t.relu()
    if name == "sigmoid":
        return t.sigmoid()
    if name in {"identity", "linear", "none"}:
        return t
    raise NamError(f"Unsupported activation: {name!r}")


def apply_activation(z: Tensor, spec: dict) -> Tensor:
    """Apply a per-layer activation. Plain: one op. Gated: split the channel
    axis in half and return ``primary(first) * secondary(second)``."""
    if spec["kind"] == "plain":
        return _act(z, spec["primary"])
    length, mid = z.ne[0], z.ne[1]
    half = mid // 2
    nb1 = z.nb[1]
    a = z.view_2d(length, half, nb1, 0).cont()
    b = z.view_2d(length, half, nb1, half * nb1).cont()
    return _act(a, spec["primary"]).mul(_act(b, spec["secondary"]))


# ============================================================================
# Config normalization (mirrors nam's export/import; no torch needed)
# ============================================================================

def _broadcast(value, n: int) -> list:
    if isinstance(value, list):
        if len(value) != n:
            raise NamError(f"expected {n} entries, got {len(value)}")
        return value
    return [value] * n


def _activation_specs(cfg: dict, n: int) -> list[dict]:
    """Per-layer activation specs: ``{"kind": "plain"|"gated", "primary", "secondary"}``."""
    primaries = _broadcast(cfg.get("activation", "Tanh"), n)
    if "gating_mode" in cfg:  # modern per-layer gating
        modes = _broadcast(cfg.get("gating_mode", "none"), n)
        secondaries = _broadcast(cfg.get("secondary_activation", None), n)
        specs = []
        for primary, mode, secondary in zip(primaries, modes, secondaries):
            if mode in (None, "none"):
                specs.append({"kind": "plain", "primary": str(primary)})
            elif mode == "gated":
                specs.append({"kind": "gated", "primary": str(primary),
                              "secondary": str(secondary or "Sigmoid")})
            else:
                raise NamError(f"Unsupported gating_mode: {mode!r}")
        return specs
    if cfg.get("gated"):  # legacy boolean gate: primary * sigmoid
        return [{"kind": "gated", "primary": str(p), "secondary": "Sigmoid"} for p in primaries]
    return [{"kind": "plain", "primary": str(p)} for p in primaries]


def _normalize_array(cfg: dict) -> dict:
    """Extract the fields this example supports, rejecting anything it doesn't."""
    out = dict(cfg)

    # head rechannel: modern {out_channels, kernel_size, bias} or legacy head_size/head_bias
    if "head_size" in out:
        head_size = int(out["head_size"])
        head_bias = bool(out.get("head_bias", True))
        head_kernel = 1
    else:
        head = dict(out.get("head", {}))
        head_size = int(head.get("out_channels", 1))
        head_bias = bool(head.get("bias", True))
        head_kernel = int(head.get("kernel_size", 1))
    if head_kernel != 1:
        raise NamError("only head rechannel kernel_size == 1 is supported")

    layer1x1 = dict(out.get("layer1x1") or {"active": True, "groups": 1})
    head1x1 = dict(out.get("head1x1") or {"active": False})
    if not layer1x1.get("active", True):
        raise NamError("layer1x1 must be active in this example")
    if head1x1.get("active", False):
        raise NamError("head1x1 is not supported in this example")
    if int(out.get("groups_input", 1)) != 1 or int(out.get("groups_input_mixin", 1)) != 1:
        raise NamError("grouped convolutions are not supported in this example")

    dilations = [int(d) for d in out["dilations"]]
    channels = int(out["channels"])
    bottleneck = int(out.get("bottleneck") or channels)
    return {
        "input_size": int(out["input_size"]),
        "condition_size": int(out["condition_size"]),
        "channels": channels,
        "bottleneck": bottleneck,
        "kernel_sizes": [int(k) for k in _broadcast(out["kernel_size"], len(dilations))]
        if "kernel_sizes" not in out
        else [int(k) for k in out["kernel_sizes"]],
        "dilations": dilations,
        "head_size": head_size,
        "head_bias": head_bias,
        "acts": _activation_specs(out, len(dilations)),
    }


# ============================================================================
# Model
# ============================================================================

class NamModel:
    """The weights (session variables) and what the forward pass needs to know."""

    def __init__(self):
        self.arrays: list[dict] = []
        self.head_scale = 1.0
        self.receptive_field = 1
        self.sample_rate = 48000


def _kernel(k: int, ic: int, oc: int) -> tuple:
    return (F32, k, ic, oc)


def _bias(oc: int) -> tuple:
    # ne = (1, OC) so it broadcasts across the length axis in ggml_add.
    return (F32, 1, oc)


def load_model(path: str, sess: Session) -> NamModel:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("architecture") != "WaveNet":
        raise NamError(f"only A1 WaveNet .nam files are supported; got {data.get('architecture')!r}")

    cfg = data["config"]
    if cfg.get("condition_dsp") is not None:
        raise NamError("condition_dsp is not supported in this example")
    if cfg.get("head") is not None:
        raise NamError("a separate output head is not supported in this example")

    arrays_cfg = cfg.get("layers") or cfg.get("layers_configs")
    if not arrays_cfg:
        raise NamError("WaveNet config is missing 'layers'")
    metas = [_normalize_array(a) for a in arrays_cfg]

    model = NamModel()
    model.sample_rate = int(round(float(data.get("sample_rate", 48000))))
    model.receptive_field = 1 + sum(
        (k - 1) * d for m in metas for k, d in zip(m["kernel_sizes"], m["dilations"])
    )

    # -- declare the weights, in NAM's export order --
    #   per array: rechannel + 5 per layer (conv_w/b, mix_w, l1x1_w/b) + head_w (+ head_b)
    spec: dict[str, tuple] = {}
    for ai, m in enumerate(metas):
        cond, chan, bott = m["condition_size"], m["channels"], m["bottleneck"]
        spec[f"a{ai}.rechannel"] = _kernel(1, m["input_size"], chan)
        for li, (ks, act) in enumerate(zip(m["kernel_sizes"], m["acts"])):
            mid = 2 * bott if act["kind"] == "gated" else bott
            spec[f"a{ai}.l{li}.conv"] = _kernel(ks, chan, mid)
            spec[f"a{ai}.l{li}.conv_b"] = _bias(mid)
            spec[f"a{ai}.l{li}.mix"] = _kernel(1, cond, mid)
            spec[f"a{ai}.l{li}.l1x1"] = _kernel(1, bott, chan)
            spec[f"a{ai}.l{li}.l1x1_b"] = _bias(chan)
        spec[f"a{ai}.head"] = _kernel(1, bott, m["head_size"])
        if m["head_bias"]:
            spec[f"a{ai}.head_b"] = _bias(m["head_size"])
    w = sess.variables(spec)

    # -- fill them, walking the flat array in that same order --
    # A flat slice is written as is: ggml's (K, IC, OC) memory layout is torch's
    # [OC, IC, K] flattened.
    weights = np.asarray(data["weights"], dtype=np.float32)
    off = 0
    for var in w.values():
        n = var.nelements
        var.write(weights[off:off + n])
        off += n

    # The exported weight blob stores head_scale as its final element (authoritative;
    # the config value can be stale), matching NAM's own import.
    if off != weights.size - 1:
        raise NamError(f"weight import consumed {off}, file has {weights.size} (+head_scale)")
    model.head_scale = float(weights[off])

    for ai, m in enumerate(metas):
        model.arrays.append({
            "meta": m,
            "rechannel": w[f"a{ai}.rechannel"],
            "layers": [{
                "conv_w": w[f"a{ai}.l{li}.conv"],
                "conv_b": w[f"a{ai}.l{li}.conv_b"],
                "mix_w": w[f"a{ai}.l{li}.mix"],
                "l1x1_w": w[f"a{ai}.l{li}.l1x1"],
                "l1x1_b": w[f"a{ai}.l{li}.l1x1_b"],
                "dilation": dil,
                "act": act,
            } for li, (dil, act) in enumerate(zip(m["dilations"], m["acts"]))],
            "head_w": w[f"a{ai}.head"],
            "head_b": w.get(f"a{ai}.head_b"),
        })
    return model


# ============================================================================
# Forward graph
# ============================================================================

def _layer_forward(layer, x, cond, out_length):
    """One WaveNet layer. Returns (residual, head_term)."""
    dil = layer["dilation"]
    zconv = conv1d(layer["conv_w"], x, dil).add(layer["conv_b"])
    mix = conv1d(layer["mix_w"], cond, 1)
    mix = slice_tail(mix, zconv.ne[0])
    z = zconv.add(mix)
    post = apply_activation(z, layer["act"])

    layer_out = conv1d(layer["l1x1_w"], post, 1).add(layer["l1x1_b"])
    head_term = slice_tail(post, out_length)
    residual = slice_tail(x, layer_out.ne[0]).add(layer_out)
    return residual, head_term


def _array_forward(arr, y, cond, head_input):
    """One layer array. Returns (head_output, layer_output)."""
    m = arr["meta"]
    rf = 1 + sum((k - 1) * d for k, d in zip(m["kernel_sizes"], m["dilations"]))
    out_length = min(y.ne[0], cond.ne[0]) - (rf - 1)
    if out_length <= 0:
        raise NamError("input is shorter than the model receptive field")

    x = conv1d(arr["rechannel"], y, 1)
    for layer in arr["layers"]:
        x, head_term = _layer_forward(layer, x, cond, out_length)
        if head_input is None:
            head_input = head_term
        else:
            head_input = slice_tail(head_input, out_length).add(head_term)

    head_out = conv1d(arr["head_w"], head_input, 1)
    if arr["head_b"] is not None:
        head_out = head_out.add(arr["head_b"])
    return head_out, slice_tail(x, out_length)


def forward(model: NamModel, g: GraphScope, x: np.ndarray) -> Tensor:
    """Write the full forward pass for the samples ``x`` into the open graph."""
    inp = g.input(x.reshape(1, -1), name="input")  # ne = (n_samples, 1)
    cond = inp
    y = inp
    head_input = None
    for arr in model.arrays:
        head_input, y = _array_forward(arr, y, cond, head_input)
    return head_input.scale(model.head_scale)


def run_offline(model: NamModel, sess: Session, audio: np.ndarray) -> np.ndarray:
    pad = model.receptive_field - 1
    x = np.concatenate([np.zeros(pad, np.float32), audio.astype(np.float32)]) if pad > 0 else audio

    with sess.graph() as g:
        y = g.run(forward(model, g, x)).reshape(-1)

    if y.size != audio.size:  # guard against off-by-one padding
        y = y[-audio.size:]
    return y


# ============================================================================
# WAV I/O (numpy-vectorized mono downmix)
# ============================================================================

def read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())

    if width == 1:  # unsigned 8-bit
        a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:  # packed 24-bit little-endian signed
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v & 0x800000, v - 0x1000000, v)
        a = v.astype(np.float32) / 8388608.0
    elif width == 4:
        a = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise NamError(f"unsupported WAV sample width: {width} bytes")

    a = a.reshape(-1, channels).mean(axis=1)
    return a.astype(np.float32), rate


def write_wav_mono(path: str, audio: np.ndarray, rate: int) -> None:
    ints = np.clip(audio, -1.0, 1.0)
    ints = np.round(ints * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(ints.tobytes())


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    devices = ", ".join(Session.devices())
    parser = argparse.ArgumentParser(description="Offline NAM A1 WaveNet inference on ggbond")
    parser.add_argument("model", help="Path to a .nam model file")
    parser.add_argument("input", help="Path to an input WAV file")
    parser.add_argument("output", help="Path to write the processed WAV")
    parser.add_argument("-b", "--backend", default=None, metavar="DEVICE",
                        help=f"Device name ({devices}) or type (cpu, gpu, ...); "
                             "default: the best available device")
    parser.add_argument("-t", "--threads", type=int, default=None,
                        help="CPU threads (default: the physical cores)")
    parser.add_argument("--allow-sample-rate-mismatch", action="store_true",
                        help="Run even when the input rate differs from the model rate")
    args = parser.parse_args()

    try:
        sess = Session(device=args.backend, n_threads=args.threads, graph_size=NAM_MAX_NODES)
    except ValueError as e:  # unknown device: the message lists the available ones
        parser.error(str(e))

    with sess:
        model = load_model(args.model, sess)
        audio, in_rate = read_wav_mono(args.input)
        if in_rate != model.sample_rate and not args.allow_sample_rate_mismatch:
            raise NamError(
                f"input WAV is {in_rate} Hz, model expects {model.sample_rate} Hz. "
                "Resample first or pass --allow-sample-rate-mismatch."
            )
        out = run_offline(model, sess, audio)
        write_wav_mono(args.output, out, in_rate)
        print(f"Wrote {args.output} ({out.size} samples, rf={model.receptive_field}, "
              f"devices={', '.join(b.name for b in sess.backends)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

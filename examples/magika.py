"""Magika file type detector using GGML neural network inference.

A Python reimplementation of vendor/ggml/examples/magika/main.cpp.
Reads file header/middle/tail (512 bytes each), applies one-hot encoding,
and runs through a Dense+LayerNorm+MaxPool network to classify into 113 file types.

Usage:
    python examples/magika.py <model.gguf> <file1> [file2 ...] [--backend cpu|metal]
"""

import argparse
import os

import numpy as np

from ggbond import ggml


MAGIKA_LABELS = [
    "ai",              "apk",             "appleplist",      "asm",             "asp",
    "batch",           "bmp",             "bzip",            "c",               "cab",
    "cat",             "chm",             "coff",            "crx",             "cs",
    "css",             "csv",             "deb",             "dex",             "dmg",
    "doc",             "docx",            "elf",             "emf",             "eml",
    "epub",            "flac",            "gif",             "go",              "gzip",
    "hlp",             "html",            "ico",             "ini",             "internetshortcut",
    "iso",             "jar",             "java",            "javabytecode",    "javascript",
    "jpeg",            "json",            "latex",           "lisp",            "lnk",
    "m3u",             "macho",           "makefile",        "markdown",        "mht",
    "mp3",             "mp4",             "mscompress",      "msi",             "mum",
    "odex",            "odp",             "ods",             "odt",             "ogg",
    "outlook",         "pcap",            "pdf",             "pebin",           "pem",
    "perl",            "php",             "png",             "postscript",      "powershell",
    "ppt",             "pptx",            "python",          "pythonbytecode",  "rar",
    "rdf",             "rpm",             "rst",             "rtf",             "ruby",
    "rust",            "scala",           "sevenzip",        "shell",           "smali",
    "sql",             "squashfs",        "svg",             "swf",             "symlinktext",
    "tar",             "tga",             "tiff",            "torrent",         "ttf",
    "txt",             "unknown",         "vba",             "wav",             "webm",
    "webp",            "winregistry",     "wmf",             "xar",             "xls",
    "xlsb",            "xlsx",            "xml",             "xpi",             "xz",
    "yaml",            "zip",             "zlibstream",
]

# Hyperparameters (matching C++ magika_hparams)
BEG_SIZE = 512
MID_SIZE = 512
END_SIZE = 512
N_LABEL = 113
F_NORM_EPS = 0.001
PADDING_TOKEN = 256
INP_BYTES = BEG_SIZE + MID_SIZE + END_SIZE  # 1536

TENSOR_NAMES = [
    "dense/kernel:0",
    "dense/bias:0",
    "layer_normalization/gamma:0",
    "layer_normalization/beta:0",
    "dense_1/kernel:0",
    "dense_1/bias:0",
    "dense_2/kernel:0",
    "dense_2/bias:0",
    "layer_normalization_1/gamma:0",
    "layer_normalization_1/beta:0",
    "target_label/kernel:0",
    "target_label/bias:0",
]


def load_model(fname, backend):
    """Load GGUF model weights and return (ctx_w, buf_w, tensors)."""
    gguf_ctx, ctx_w = ggml.gguf_init_from_file(fname, no_alloc=True)
    buf_w = ggml.backend_alloc_ctx_tensors(ctx_w, backend)

    tensors = {}
    for name in TENSOR_NAMES:
        tensors[name] = ggml.get_tensor(ctx_w, name)

    n_tensors = ggml.gguf_get_n_tensors(gguf_ctx)
    data_offset = ggml.gguf_get_data_offset(gguf_ctx)

    with open(fname, "rb") as f:
        for i in range(n_tensors):
            name = ggml.gguf_get_tensor_name(gguf_ctx, i)
            tensor = ggml.get_tensor(ctx_w, name)
            offset = data_offset + ggml.gguf_get_tensor_offset(gguf_ctx, i)
            nbytes = ggml.nbytes(tensor)
            f.seek(offset)
            data = np.frombuffer(f.read(nbytes), dtype=np.uint8)
            ggml.backend_tensor_set(tensor, data, 0, nbytes)

    ggml.gguf_free(gguf_ctx)
    return ctx_w, buf_w, tensors


def build_graph(tensors, n_files):
    """Build computation graph matching C++ magika_graph."""
    buf_size = ggml.tensor_overhead() * ggml.DEFAULT_GRAPH_SIZE() + ggml.graph_overhead()
    ctx = ggml.context_init(buf_size, no_alloc=True)
    gf = ggml.new_graph(ctx)

    inp = ggml.new_tensor_3d(ctx, ggml.Type.F32, 257, INP_BYTES, n_files)
    ggml.set_name(inp, "input")
    ggml.set_input(inp)

    # dense
    cur = ggml.mul_mat(ctx, tensors["dense/kernel:0"], inp)
    cur = ggml.add(ctx, cur, tensors["dense/bias:0"])  # [128, 1536, n_files]
    cur = ggml.gelu(ctx, cur)

    # reshape
    cur = ggml.reshape_3d(ctx, cur, 512, 384, n_files)  # [512, 384, n_files]
    cur = ggml.cont(ctx, ggml.transpose(ctx, cur))

    # layer normalization
    cur = ggml.norm(ctx, cur, F_NORM_EPS)
    cur = ggml.mul(ctx, cur, tensors["layer_normalization/gamma:0"])  # [384, 512, n_files]
    cur = ggml.add(ctx, cur, tensors["layer_normalization/beta:0"])

    # dense_1
    cur = ggml.cont(ctx, ggml.transpose(ctx, cur))
    cur = ggml.mul_mat(ctx, tensors["dense_1/kernel:0"], cur)
    cur = ggml.add(ctx, cur, tensors["dense_1/bias:0"])  # [256, 384, n_files]
    cur = ggml.gelu(ctx, cur)

    # dense_2
    cur = ggml.mul_mat(ctx, tensors["dense_2/kernel:0"], cur)
    cur = ggml.add(ctx, cur, tensors["dense_2/bias:0"])  # [256, 384, n_files]
    cur = ggml.gelu(ctx, cur)

    # global_max_pooling1d
    cur = ggml.cont(ctx, ggml.transpose(ctx, cur))  # [384, 256, n_files]
    cur = ggml.pool_1d(ctx, cur, ggml.OpPool.MAX, 384, 384, 0)  # [1, 256, n_files]
    cur = ggml.reshape_2d(ctx, cur, 256, n_files)  # [256, n_files]

    # layer normalization 1
    cur = ggml.norm(ctx, cur, F_NORM_EPS)
    cur = ggml.mul(ctx, cur, tensors["layer_normalization_1/gamma:0"])  # [256, n_files]
    cur = ggml.add(ctx, cur, tensors["layer_normalization_1/beta:0"])

    # target_label
    cur = ggml.mul_mat(ctx, tensors["target_label/kernel:0"], cur)
    cur = ggml.add(ctx, cur, tensors["target_label/bias:0"])  # [n_label, n_files]
    cur = ggml.soft_max(ctx, cur)  # [n_label, n_files]
    ggml.set_name(cur, "target_label_probs")
    ggml.set_output(cur)

    ggml.build_forward_expand(gf, cur)
    return gf, ctx


def preprocess_file(fpath):
    """Read file beg/mid/end and convert to one-hot encoding."""
    fsize = os.path.getsize(fpath)
    buf = np.full(INP_BYTES, PADDING_TOKEN, dtype=np.int32)

    with open(fpath, "rb") as f:
        # read beg
        beg_data = f.read(BEG_SIZE)
        buf[:len(beg_data)] = np.frombuffer(beg_data, dtype=np.uint8)

        # read mid
        mid_offs = max(0, (fsize - MID_SIZE) // 2)
        f.seek(mid_offs)
        mid_data = f.read(MID_SIZE)
        n_read = len(mid_data)
        mid_start = BEG_SIZE + (MID_SIZE // 2) - n_read // 2
        buf[mid_start:mid_start + n_read] = np.frombuffer(mid_data, dtype=np.uint8)

        # read end
        end_offs = max(0, fsize - END_SIZE)
        f.seek(end_offs)
        end_data = f.read(END_SIZE)
        n_read = len(end_data)
        end_start = BEG_SIZE + MID_SIZE + END_SIZE - n_read
        buf[end_start:end_start + n_read] = np.frombuffer(end_data, dtype=np.uint8)

    # convert to one-hot (vectorized)
    one_hot = np.zeros((INP_BYTES, 257), dtype=np.float32)
    one_hot[np.arange(INP_BYTES), buf] = 1.0
    return one_hot.ravel()


def evaluate(backend, tensors, file_paths):
    """Run inference and print top-5 results for each file."""
    n_files = len(file_paths)
    gf, ctx_graph = build_graph(tensors, n_files)

    allocr = ggml.gallocr_new(ggml.backend_get_default_buffer_type(backend))
    ggml.gallocr_reserve(allocr, gf)
    ggml.gallocr_alloc_graph(allocr, gf)

    inp = ggml.graph_get_tensor(gf, "input")
    inp_size = 257 * INP_BYTES

    for i, fpath in enumerate(file_paths):
        one_hot = preprocess_file(fpath)
        ggml.backend_tensor_set(inp, one_hot, inp_size * i * 4, inp_size * 4)

    if ggml.backend_is_cpu(backend):
        ggml.backend_cpu_set_n_threads(backend, 4)
    ggml.backend_graph_compute(backend, gf)

    target_probs = ggml.graph_get_tensor(gf, "target_label_probs")

    for i, fpath in enumerate(file_paths):
        probs = np.empty(N_LABEL, dtype=np.float32)
        ggml.backend_tensor_get(target_probs, probs, N_LABEL * i * 4, N_LABEL * 4)

        idx = np.argsort(probs)[::-1]
        top_n = 5
        parts = []
        for j in range(top_n):
            parts.append(f"{MAGIKA_LABELS[idx[j]]} ({probs[idx[j]] * 100:.2f}%)")
        print(f"{fpath:<30s}: {' '.join(parts)}")

    ggml.gallocr_free(allocr)
    ggml.context_free(ctx_graph)


def main():
    parser = argparse.ArgumentParser(description="Magika file type detector")
    parser.add_argument("model_path", help="Path to magika GGUF model")
    parser.add_argument("files", nargs="+", help="Files to classify")
    parser.add_argument("-b", "--backend", default="cpu", choices=["cpu", "metal"],
                        help="Backend to use (default: cpu)")
    args = parser.parse_args()

    if args.backend == "metal":
        backend = ggml.backend_metal_init()
    else:
        backend = ggml.backend_cpu_init()

    ctx_w, buf_w, tensors = load_model(args.model_path, backend)

    evaluate(backend, tensors, args.files)

    ggml.context_free(ctx_w)
    ggml.backend_buffer_free(buf_w)
    ggml.backend_free(backend)


if __name__ == "__main__":
    main()

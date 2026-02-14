"""Magika file type detector using ggbond Session API.

Usage:
    python examples/ggbond_magika.py <model.gguf> <file1> [file2 ...] [--backend cpu|metal]
"""

import argparse
import os

import numpy as np

import ggbond
from ggbond import ggml
from ggbond.context import Context


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

BEG_SIZE = 512
MID_SIZE = 512
END_SIZE = 512
N_LABEL = 113
F_NORM_EPS = 0.001
PADDING_TOKEN = 256
INP_BYTES = BEG_SIZE + MID_SIZE + END_SIZE


def build_graph(tensors, n_files):
    """Build computation graph matching C++ magika_graph."""
    ctx = Context.for_graph()
    gf = ctx.new_graph()

    inp = ctx.new_tensor(ggml.Type.F32, 257, INP_BYTES, n_files, name="input")
    ggml.set_input(inp)

    # dense
    cur = ggml.mul_mat(ctx.raw, tensors["dense/kernel:0"], inp)
    cur = ggml.add(ctx.raw, cur, tensors["dense/bias:0"])
    cur = ggml.gelu(ctx.raw, cur)

    # reshape
    cur = ggml.reshape_3d(ctx.raw, cur, 512, 384, n_files)
    cur = ggml.cont(ctx.raw, ggml.transpose(ctx.raw, cur))

    # layer normalization
    cur = ggml.norm(ctx.raw, cur, F_NORM_EPS)
    cur = ggml.mul(ctx.raw, cur, tensors["layer_normalization/gamma:0"])
    cur = ggml.add(ctx.raw, cur, tensors["layer_normalization/beta:0"])

    # dense_1
    cur = ggml.cont(ctx.raw, ggml.transpose(ctx.raw, cur))
    cur = ggml.mul_mat(ctx.raw, tensors["dense_1/kernel:0"], cur)
    cur = ggml.add(ctx.raw, cur, tensors["dense_1/bias:0"])
    cur = ggml.gelu(ctx.raw, cur)

    # dense_2
    cur = ggml.mul_mat(ctx.raw, tensors["dense_2/kernel:0"], cur)
    cur = ggml.add(ctx.raw, cur, tensors["dense_2/bias:0"])
    cur = ggml.gelu(ctx.raw, cur)

    # global_max_pooling1d
    cur = ggml.cont(ctx.raw, ggml.transpose(ctx.raw, cur))
    cur = ggml.pool_1d(ctx.raw, cur, ggml.OpPool.MAX, 384, 384, 0)
    cur = ggml.reshape_2d(ctx.raw, cur, 256, n_files)

    # layer normalization 1
    cur = ggml.norm(ctx.raw, cur, F_NORM_EPS)
    cur = ggml.mul(ctx.raw, cur, tensors["layer_normalization_1/gamma:0"])
    cur = ggml.add(ctx.raw, cur, tensors["layer_normalization_1/beta:0"])

    # target_label
    cur = ggml.mul_mat(ctx.raw, tensors["target_label/kernel:0"], cur)
    cur = ggml.add(ctx.raw, cur, tensors["target_label/bias:0"])
    cur = ggml.soft_max(ctx.raw, cur)
    ggml.set_name(cur, "target_label_probs")
    ggml.set_output(cur)

    ggml.build_forward_expand(gf, cur)
    return gf, ctx


def preprocess_file(fpath):
    """Read file beg/mid/end and convert to one-hot encoding."""
    fsize = os.path.getsize(fpath)
    buf = np.full(INP_BYTES, PADDING_TOKEN, dtype=np.int32)

    with open(fpath, "rb") as f:
        beg_data = f.read(BEG_SIZE)
        buf[:len(beg_data)] = np.frombuffer(beg_data, dtype=np.uint8)

        mid_offs = max(0, (fsize - MID_SIZE) // 2)
        f.seek(mid_offs)
        mid_data = f.read(MID_SIZE)
        n_read = len(mid_data)
        mid_start = BEG_SIZE + (MID_SIZE // 2) - n_read // 2
        buf[mid_start:mid_start + n_read] = np.frombuffer(mid_data, dtype=np.uint8)

        end_offs = max(0, fsize - END_SIZE)
        f.seek(end_offs)
        end_data = f.read(END_SIZE)
        n_read = len(end_data)
        end_start = BEG_SIZE + MID_SIZE + END_SIZE - n_read
        buf[end_start:end_start + n_read] = np.frombuffer(end_data, dtype=np.uint8)

    one_hot = np.zeros((INP_BYTES, 257), dtype=np.float32)
    one_hot[np.arange(INP_BYTES), buf] = 1.0
    return one_hot.ravel()


def main():
    parser = argparse.ArgumentParser(description="Magika file type detector (ggbond Session)")
    parser.add_argument("model_path", help="Path to magika GGUF model")
    parser.add_argument("files", nargs="+", help="Files to classify")
    parser.add_argument("-b", "--backend", default="cpu", choices=["cpu", "metal"],
                        help="Backend to use (default: cpu)")
    args = parser.parse_args()

    with ggbond.Session(args.backend) as s:
        ctx_w, tensors = s.load_gguf(args.model_path)

        n_files = len(args.files)
        gf, ctx_graph = build_graph(tensors, n_files)

        inp_size = 257 * INP_BYTES
        all_inputs = np.empty(inp_size * n_files, dtype=np.float32)
        for i, fpath in enumerate(args.files):
            all_inputs[inp_size * i : inp_size * (i + 1)] = preprocess_file(fpath)

        s.run(gf, inputs={"input": all_inputs})

        for i, fpath in enumerate(args.files):
            probs = s.get_slice(
                gf, "target_label_probs",
                offset=N_LABEL * i * 4, count=N_LABEL,
            )
            idx = np.argsort(probs)[::-1]
            parts = [
                f"{MAGIKA_LABELS[idx[j]]} ({probs[idx[j]] * 100:.2f}%)"
                for j in range(5)
            ]
            print(f"{fpath:<30s}: {' '.join(parts)}")

        ctx_graph.close()


if __name__ == "__main__":
    main()

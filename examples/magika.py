"""Magika file type detector using ggbond Tensor API.

Usage:
    python examples/magika.py <model.gguf> <file1> [file2 ...]
"""

import argparse
import os
from dataclasses import dataclass

import numpy as np

from ggbond.ggml import Backend, Context, GAllocr, GGUF, Tensor, F32, POOL_MAX


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


def _read_exact(f, size: int, what: str) -> bytes:
    data = f.read(size)
    if len(data) != size:
        raise ValueError(f"truncated model while reading {what}: expected {size} bytes, got {len(data)}")
    return data


@dataclass
class Prediction:
    """Single file type prediction result."""
    label: str
    score: float

    def __str__(self):
        return f"{self.label} ({self.score * 100:.2f}%)"


class Magika:
    """Magika file type detector."""

    BEG_SIZE = 512
    MID_SIZE = 512
    END_SIZE = 512
    N_LABEL = 113
    F_NORM_EPS = 0.001
    PADDING_TOKEN = 256
    INP_BYTES = BEG_SIZE + MID_SIZE + END_SIZE

    def __init__(self, model_path: str):
        self._backend = Backend.cpu_init()
        self._backend.set_n_threads(os.cpu_count() or 4)
        # GGUF holds the parsed header; its .context holds the weight metadata.
        # GGUF doesn't read tensor data -- allocate backend buffers, then upload
        # each tensor's bytes straight from the file (standard ggml pattern).
        self._gguf = GGUF.from_file(model_path, no_alloc=True)
        self._backend.alloc_ctx_tensors(self._gguf.context)
        with open(model_path, "rb") as f:
            for i in range(self._gguf.n_tensors):
                name = self._gguf.get_tensor_name(i)
                t = self._gguf.context.get_tensor(name)
                if t is None:
                    raise ValueError(f"GGUF tensor metadata is missing '{name}'")
                f.seek(self._gguf.data_offset + self._gguf.get_tensor_offset(i))
                t.set_raw(_read_exact(f, t.nbytes, f"tensor '{name}' data"))
        self._weights = {
            self._gguf.get_tensor_name(i): self._gguf.context.get_tensor(
                self._gguf.get_tensor_name(i)
            )
            for i in range(self._gguf.n_tensors)
        }

    def predict(self, files: list[str], *, top_k: int = 5) -> list[list[Prediction]]:
        """Predict file types for a list of files.

        Returns a list of top-k predictions per file.
        """
        n_files = len(files)
        input_data = np.concatenate([self._preprocess_file(f) for f in files])

        # Weights live in self._gguf.context (already allocated). Inputs and
        # intermediates live in this compute context; GAllocr allocates them.
        with Context(mem_size=1 << 20, no_alloc=True) as ctx:
            # Rebind weights to this context so op-result nodes land here, not in
            # the GGUF metadata context (which is sized only for the weights). The
            # ggml_tensor pointers are shared; a graph may span both contexts.
            w = {name: Tensor(ctx, t.ptr) for name, t in self._weights.items()}

            inp = ctx.new_tensor_3d(F32, 257, self.INP_BYTES, n_files, name="input").set_input()

            # dense
            cur = w["dense/kernel:0"].mul_mat(inp).add(w["dense/bias:0"]).gelu()

            # reshape + transpose
            cur = cur.reshape_3d(512, 384, n_files).transpose().cont()

            # layer normalization
            cur = cur.norm(eps=self.F_NORM_EPS).mul(w["layer_normalization/gamma:0"]).add(w["layer_normalization/beta:0"])

            # dense_1
            cur = w["dense_1/kernel:0"].mul_mat(cur.transpose().cont()).add(w["dense_1/bias:0"]).gelu()

            # dense_2
            cur = w["dense_2/kernel:0"].mul_mat(cur).add(w["dense_2/bias:0"]).gelu()

            # global_max_pooling1d
            cur = cur.transpose().cont().pool_1d(POOL_MAX, 384, 384, 0).reshape_2d(256, n_files)

            # layer normalization 1
            cur = cur.norm(eps=self.F_NORM_EPS).mul(w["layer_normalization_1/gamma:0"]).add(w["layer_normalization_1/beta:0"])

            # target_label
            out = w["target_label/kernel:0"].mul_mat(cur).add(w["target_label/bias:0"]).soft_max()
            out.set_output()

            graph = ctx.new_graph().build_forward_expand(out)
            with GAllocr.from_backend(self._backend) as alloc:
                alloc.alloc_graph(graph)
                inp.set(input_data.reshape(n_files, self.INP_BYTES, 257))
                self._backend.compute(graph)
                self._backend.synchronize()
                all_probs = out.get(
                    np.empty(tuple(reversed(out.ne)), dtype=np.float32)
                ).reshape(n_files, self.N_LABEL)

        results = []
        for probs in all_probs:
            top_indices = np.argsort(probs)[::-1][:top_k]
            results.append([
                Prediction(MAGIKA_LABELS[i], probs[i]) for i in top_indices
            ])
        return results

    def close(self):
        # gguf.context (weights) must outlive any graph referencing it; both it
        # and the gguf_context are owned separately from the backend.
        self._gguf.context.close()
        self._gguf.close()
        self._backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @staticmethod
    def _read_segment(f, offset: int, size: int) -> np.ndarray:
        """Read a segment of bytes from file at the given offset."""
        f.seek(offset)
        data = f.read(size)
        return np.frombuffer(data, dtype=np.uint8)

    @classmethod
    def _preprocess_file(cls, fpath: str) -> np.ndarray:
        """Read file beg/mid/end and convert to one-hot encoding."""
        fsize = os.path.getsize(fpath)
        buf = np.full(cls.INP_BYTES, cls.PADDING_TOKEN, dtype=np.int32)

        with open(fpath, "rb") as f:
            # beginning
            beg = cls._read_segment(f, 0, cls.BEG_SIZE)
            buf[:len(beg)] = beg

            # middle (centered)
            mid_offset = max(0, (fsize - cls.MID_SIZE) // 2)
            mid = cls._read_segment(f, mid_offset, cls.MID_SIZE)
            mid_start = cls.BEG_SIZE + cls.MID_SIZE // 2 - len(mid) // 2
            buf[mid_start:mid_start + len(mid)] = mid

            # end (right-aligned)
            end_offset = max(0, fsize - cls.END_SIZE)
            end = cls._read_segment(f, end_offset, cls.END_SIZE)
            end_start = cls.BEG_SIZE + cls.MID_SIZE + cls.END_SIZE - len(end)
            buf[end_start:end_start + len(end)] = end

        one_hot = np.zeros((cls.INP_BYTES, 257), dtype=np.float32)
        one_hot[np.arange(cls.INP_BYTES), buf] = 1.0
        return one_hot.ravel()


def _expand_paths(paths: list[str]) -> list[str]:
    """Expand directories to their contained files (non-recursive)."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(
                os.path.join(p, name)
                for name in sorted(os.listdir(p))
                if os.path.isfile(os.path.join(p, name))
            )
        else:
            files.append(p)
    return files


def main():
    parser = argparse.ArgumentParser(description="Magika file type detector")
    parser.add_argument("model_path", help="Path to magika GGUF model")
    parser.add_argument("files", nargs="+", help="Files or directories to classify")
    args = parser.parse_args()

    files = _expand_paths(args.files)
    if not files:
        print("No files to classify.")
        return

    with Magika(args.model_path) as magika:
        results = magika.predict(files)
        for fpath, preds in zip(files, results):
            top_str = " ".join(str(p) for p in preds)
            print(f"{fpath:<30s}: {top_str}")


if __name__ == "__main__":
    main()

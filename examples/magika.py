"""Magika file type detector using the ggbond high-level API.

Usage:
    python examples/magika.py <model.gguf> <file1> [file2 ...]
"""

import argparse
import os
from dataclasses import dataclass

import numpy as np

from ggbond import PoolOp, Session


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
        self._sess = Session()
        try:
            # Every tensor of the file becomes a variable, kept until the session closes.
            self._weights, _ = self._sess.load_gguf(model_path)
        except BaseException:
            self._sess.close()
            raise

    def predict(self, files: list[str], *, top_k: int = 5) -> list[list[Prediction]]:
        """Predict file types for a list of files.

        Returns a list of top-k predictions per file.
        """
        n_files = len(files)
        input_data = np.stack([self._preprocess_file(f) for f in files])
        w = self._weights

        # The shapes depend on the batch size, so each call builds its own graph.
        with self._sess.graph() as g:
            # numpy (n_files, INP_BYTES, 257) is ggml (257, INP_BYTES, n_files)
            inp = g.input(input_data, name="input")

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
            cur = cur.transpose().cont().pool_1d(PoolOp.MAX, 384, 384, 0).reshape_2d(256, n_files)

            # layer normalization 1
            cur = cur.norm(eps=self.F_NORM_EPS).mul(w["layer_normalization_1/gamma:0"]).add(w["layer_normalization_1/beta:0"])

            # target_label
            out = w["target_label/kernel:0"].mul_mat(cur).add(w["target_label/bias:0"]).soft_max()

            all_probs = g.run(out).reshape(n_files, self.N_LABEL)

        results = []
        for probs in all_probs:
            top_indices = np.argsort(probs)[::-1][:top_k]
            results.append([
                Prediction(MAGIKA_LABELS[i], probs[i]) for i in top_indices
            ])
        return results

    def close(self):
        self._sess.close()

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
        """Read file beg/mid/end and convert to one-hot encoding, shape (INP_BYTES, 257)."""
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
        return one_hot


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

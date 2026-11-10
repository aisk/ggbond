"""Regenerate ggbond/ggml/ops.pyi from the op tables.

Run after adding or changing an entry in the op tables of
``ggbond/ggml/tensor.py``::

    python scripts/gen_stubs.py
"""

from pathlib import Path

from ggbond.ggml.tensor import render_ops_stub

STUB = Path(__file__).resolve().parent.parent / "ggbond" / "ggml" / "ops.pyi"


def main() -> None:
    STUB.write_text(render_ops_stub())
    print(f"wrote {STUB}")


if __name__ == "__main__":
    main()

"""Holders for the generated op methods.

:class:`TensorOps` and :class:`ContextOps` are empty at import time. The op
tables in :mod:`ggbond.ggml.tensor` install one method per ggml op onto each of
them, and :class:`Tensor` / :class:`Context` inherit those methods.

They live in their own module so that ``ops.pyi`` can describe exactly the
generated methods to type checkers and IDEs, which cannot see methods created
with ``exec``. Regenerate the stub with ``python scripts/gen_stubs.py`` after
changing an op table; a test fails when it is out of date.
"""

from __future__ import annotations


class TensorOps:
    """Every ggml op as ``a.op(...)``, sugar for ``a.ctx.op(a, ...)``."""

    __slots__ = ()


class ContextOps:
    """Every ggml op as ``ctx.op(a, ...)``, i.e. ``ggml_op(ctx, a, ...)``."""

    __slots__ = ()

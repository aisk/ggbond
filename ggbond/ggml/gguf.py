"""Thin OO wrapper around ``gguf_context``.

A :class:`GGUF` wraps the ``gguf_context`` returned by ``gguf_init_from_file``
and exposes its readers ~1:1 as Python methods. It does *not* read tensor data
itself -- ``data_offset`` / ``get_tensor_offset`` / ``get_tensor_name`` give the
caller everything needed to seek into the file and upload weights via
``Tensor.set`` / ``ggml_backend_tensor_set``.
"""

from __future__ import annotations

import ctypes
from typing import Optional

import ggml as _ggml

from .errors import nonnull
from .context import Context
from .types import DType


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else value


class GGUF:
    """Owns a ``gguf_context`` (the parsed GGUF header / metadata).

    :meth:`from_file` optionally also creates the ``ggml_context`` that holds the
    tensor metadata, exposed as :attr:`context`. Ownership is split: :meth:`close`
    frees *only* the ``gguf_context`` (via ``gguf_free``); the data :attr:`context`
    is an independently owned :class:`Context` that holds the tensors and must
    outlive any graph referencing them -- close it yourself.
    """

    def __init__(self, ptr, *, context: Optional[Context] = None, owns: bool = True):
        self.ptr = nonnull(ptr, "gguf_context")
        self.context = context
        self._owns = owns
        self._closed = False

    @classmethod
    def from_file(cls, path: str, *, no_alloc: bool = True, create_context: bool = True) -> "GGUF":
        """Parse ``path`` via ``gguf_init_from_file``.

        When ``create_context`` is True (the default), the metadata ``ggml_context``
        filled in by ggml is wrapped as :attr:`context`; pass False to read only
        the header (``params.ctx = NULL``).
        """
        if create_context:
            holder = _ggml.ggml_context_p_ctypes()
            params = _ggml.gguf_init_params(no_alloc=no_alloc, ctx=ctypes.pointer(holder))
        else:
            holder = None
            params = _ggml.gguf_init_params(no_alloc=no_alloc, ctx=None)
        ptr = nonnull(
            _ggml.gguf_init_from_file(str(path).encode(), params),
            "gguf_init_from_file",
        )
        context = Context.from_ptr(holder, no_alloc=no_alloc, owns=True) if create_context else None
        return cls(ptr, context=context, owns=True)

    @classmethod
    def from_ptr(cls, ptr, *, context: Optional[Context] = None, owns: bool = True) -> "GGUF":
        """Wrap an externally created ``gguf_context``."""
        return cls(ptr, context=context, owns=owns)

    @classmethod
    def init_empty(cls) -> "GGUF":
        """Create an empty ``gguf_context`` via ``gguf_init_empty``."""
        return cls(nonnull(_ggml.gguf_init_empty(), "gguf_init_empty"), owns=True)

    # -- header -------------------------------------------------------------

    @property
    def version(self) -> int:
        return _ggml.gguf_get_version(self.ptr)

    @property
    def alignment(self) -> int:
        return _ggml.gguf_get_alignment(self.ptr)

    @property
    def data_offset(self) -> int:
        return _ggml.gguf_get_data_offset(self.ptr)

    # -- key/value metadata -------------------------------------------------

    @property
    def n_kv(self) -> int:
        return _ggml.gguf_get_n_kv(self.ptr)

    def find_key(self, key: str) -> int:
        return _ggml.gguf_find_key(self.ptr, key.encode())

    def get_key(self, i: int) -> str:
        return _decode(_ggml.gguf_get_key(self.ptr, i))

    def get_kv_type(self, i: int) -> int:
        return _ggml.gguf_get_kv_type(self.ptr, i)

    def get_val_u8(self, i: int) -> int:
        return _ggml.gguf_get_val_u8(self.ptr, i)

    def get_val_i8(self, i: int) -> int:
        return _ggml.gguf_get_val_i8(self.ptr, i)

    def get_val_u16(self, i: int) -> int:
        return _ggml.gguf_get_val_u16(self.ptr, i)

    def get_val_i16(self, i: int) -> int:
        return _ggml.gguf_get_val_i16(self.ptr, i)

    def get_val_u32(self, i: int) -> int:
        return _ggml.gguf_get_val_u32(self.ptr, i)

    def get_val_i32(self, i: int) -> int:
        return _ggml.gguf_get_val_i32(self.ptr, i)

    def get_val_u64(self, i: int) -> int:
        return _ggml.gguf_get_val_u64(self.ptr, i)

    def get_val_i64(self, i: int) -> int:
        return _ggml.gguf_get_val_i64(self.ptr, i)

    def get_val_f32(self, i: int) -> float:
        return _ggml.gguf_get_val_f32(self.ptr, i)

    def get_val_f64(self, i: int) -> float:
        return _ggml.gguf_get_val_f64(self.ptr, i)

    def get_val_bool(self, i: int) -> bool:
        return _ggml.gguf_get_val_bool(self.ptr, i)

    def get_val_str(self, i: int) -> str:
        return _decode(_ggml.gguf_get_val_str(self.ptr, i))

    def get_arr_type(self, i: int) -> int:
        return _ggml.gguf_get_arr_type(self.ptr, i)

    def get_arr_n(self, i: int) -> int:
        return _ggml.gguf_get_arr_n(self.ptr, i)

    def get_arr_str(self, i: int, j: int) -> str:
        return _decode(_ggml.gguf_get_arr_str(self.ptr, i, j))

    # -- tensors ------------------------------------------------------------

    @property
    def n_tensors(self) -> int:
        return _ggml.gguf_get_n_tensors(self.ptr)

    def find_tensor(self, name: str) -> int:
        return _ggml.gguf_find_tensor(self.ptr, name.encode())

    def get_tensor_name(self, i: int) -> str:
        return _decode(_ggml.gguf_get_tensor_name(self.ptr, i))

    def get_tensor_offset(self, i: int) -> int:
        return _ggml.gguf_get_tensor_offset(self.ptr, i)

    def get_tensor_type(self, i: int) -> "DType":
        return DType(_ggml.gguf_get_tensor_type(self.ptr, i))

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        if not self._closed and self.ptr is not None:
            if self._owns:
                _ggml.gguf_free(self.ptr)
            self._closed = True
            self.ptr = None

    def __enter__(self) -> "GGUF":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"GGUF(n_kv={self.n_kv}, n_tensors={self.n_tensors})"

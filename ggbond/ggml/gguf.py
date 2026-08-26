"""Thin OO wrapper around ``gguf_context``.

A :class:`GGUF` wraps the ``gguf_context`` returned by ``gguf_init_from_file``
and exposes its readers and writers ~1:1 as Python methods. It does *not* read
tensor data itself -- ``data_offset`` / ``get_tensor_offset`` /
``get_tensor_name`` give the caller everything needed to seek into the file and
upload weights via ``Tensor.set`` / ``ggml_backend_tensor_set``.

The typed ``get_val_*`` / ``set_val_*`` accessors are generated from
:data:`_KV_TYPES`; :meth:`GGUF.value` dispatches over them on the stored type
when you do not want to write the switch yourself.
"""

from __future__ import annotations

import ctypes
from typing import Optional, Union

import numpy as np

import ggml as _ggml

from .errors import GGMLError, nonnull
from .context import Context
from .tensor import Tensor
from .types import DType

# GGUF metadata value types (mirror ggml's ``enum gguf_type``).
TYPE_UINT8 = _ggml.GGUF_TYPE_UINT8
TYPE_INT8 = _ggml.GGUF_TYPE_INT8
TYPE_UINT16 = _ggml.GGUF_TYPE_UINT16
TYPE_INT16 = _ggml.GGUF_TYPE_INT16
TYPE_UINT32 = _ggml.GGUF_TYPE_UINT32
TYPE_INT32 = _ggml.GGUF_TYPE_INT32
TYPE_FLOAT32 = _ggml.GGUF_TYPE_FLOAT32
TYPE_BOOL = _ggml.GGUF_TYPE_BOOL
TYPE_STRING = _ggml.GGUF_TYPE_STRING
TYPE_ARRAY = _ggml.GGUF_TYPE_ARRAY
TYPE_UINT64 = _ggml.GGUF_TYPE_UINT64
TYPE_INT64 = _ggml.GGUF_TYPE_INT64
TYPE_FLOAT64 = _ggml.GGUF_TYPE_FLOAT64

# suffix -> (gguf type, python coercion, numpy dtype for arrays of this type).
# ``str`` and ``bool`` need their own conversions and are handled explicitly.
_KV_TYPES = {
    "u8": (TYPE_UINT8, int, np.uint8),
    "i8": (TYPE_INT8, int, np.int8),
    "u16": (TYPE_UINT16, int, np.uint16),
    "i16": (TYPE_INT16, int, np.int16),
    "u32": (TYPE_UINT32, int, np.uint32),
    "i32": (TYPE_INT32, int, np.int32),
    "u64": (TYPE_UINT64, int, np.uint64),
    "i64": (TYPE_INT64, int, np.int64),
    "f32": (TYPE_FLOAT32, float, np.float32),
    "f64": (TYPE_FLOAT64, float, np.float64),
    "bool": (TYPE_BOOL, bool, np.int8),
}

# gguf type -> the ``get_val_*`` method name that reads it.
_TYPE_READER = {gguf_type: f"get_val_{suffix}" for suffix, (gguf_type, _, _) in _KV_TYPES.items()}
_TYPE_READER[TYPE_STRING] = "get_val_str"

# gguf type -> numpy dtype, for reading array values in bulk.
_TYPE_NUMPY = {gguf_type: dtype for _, (gguf_type, _, dtype) in _KV_TYPES.items()}


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else value


def type_name(gguf_type: int) -> str:
    """Return ggml's name for a gguf value type (``gguf_type_name``)."""
    return _decode(_ggml.gguf_type_name(int(gguf_type)))


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
        # Buffers handed to gguf by pointer, kept alive until this is closed.
        self._retained = []

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

    def get_val_str(self, i: int) -> str:
        return _decode(_ggml.gguf_get_val_str(self.ptr, i))

    def set_val_str(self, key: str, value: str) -> None:
        _ggml.gguf_set_val_str(self.ptr, key.encode(), str(value).encode())

    def get_arr_type(self, i: int) -> int:
        return _ggml.gguf_get_arr_type(self.ptr, i)

    def get_arr_n(self, i: int) -> int:
        return _ggml.gguf_get_arr_n(self.ptr, i)

    def get_arr_str(self, i: int, j: int) -> str:
        return _decode(_ggml.gguf_get_arr_str(self.ptr, i, j))

    def get_arr_data(self, i: int) -> "np.ndarray":
        """Return the numeric array stored at key index ``i`` as a numpy copy.

        Raises :class:`GGMLError` for string arrays -- read those element by
        element with :meth:`get_arr_str`.
        """
        element_type = self.get_arr_type(i)
        if element_type == TYPE_STRING:
            raise GGMLError(
                f"key {self.get_key(i)!r} holds a string array; use get_arr_str"
            )
        try:
            dtype = np.dtype(_TYPE_NUMPY[element_type])
        except KeyError:
            raise GGMLError(
                f"key {self.get_key(i)!r} has unsupported array type "
                f"{type_name(element_type)}"
            ) from None
        n = self.get_arr_n(i)
        address = nonnull(_ggml.gguf_get_arr_data(self.ptr, i), "gguf_get_arr_data")
        buffer = (ctypes.c_char * (n * dtype.itemsize)).from_address(address)
        values = np.frombuffer(buffer, dtype=dtype, count=n).copy()
        return values.astype(np.bool_) if element_type == TYPE_BOOL else values

    def set_arr_data(self, key: str, gguf_type: int, values) -> None:
        """Store ``values`` as an array of ``gguf_type`` under ``key``."""
        try:
            dtype = np.dtype(_TYPE_NUMPY[gguf_type])
        except KeyError:
            raise GGMLError(f"unsupported gguf array type {gguf_type}") from None
        array = np.ascontiguousarray(values, dtype=dtype)
        _ggml.gguf_set_arr_data(
            self.ptr,
            key.encode(),
            int(gguf_type),
            array.ctypes.data_as(ctypes.c_void_p),
            len(array),
        )

    def set_arr_str(self, key: str, values) -> None:
        """Store ``values`` as an array of strings under ``key``."""
        encoded = [str(value).encode() for value in values]
        array = (ctypes.c_char_p * len(encoded))(*encoded)
        _ggml.gguf_set_arr_str(self.ptr, key.encode(), array, len(encoded))

    def remove_key(self, key: str) -> int:
        """Remove ``key``, returning the index it had (or -1 if absent)."""
        return _ggml.gguf_remove_key(self.ptr, key.encode())

    # -- typed metadata by key ----------------------------------------------

    def key_index(self, key: "Union[str, int]") -> int:
        """Resolve ``key`` to a key index, raising :class:`KeyError` if absent."""
        if isinstance(key, int):
            if not (0 <= key < self.n_kv):
                raise KeyError(key)
            return key
        index = self.find_key(key)
        if index < 0:
            raise KeyError(key)
        return index

    def value(self, key: "Union[str, int]"):
        """Return the value stored under ``key``, dispatching on its type.

        Scalars come back as Python ``int`` / ``float`` / ``bool`` / ``str``,
        numeric arrays as a numpy array, and string arrays as a list of ``str``.
        The typed ``get_val_*`` accessors remain the exact-semantics path.
        """
        index = self.key_index(key)
        kv_type = self.get_kv_type(index)
        if kv_type == TYPE_ARRAY:
            if self.get_arr_type(index) == TYPE_STRING:
                return [self.get_arr_str(index, j) for j in range(self.get_arr_n(index))]
            return self.get_arr_data(index)
        try:
            reader = _TYPE_READER[kv_type]
        except KeyError:
            raise GGMLError(
                f"key {self.get_key(index)!r} has unsupported type {kv_type}"
            ) from None
        return getattr(self, reader)(index)

    def keys(self):
        """Iterate over every metadata key, in file order."""
        return (self.get_key(i) for i in range(self.n_kv))

    def items(self):
        """Iterate over ``(key, value)`` for every metadata entry."""
        for i in range(self.n_kv):
            yield self.get_key(i), self.value(i)

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

    def get_tensor_size(self, i: int) -> int:
        return _ggml.gguf_get_tensor_size(self.ptr, i)

    def get_tensor_type(self, i: int) -> "DType":
        return DType(_ggml.gguf_get_tensor_type(self.ptr, i))

    def add_tensor(self, tensor: "Tensor") -> None:
        """Add ``tensor``'s name, shape and dtype to the header.

        The tensor's data is written by :meth:`write_to_file` from wherever the
        tensor points, so it must stay alive and allocated until then. Use
        :meth:`set_tensor_data` to point an entry at a different buffer.
        """
        _ggml.gguf_add_tensor(self.ptr, tensor.ptr)

    def set_tensor_data(self, name: str, data) -> None:
        """Point the entry ``name`` at a copy of ``data`` (a bytes-like buffer).

        gguf only stores the pointer, so the copy is retained by this
        :class:`GGUF` until it is closed.
        """
        source = memoryview(data).cast("B")
        copy = (ctypes.c_char * source.nbytes).from_buffer_copy(source)
        self._retained.append(copy)
        _ggml.gguf_set_tensor_data(
            self.ptr, name.encode(), ctypes.cast(copy, ctypes.c_void_p)
        )

    def set_tensor_type(self, name: str, dtype: "DType") -> None:
        _ggml.gguf_set_tensor_type(self.ptr, name.encode(), int(dtype))

    # -- writing ------------------------------------------------------------

    def write_to_file(self, path: str, *, only_meta: bool = False) -> None:
        """Write this header (and, unless ``only_meta``, the tensor data) out."""
        if not _ggml.gguf_write_to_file(self.ptr, str(path).encode(), only_meta):
            raise GGMLError(f"gguf_write_to_file failed for {path!r}")

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        if not self._closed and self.ptr is not None:
            if self._owns:
                _ggml.gguf_free(self.ptr)
            self._closed = True
            self.ptr = None
            self._retained.clear()

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


def _make_getter(suffix: str, coerce):
    fn = getattr(_ggml, f"gguf_get_val_{suffix}")

    def getter(self, i: int):
        return coerce(fn(self.ptr, i))

    getter.__name__ = f"get_val_{suffix}"
    getter.__qualname__ = f"GGUF.get_val_{suffix}"
    getter.__doc__ = f"Read key index ``i`` as {suffix} (``gguf_get_val_{suffix}``)."
    return getter


def _make_setter(suffix: str, coerce):
    fn = getattr(_ggml, f"gguf_set_val_{suffix}")

    def setter(self, key: str, value) -> None:
        fn(self.ptr, key.encode(), coerce(value))

    setter.__name__ = f"set_val_{suffix}"
    setter.__qualname__ = f"GGUF.set_val_{suffix}"
    setter.__doc__ = f"Store ``value`` under ``key`` as {suffix} (``gguf_set_val_{suffix}``)."
    return setter


for _suffix, (_type, _coerce, _dtype) in _KV_TYPES.items():
    setattr(GGUF, f"get_val_{_suffix}", _make_getter(_suffix, _coerce))
    setattr(GGUF, f"set_val_{_suffix}", _make_setter(_suffix, _coerce))

del _suffix, _type, _coerce, _dtype

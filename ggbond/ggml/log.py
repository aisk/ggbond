"""ggml's log callback (``ggml_log_set``)."""

from __future__ import annotations

from enum import IntEnum
from typing import Callable, Optional

import ggml as _ggml


class LogLevel(IntEnum):
    """Severity passed to a log callback (``enum ggml_log_level``)."""

    NONE = _ggml.GGML_LOG_LEVEL_NONE
    DEBUG = _ggml.GGML_LOG_LEVEL_DEBUG
    INFO = _ggml.GGML_LOG_LEVEL_INFO
    WARN = _ggml.GGML_LOG_LEVEL_WARN
    ERROR = _ggml.GGML_LOG_LEVEL_ERROR
    CONT = _ggml.GGML_LOG_LEVEL_CONT


# ggml stores only the function pointer, so the ctypes thunk must stay alive
# for as long as it is installed.
_installed = None


def log_set(callback: "Optional[Callable[[LogLevel, str], None]]") -> None:
    """Route ggml's log output to ``callback(level, text)``.

    ``None`` restores ggml's default, which prints to stderr. To silence ggml
    entirely pass a callback that does nothing: ``log_set(lambda level, text:
    None)``. Exceptions raised by the callback are swallowed, since they cannot
    propagate through C.
    """
    global _installed
    if callback is None:
        _ggml.ggml_log_set(_ggml.ggml_log_callback(), None)
        _installed = None
        return

    def thunk(level, text, user_data):
        try:
            callback(LogLevel(level), text.decode(errors="replace") if text else "")
        except Exception:
            pass

    installed = _ggml.ggml_log_callback(thunk)
    _ggml.ggml_log_set(installed, None)
    _installed = installed

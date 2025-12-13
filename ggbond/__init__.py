"""Simple and naive GGML binding."""

import os
import sys
import importlib.util

try:
    from . import ggml
except ImportError:
    # Try to find the module in the build directory (for editable installs).
    build_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'build')
    module_path = None

    for root, dirs, files in os.walk(build_dir):
        if 'ggml.so' in files:
            module_path = os.path.join(root, 'ggml.so')
            break

    if module_path:
        spec = importlib.util.spec_from_file_location("ggml", module_path)
        ggml = importlib.util.module_from_spec(spec)
        sys.modules['ggbond.ggml'] = ggml
        spec.loader.exec_module(ggml)
    else:
        raise ImportError("Could not find compiled ggml module")

__version__ = "0.1.0"
__all__ = []
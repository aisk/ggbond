import importlib
import unittest
from unittest import mock

from ggbond.ggml import Backend, GGMLError

backend_module = importlib.import_module("ggbond.ggml.backend")


class OptionalBackendErrorTests(unittest.TestCase):
    def test_cuda_binding_error_becomes_ggml_error(self):
        with mock.patch.object(
            backend_module._ggml,
            "ggml_backend_cuda_init",
            side_effect=RuntimeError("not compiled"),
        ):
            with self.assertRaisesRegex(GGMLError, "CUDA backend initialization failed"):
                Backend.cuda_init()

    def test_metal_binding_error_becomes_ggml_error(self):
        with mock.patch.object(
            backend_module._ggml,
            "ggml_backend_metal_init",
            side_effect=RuntimeError("not compiled"),
        ):
            with self.assertRaisesRegex(GGMLError, "Metal backend initialization failed"):
                Backend.metal_init()

    def test_hip_binding_error_becomes_ggml_error(self):
        with mock.patch.object(
            backend_module._ggml,
            "ggml_backend_hip_init",
            side_effect=RuntimeError("not compiled"),
            create=True,
        ):
            with self.assertRaisesRegex(GGMLError, "HIP backend initialization failed"):
                Backend.hip_init()


if __name__ == "__main__":
    unittest.main()

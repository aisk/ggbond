import unittest
from enum import IntEnum

import ggml as _ggml

from ggbond.ggml import (
    DEVICE_CPU,
    POOL_MAX,
    ROPE_TYPE_NEOX,
    Backend,
    Context,
    Device,
    DeviceType,
    F32,
    GGUF,
    GGUFType,
    I32,
    Op,
    PoolOp,
    RopeType,
)
from ggbond.ggml.gguf import TYPE_STRING


class EnumTests(unittest.TestCase):
    def test_enums_are_int_enums_with_binding_values(self):
        for enum in (Op, PoolOp, RopeType, DeviceType, GGUFType):
            self.assertTrue(issubclass(enum, IntEnum), enum)
        self.assertEqual(Op.MUL_MAT, _ggml.GGML_OP_MUL_MAT)
        self.assertEqual(Op.POOL_1D, _ggml.GGML_OP_POOL_1D)
        self.assertEqual(PoolOp.AVG, _ggml.GGML_OP_POOL_AVG)
        self.assertEqual(RopeType.IMROPE, _ggml.GGML_ROPE_TYPE_IMROPE)
        self.assertEqual(DeviceType.GPU, _ggml.GGML_BACKEND_DEVICE_TYPE_GPU)
        self.assertEqual(GGUFType.FLOAT64, _ggml.GGUF_TYPE_FLOAT64)
        self.assertNotIn("COUNT", Op.__members__)

    def test_module_constants_are_enum_members(self):
        self.assertIs(POOL_MAX, PoolOp.MAX)
        self.assertIs(ROPE_TYPE_NEOX, RopeType.NEOX)
        self.assertIs(DEVICE_CPU, DeviceType.CPU)
        self.assertIs(TYPE_STRING, GGUFType.STRING)

    def test_tensor_op_reports_an_op(self):
        with Context(1 << 20, no_alloc=True) as ctx:
            a = ctx.new_tensor_2d(F32, 3, 4)
            b = ctx.new_tensor_2d(F32, 3, 2)
            self.assertIs(a.op, Op.NONE)
            self.assertIs(a.mul_mat(b).op, Op.MUL_MAT)
            self.assertIs(a.pool_1d(PoolOp.MAX, 1, 1, 0).op, Op.POOL_1D)
            # Plain ints are still accepted where an enum is expected.
            self.assertIs(a.pool_1d(0, 1, 1, 0).op, Op.POOL_1D)
            positions = ctx.new_tensor_1d(I32, 4)
            rope_input = ctx.new_tensor_3d(F32, 8, 1, 4)
            self.assertIs(rope_input.rope(positions, 8, RopeType.NEOX).op, Op.ROPE)

    def test_device_type_is_an_enum(self):
        Backend.load_all()
        cpu = Device.by_type(DeviceType.CPU)
        self.assertIsNotNone(cpu)
        self.assertIs(cpu.type, DeviceType.CPU)
        with Backend.init_by_type(DeviceType.CPU) as backend:
            self.assertTrue(backend.is_cpu)
            self.assertEqual(backend.kind, "cpu")

    def test_gguf_types_are_enums(self):
        with GGUF.init_empty() as gguf:
            gguf.set_val_u32("n", 1)
            gguf.set_arr_data("a", GGUFType.INT32, [1, 2])
            gguf.set_arr_data("b", int(GGUFType.INT32), [3])
            self.assertIs(gguf.get_kv_type(0), GGUFType.UINT32)
            self.assertIs(gguf.get_kv_type(1), GGUFType.ARRAY)
            self.assertIs(gguf.get_arr_type(1), GGUFType.INT32)
            self.assertEqual(list(gguf.value("b")), [3])


if __name__ == "__main__":
    unittest.main()

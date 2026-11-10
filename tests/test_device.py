import unittest

from ggbond.ggml import Backend, Device, DeviceType, GGMLError


class DeviceRegistryTests(unittest.TestCase):
    def test_registry_always_reports_a_cpu_device(self):
        Backend.load_all()
        devices = Device.all()

        self.assertGreaterEqual(len(devices), 1)
        self.assertEqual(len(devices), Device.count())
        self.assertIn(DeviceType.CPU, [device.type for device in devices])

    def test_device_reports_name_description_and_memory(self):
        device = Device.by_type(DeviceType.CPU)
        self.assertIsNotNone(device)

        free, total = device.memory()
        self.assertIsInstance(device.name, str)
        self.assertIsInstance(device.description, str)
        self.assertGreater(total, 0)
        self.assertLessEqual(free, total)

    def test_device_initializes_a_backend_that_points_back(self):
        device = Device.by_type(DeviceType.CPU)

        with device.init_backend() as backend:
            self.assertTrue(backend.is_cpu)
            self.assertEqual(backend.device.name, device.name)
            self.assertEqual(device.buffer_type.name, backend.buffer_type.name)

    def test_unknown_names_and_indices_are_reported(self):
        self.assertIsNone(Device.by_name("no-such-device"))
        with self.assertRaises(IndexError):
            Device.get(Device.count())
        with self.assertRaisesRegex(GGMLError, "no device named"):
            Backend.init_by_name("no-such-device")

    def test_backend_initializes_by_device_name(self):
        Backend.load_all()
        device = Device.by_type(DeviceType.CPU)

        with Backend.init_by_name(device.name) as backend:
            self.assertTrue(backend.is_cpu)


if __name__ == "__main__":
    unittest.main()

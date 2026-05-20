import unittest

from onnx9000_profiler import Profiler


class TestProfiler(unittest.TestCase):
    def test_profiler_run(self):
        profiler = Profiler("fake_model.onnx")
        self.assertEqual(profiler.get_peak_memory(), 0.0)
        profiler.run()
        self.assertEqual(profiler.get_peak_memory(), 42.5)


if __name__ == "__main__":
    unittest.main()

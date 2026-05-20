import subprocess
import unittest


class TestProfilerCmd(unittest.TestCase):
    def test_profiler_cmd(self):
        res = subprocess.run(
            ["uv", "run", "onnx9000", "profiler", "fake.onnx"], capture_output=True, text=True
        )
        self.assertIn("Running profiler", res.stdout)
        self.assertIn("Peak Memory", res.stdout)


if __name__ == "__main__":
    unittest.main()

"""Unit tests and verification suite for profiler cmd functionality."""

import subprocess
import unittest


class TestProfilerCmd(unittest.TestCase):
    """Represents test profiler cmd inheriting from unittest.TestCase, supporting operations such as test_profiler_cmd."""

    def test_profiler_cmd(self) -> None:
        """Verify that profiler cmd behaves as expected."""
        res = subprocess.run(
            ["uv", "run", "onnx9000", "profiler", "fake.onnx"],
            capture_output=True,
            text=True,
        )
        self.assertIn("Running profiler", res.stdout)
        self.assertIn("Peak Memory", res.stdout)


if __name__ == "__main__":
    unittest.main()

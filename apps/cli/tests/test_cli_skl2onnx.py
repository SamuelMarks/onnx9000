"""Unit tests and verification suite for cli skl2onnx functionality."""

import subprocess
import unittest


class TestCliSKL2ONNX(unittest.TestCase):
    """Represents test cli skl2 onnx inheriting from unittest.TestCase, supporting operations such as test_skl2onnx."""

    def test_skl2onnx(self) -> None:
        """Verify that skl2onnx behaves as expected."""
        res = subprocess.run(
            ["uv", "run", "onnx9000", "skl2onnx", "model"],
            capture_output=True,
            text=True,
        )
        self.assertIn("SKL2ONNX processed", res.stdout)


if __name__ == "__main__":
    unittest.main()

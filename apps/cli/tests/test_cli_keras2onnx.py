"""Unit tests and verification suite for cli keras2onnx functionality."""

import subprocess
import unittest


class TestCliKeras2ONNX(unittest.TestCase):
    """Represents test cli keras2 onnx inheriting from unittest.TestCase, supporting operations such as test_keras2onnx."""

    def test_keras2onnx(self) -> None:
        """Verify that keras2onnx behaves as expected."""
        res = subprocess.run(
            ["uv", "run", "onnx9000", "keras2onnx", "model"],
            capture_output=True,
            text=True,
        )
        self.assertIn("Keras2ONNX processed", res.stdout)


if __name__ == "__main__":
    unittest.main()

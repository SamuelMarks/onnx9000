"""Unit tests and verification suite for cli paddle2onnx functionality."""

import subprocess
import unittest


class TestCliPaddle2ONNX(unittest.TestCase):
    """Represents test cli paddle2 onnx inheriting from unittest.TestCase, supporting operations such as test_paddle2onnx."""

    def test_paddle2onnx(self) -> None:
        """Verify that paddle2onnx behaves as expected."""
        res = subprocess.run(
            ["uv", "run", "onnx9000", "paddle2onnx", "model"],
            capture_output=True,
            text=True,
        )
        self.assertIn("Paddle2ONNX processed", res.stdout)


if __name__ == "__main__":
    unittest.main()

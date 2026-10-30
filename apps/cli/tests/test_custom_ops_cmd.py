"""Unit tests and verification suite for custom ops cmd functionality."""

import subprocess
import unittest


class TestCustomOpsCmd(unittest.TestCase):
    """Represents test custom ops cmd inheriting from unittest.TestCase, supporting operations such as test_custom_ops_cmd."""

    def test_custom_ops_cmd(self) -> None:
        """Verify that custom ops cmd behaves as expected."""
        res = subprocess.run(
            ["uv", "run", "onnx9000", "custom-ops", "fake_ops.json"],
            capture_output=True,
            text=True,
        )
        self.assertIn("Registering custom ops", res.stdout)


if __name__ == "__main__":
    unittest.main()

"""Unit tests and verification suite for cli arena functionality."""

import subprocess
import unittest


class TestCliArena(unittest.TestCase):
    """Represents test cli arena inheriting from unittest.TestCase, supporting operations such as test_arena."""

    def test_arena(self) -> None:
        """Verify that arena behaves as expected."""
        res = subprocess.run(
            ["uv", "run", "onnx9000", "arena", "model"], capture_output=True, text=True
        )
        self.assertIn("Arena processed", res.stdout)


if __name__ == "__main__":
    unittest.main()
